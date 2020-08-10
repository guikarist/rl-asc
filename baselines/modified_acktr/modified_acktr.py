import os.path as osp
import time
import functools
import tensorflow as tf
import numpy as np
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.modified_a2c.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables

from baselines.modified_a2c.runner import Runner
from baselines.modified_a2c.utils import Scheduler, find_trainable_variables
from baselines.modified_a2c.modified_a2c import shift
from baselines.modified_acktr import kfac
from baselines.ppo2.ppo2 import safemean
from collections import deque


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 lambda_=0.1, margin=0.1, kfac_clip=0.001, lrschedule='linear', is_async=True):

        self.sess = sess = get_session()
        nbatch = nenvs * nsteps
        with tf.variable_scope('acktr_model', reuse=tf.AUTO_REUSE):
            self.model = step_model = policy(nenvs, 1, sess=sess)
            self.model2 = train_model = policy(nenvs*nsteps, nsteps, sess=sess)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        self.logits = train_model.pi

        # Representation loss
        f_features_tmi = train_model.f_features[0]
        f_features_t = train_model.f_features[1]
        f_features_tp1 = train_model.f_features[2]
        f_error_1 = tf.reduce_sum(tf.square(f_features_t - f_features_tp1), 1)
        f_error_2 = tf.reduce_sum(tf.square(f_features_tmi - f_features_tp1), 1)

        has_triplet_mask_ph = tf.placeholder(tf.float32, [None], name="has_triplet")
        representation_loss = tf.reduce_mean(has_triplet_mask_ph * tf.maximum(0., margin + f_error_1 - f_error_2))
        delta_d = tf.reduce_mean(has_triplet_mask_ph * (f_error_1 - f_error_2))

        ##training loss
        pg_loss = tf.reduce_mean(ADV*neglogpac)
        entropy = tf.reduce_mean(train_model.pd.entropy())
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.losses.mean_squared_error(tf.squeeze(train_model.vf), R)
        train_loss = pg_loss + vf_coef * vf_loss + lambda_ * representation_loss

        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params=params = find_trainable_variables("acktr_model")

        self.grads_check = grads = tf.gradients(train_loss,params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,\
                momentum=0.9, kfac_update=1, epsilon=0.01,\
                stats_decay=0.99, is_async=is_async, cold_iter=10, max_grad_norm=max_grad_norm)

            # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs_tmi, obs, obs_tp1, states, rewards, masks, actions, values, has_obs_tmi):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {
                train_model.Xs: [obs_tmi, obs, obs_tp1],
                A: actions,
                ADV: advs,
                R: rewards,
                PG_LR: cur_lr,
                VF_LR: cur_lr,
                has_triplet_mask_ph: has_obs_tmi
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _representation_loss, _delta_d, _ = sess.run(
                [pg_loss, vf_loss, entropy, representation_loss, delta_d, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, _representation_loss, _delta_d


        self.train = train
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)


def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=100, nprocs=32, nsteps=20,
          ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5, lambda_=0.1, margin=0.1,
          i_before=1, kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True,
          **network_kwargs):
    set_global_seeds(seed)

    if network == 'modified_cnn_v2':
        network_kwargs['one_dim_bias'] = True

    policy = build_policy(env, network, **network_kwargs)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda : Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule, is_async=is_async, lambda_=lambda_, margin=margin)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    if load_path is not None:
        model.load(load_path)

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    if is_async:
        enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)
    else:
        enqueue_threads = []

    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        left_obs = []
        has_left_obs = []
        for start in range(0, nbatch, nsteps):
            result, has_obs = shift(obs[start: start + nsteps], i_before, fill_value=np.zeros_like(obs[0]))
            left_obs.append(result)
            has_left_obs.append(has_obs)
        left_obs = np.vstack(left_obs)
        has_left_obs = np.hstack(has_left_obs)

        right_obs = []
        has_right_obs = []
        for start in range(0, nbatch, nsteps):
            result, has_obs = shift(obs[start: start + nsteps], -1, fill_value=np.zeros_like(obs[0]))
            right_obs.append(result)
            has_right_obs.append(has_obs)
        right_obs = np.vstack(right_obs)
        has_right_obs = np.hstack(has_right_obs)

        has_triplet = np.logical_and(has_left_obs, has_right_obs).astype(float)

        policy_loss, value_loss, policy_entropy, repr_loss, delta_d = model.train(left_obs, obs, right_obs, states, rewards,
                                                                                  masks, actions,
                                                                                  values, has_triplet)
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("repr_loss", float(repr_loss))
            logger.record_tabular("delta_d", float(delta_d))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    return model
