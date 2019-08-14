import time
import functools
import tensorflow as tf
import numpy as np

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.modified_a2c.policies import build_policy

from baselines.modified_a2c.utils import Scheduler, find_trainable_variables
from baselines.modified_a2c.runner import Runner
from baselines.ppo2.ppo2 import safemean
from collections import deque

from tensorflow import losses


class Model(object):
    def __init__(self, policy, env, nsteps,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, lambda_=0.1, margin=0.1, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs * nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        # Representation loss
        f_features_tmi = train_model.f_features[0]
        f_features_t = train_model.f_features[1]
        f_features_tp1 = train_model.f_features[2]
        f_error_1 = tf.reduce_sum(tf.square(f_features_t - f_features_tp1), 1)
        f_error_2 = tf.reduce_sum(tf.square(f_features_tmi - f_features_tp1), 1)
        representation_loss = tf.reduce_mean(tf.maximum(0., margin + f_error_1 - f_error_2))
        has_obs_tmi_mask_ph = tf.placeholder(tf.float32, None, name="has_obs_tmi")

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + has_obs_tmi_mask_ph * lambda_ * representation_loss

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs_tmi, obs, obs_tp1, states, rewards, masks, actions, values, has_obs_tmi):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {
                train_model.Xs: [obs_tmi, obs, obs_tp1],
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr,
                has_obs_tmi_mask_ph: has_obs_tmi
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _representation_loss, _ = sess.run(
                [pg_loss, vf_loss, entropy, representation_loss, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, _representation_loss

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
        network,
        env,
        seed=None,
        nsteps=5,
        total_timesteps=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        alpha=0.99,
        gamma=0.99,
        lambda_=0.1,
        margin=0.1,
        i_before=1,
        log_interval=100,
        load_path=None,
        **network_kwargs):
    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule, lambda_=lambda_, margin=margin)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    nbatch = nenvs * nsteps

    # Start total timer
    tstart = time.time()

    obses_before: deque[np.ndarray] = deque(maxlen=i_before + 1)

    for update in range(1, total_timesteps // nbatch + 1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        if len(obses_before) < obses_before.maxlen:
            obses_before.append(obs)
            left_obs = np.zeros_like(obs)
            right_obs = np.zeros_like(obs)
            has_obs_tmi = False
        else:
            left_obs = obses_before.popleft()
            right_obs = obses_before.pop()
            has_obs_tmi = True

        policy_loss, value_loss, policy_entropy, repr_loss = model.train(left_obs, right_obs, obs, states, rewards, masks, actions,
                                                              values, float(has_obs_tmi))
        if has_obs_tmi:
            obses_before.append(right_obs)
            obses_before.append(obs)

        nseconds = time.time() - tstart

        # Calculate the fps (frame per second)
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("repr_loss", float(repr_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
    return model
