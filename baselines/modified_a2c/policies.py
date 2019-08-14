from baselines.common import tf_util
from baselines.modified_a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder
from baselines.common.tf_util import adjust_shape
from baselines.common.models import get_network_builder
from gym.spaces import Box

import gym
import numpy as np
import tensorflow as tf


class ModifiedPolicyWithValue(object):
    def __init__(self, env, observations, latents, f_features, estimate_q=False, vf_latents=None, sess=None, **tensors):
        self.Xs = observations
        self.X = observations[1]
        self.f_features = f_features

        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latents = vf_latents if vf_latents is not None else latents

        vf_latents = tf.layers.flatten(vf_latents[1])
        latents = tf.layers.flatten(latents[1])

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latents, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latents, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latents, 'vf', 1)
            self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        observation = np.array([np.zeros_like(observation), observation, np.zeros_like(observation)])
        sess = self.sess
        feed_dict = {self.Xs: adjust_shape(self.Xs, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation,
                                              **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_policy(env, policy_network='', value_network=None, normalize_observations=False, estimate_q=False,
                 **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None):
        ob_space = env.observation_space

        Xs = tf.stack([observation_placeholder(ob_space, batch_size=nbatch)] * 3)

        extra_tensors = {}

        if normalize_observations and Xs.dtype == tf.float32:
            raise NotImplementedError
        else:
            encoded_x = Xs

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            # policy_latent, f_feature = policy_network(encoded_x)
            policy_latents, f_features = policy_network(encoded_x)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latents = policy_latents
        else:
            raise NotImplementedError

        policy = ModifiedPolicyWithValue(
            env=env,
            observations=Xs,
            latents=policy_latents,
            f_features=f_features,
            vf_latents=vf_latents,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def encode_observation(ob_space, placeholder):
    '''
    Encode input in the way that is appropriate to the observation space

    Parameters:
    ----------

    ob_space: gym.Space             observation space

    placeholder: tf.placeholder     observation input placeholder
    '''
    if isinstance(ob_space, Box):
        return tf.to_float(placeholder)
    else:
        raise NotImplementedError
