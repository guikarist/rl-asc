from baselines.common.models import register

import tensorflow as tf
import tensorflow.contrib.layers as layers


@register("modified_conv_only")
def modified_conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for i, (num_outputs, kernel_size, stride) in enumerate(convs):
                if i != len(convs) - 1:
                    out = tf.contrib.layers.convolution2d(out,
                                                          num_outputs=num_outputs,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          activation_fn=tf.nn.relu,
                                                          **conv_kwargs)
                else:
                    out = tf.contrib.layers.convolution2d(out,
                                                          num_outputs=num_outputs,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          activation_fn=None,
                                                          **conv_kwargs)
                    f_features = tf.sigmoid(tf.layers.flatten(out))
                    out = tf.nn.relu(out)

        return out, f_features

    return network_fn


@register("modified_conv_only_v2")
def modified_conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for i, (num_outputs, kernel_size, stride) in enumerate(convs):
                if i != len(convs) - 1:
                    out = tf.contrib.layers.convolution2d(out,
                                                          num_outputs=num_outputs,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          activation_fn=tf.nn.relu,
                                                          **conv_kwargs)
                else:
                    out = tf.contrib.layers.convolution2d(out,
                                                          num_outputs=num_outputs,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          activation_fn=None,
                                                          **conv_kwargs)
                    out = tf.sigmoid(out)
                    f_features = tf.layers.flatten(out)

        return out, f_features

    return network_fn


def build_q_func(network, hiddens=[256], dueling=False, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent, f_features = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out, f_features

    return q_func_builder
