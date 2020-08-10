import numpy as np
import tensorflow as tf
from baselines.modified_a2c.utils import conv, fc, conv_to_fc
from baselines.common.models import register


@register("modified_cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)

    return network_fn


@register("modified_cnn_v2")
def cnn_v2(**conv_kwargs):
    def network_fn(X):
        return nature_cnn_v2(X, **conv_kwargs)

    return network_fn


@register("modified_mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    def network_fn(X):
        h = tf.map_fn(tf.layers.flatten, X)

        def fc_func(x):
            return fc(x, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))

        def norm_func(x):
            return tf.contrib.layers.layer_norm(x, center=True, scale=True)

        for i in range(num_layers):
            h = tf.map_fn(fc_func, h)
            if layer_norm:
                h = tf.map_fn(norm_func, h)

            if i != num_layers - 1:
                h = tf.map_fn(activation, h)

        def f_features_func(x):
            return tf.sigmoid(tf.layers.flatten(x))

        f_features = tf.map_fn(f_features_func, h)
        h = tf.map_fn(activation, h)

        return h, f_features

    return network_fn


@register("modified_mlp_v2")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    def network_fn(X):
        h = tf.map_fn(tf.layers.flatten, X)

        def fc_func(x):
            return fc(x, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))

        def norm_func(x):
            return tf.contrib.layers.layer_norm(x, center=True, scale=True)

        for i in range(num_layers):
            h = tf.map_fn(fc_func, h)
            if layer_norm:
                h = tf.map_fn(norm_func, h)

            h = tf.map_fn(activation, h)

        f_features = tf.map_fn(tf.layers.flatten, h)

        return h, f_features

    return network_fn


def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu

    def h_func(x): return activ(conv(x, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                                     **conv_kwargs))

    h = tf.map_fn(h_func, scaled_images)

    def h2_func(x): return activ(conv(x, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))

    h2 = tf.map_fn(h2_func, h)

    def out_func(x): return activ(conv(x, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))

    out = tf.map_fn(out_func, h2)

    f_features = tf.map_fn(tf.layers.flatten, out)
    h3 = tf.map_fn(conv_to_fc, out)

    def h4_func(x): return activ(fc(x, 'fc1', nh=512, init_scale=np.sqrt(2)))

    h4 = tf.map_fn(h4_func, h3)

    return h4, f_features


def nature_cnn_v2(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    out = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(out)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))), tf.layers.flatten(out)
