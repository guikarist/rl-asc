import tensorflow as tf
import baselines.common.tf_util as U


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def default_param_noise_filter(var):
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values, _ = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                          outputs=output_actions,
                          givens={update_eps_ph: -1.0, stochastic_ph: True},
                          updates=[update_eps_expr])

        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)

        return act, update_eps_ph


def build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None,
                               param_noise_filter_func=None):
    """Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    if param_noise_filter_func is None:
        param_noise_filter_func = default_param_noise_filter

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), name="update_param_noise_threshold")
        update_param_noise_scale_ph = tf.placeholder(tf.bool, (), name="update_param_noise_scale")
        reset_ph = tf.placeholder(tf.bool, (), name="reset")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01),
                                            trainable=False)
        param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05),
                                                trainable=False)

        # Unmodified Q.
        q_values, _ = q_func(observations_ph.get(), num_actions, scope="q_func")

        # Perturbable Q used for the actual rollout.
        q_values_perturbed = q_func(observations_ph.get(), num_actions, scope="perturbed_q_func")

        # We have to wrap this code into a function due to the way tf.cond() works. See
        # https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
        # a more detailed discussion.
        def perturb_vars(original_scope, perturbed_scope):
            all_vars = scope_vars(absolute_scope_name(original_scope))
            all_perturbed_vars = scope_vars(absolute_scope_name(perturbed_scope))
            assert len(all_vars) == len(all_perturbed_vars)
            perturb_ops = []
            for var, perturbed_var in zip(all_vars, all_perturbed_vars):
                if param_noise_filter_func(perturbed_var):
                    # Perturb this variable.
                    op = tf.assign(perturbed_var,
                                   var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=param_noise_scale))
                else:
                    # Do not perturb, just assign.
                    op = tf.assign(perturbed_var, var)
                perturb_ops.append(op)
            assert len(perturb_ops) == len(all_vars)
            return tf.group(*perturb_ops)

        # Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
        # of the network and measures the effect of that perturbation in action space. If the perturbation
        # is too big, reduce scale of perturbation, otherwise increase.
        q_values_adaptive = q_func(observations_ph.get(), num_actions, scope="adaptive_q_func")
        perturb_for_adaption = perturb_vars(original_scope="q_func", perturbed_scope="adaptive_q_func")
        kl = tf.reduce_sum(
            tf.nn.softmax(q_values) * (tf.log(tf.nn.softmax(q_values)) - tf.log(tf.nn.softmax(q_values_adaptive))),
            axis=-1
        )
        mean_kl = tf.reduce_mean(kl)

        def update_scale():
            with tf.control_dependencies([perturb_for_adaption]):
                update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                                            lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                                            lambda: param_noise_scale.assign(param_noise_scale / 1.01),
                                            )
            return update_scale_expr

        # Functionality to update the threshold for parameter space noise.
        update_param_noise_threshold_expr = param_noise_threshold.assign(tf.cond(update_param_noise_threshold_ph >= 0,
                                                                                 lambda: update_param_noise_threshold_ph,
                                                                                 lambda: param_noise_threshold))

        # Put everything together.
        deterministic_actions = tf.argmax(q_values_perturbed, axis=1)
        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        updates = [
            update_eps_expr,
            tf.cond(reset_ph, lambda: perturb_vars(original_scope="q_func", perturbed_scope="perturbed_q_func"),
                    lambda: tf.group(*[])),
            tf.cond(update_param_noise_scale_ph, lambda: update_scale(), lambda: tf.Variable(0., trainable=False)),
            update_param_noise_threshold_expr,
        ]
        _act = U.function(
            inputs=[observations_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph,
                    update_param_noise_scale_ph],
            outputs=output_actions,
            givens={update_eps_ph: -1.0, stochastic_ph: True, reset_ph: False, update_param_noise_threshold_ph: False,
                    update_param_noise_scale_ph: False},
            updates=updates)

        def act(ob, reset=False, update_param_noise_threshold=False, update_param_noise_scale=False, stochastic=True,
                update_eps=-1):
            return _act(ob, stochastic, update_eps, reset, update_param_noise_threshold, update_param_noise_scale)

        return act, update_eps_ph


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, lambda_=0.1, margin=0.1,
                gamma=1.0, double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None,
                exploration_final_eps=0.1, modified_part=None):
    if param_noise:
        act_f, update_eps_ph = build_act_with_param_noise(
            make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func
        )
    else:
        act_f, update_eps_ph = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_tmi_input = make_obs_ph("obs_tmi")
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        has_obs_tmi_mask_ph = tf.placeholder(tf.float32, [None], name="has_obs_tmi")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # f features of the last and next obs.
        _, f_features_tmi = q_func(obs_tmi_input.get(), num_actions, scope="q_func", reuse=True)
        _, f_features_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)

        # q network evaluation
        q_t, f_features_t = q_func(obs_t_input.get(), num_actions, scope="q_func",
                                   reuse=True)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1, _ = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net, _ = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        f_error_1 = tf.reduce_sum(tf.square(f_features_t - f_features_tp1), 1)
        f_error_2 = tf.reduce_sum(tf.square(f_features_tmi - f_features_tp1), 1)
        representation_loss = tf.reduce_mean(has_obs_tmi_mask_ph * tf.maximum(0., margin + f_error_1 - f_error_2))

        if modified_part is not None:
            if modified_part == 'before':
                loss = tf.cond(
                    tf.less(update_eps_ph - tf.Variable(exploration_final_eps), tf.Variable(0.0001)),
                    lambda: weighted_error,
                    lambda: weighted_error + lambda_ * representation_loss
                )
            elif modified_part == 'after':
                loss = tf.cond(
                    tf.less(update_eps_ph - tf.Variable(exploration_final_eps), tf.Variable(0.0001)),
                    lambda: weighted_error + lambda_ * representation_loss,
                    lambda: weighted_error
                )
            else:
                raise ValueError("Invalid value of 'modified_part'")
        else:
            loss = weighted_error + lambda_ * representation_loss

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(loss, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(loss, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_tmi_input,
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                has_obs_tmi_mask_ph,
                importance_weights_ph,
                update_eps_ph
            ],
            outputs=[td_error],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}
