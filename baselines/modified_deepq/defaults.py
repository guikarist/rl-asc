def atari():
    return dict(
        network='modified_conv_only',
        lr=1e-4,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        lambda_=0.1,
        margin=0.1,
        i_before=1,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=False,
        double_q=False,
        modified_part=None
    )


def retro():
    return atari()
