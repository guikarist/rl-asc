def mujoco():
    return dict(
        nsteps=2500,
        value_network='copy'
    )

def atari():
    return dict(
        network='modified_cnn_v2'
    )
