from baselines.modified_deepq import models  # noqa
from baselines.modified_deepq.build_graph import build_act, build_train  # noqa
from baselines.modified_deepq.modified_deepq import learn, load_act  # noqa
from baselines.modified_deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
