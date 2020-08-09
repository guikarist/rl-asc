from argparse import ArgumentParser
from utils import query_yes_no

import GPUtil
import datetime
import os


def main():
    args = parser.parse_args()
    if not args.only_dqn and (args.lambda_ is None or args.margin is None or args.i_before is None):
        parser.error('--lambda, --margin and --alpha are required unless --only_dqn is present.')

    initial_max_load = 0.5
    initial_max_memory = 0.5

    if not args.only_dqn:
        modified_dqn_exps = [(x, y, z) for x in args.lambda_ for y in args.margin for z in args.i_before]
        num_exps_each_rep = len(modified_dqn_exps) if args.only_modified_dqn else len(modified_dqn_exps) + 1
    else:
        modified_dqn_exps = None
        num_exps_each_rep = 1
    no_enough_cards = False
    num_launched_reps = 0

    config = {
        'env': args.env,
        'num_steps': args.num_steps,
        'double_q': args.double_q,
        'dueling': args.dueling,
        'prioritized_replay': args.prioritized_replay
    }

    parent_directory = os.path.join(
        os.getcwd(),
        '-'.join([args.env, 'deepq', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
    )
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print('Logging to {}'.format(parent_directory))
    else:
        raise FileExistsError('The training directory already exists')

    for i in range(args.num_repeat_times):
        need_dqn_exp = not args.only_modified_dqn
        need_modified_dqn_exp = not args.only_dqn

        num_committed_exps = 0
        if modified_dqn_exps is not None:
            modified_dqn_exps_to_do = modified_dqn_exps.copy()
        else:
            modified_dqn_exps_to_do = None
        while num_committed_exps < num_exps_each_rep:
            new_available_gpu = GPUtil.getAvailable(
                order='first', limit=num_exps_each_rep, maxLoad=initial_max_load, maxMemory=initial_max_memory
            )
            if len(new_available_gpu) == 0:
                if query_yes_no('No enough cards for one repetition (maxLoad={:.3}, maxMemory={:.3}), '
                                'would you like to increase limit?'.format(initial_max_load, initial_max_memory)):
                    initial_max_load += 0.1
                    initial_max_memory += 0.1

                    continue
                else:
                    no_enough_cards = True
                    break

            if need_dqn_exp:
                execute_training('deepq', new_available_gpu.pop(), parent_directory, i, config)
                need_dqn_exp = False
                num_committed_exps += 1

            if need_modified_dqn_exp and modified_dqn_exps_to_do is not None:
                while len(new_available_gpu) > 0 and len(modified_dqn_exps_to_do) > 0:
                    exp = modified_dqn_exps_to_do.pop()
                    execute_training(
                        'modified_deepq', new_available_gpu.pop(), parent_directory, i, config, exp[0], exp[1], exp[2]
                    )
                    num_committed_exps += 1

        if no_enough_cards:
            break

        num_launched_reps += 1

    print('{} training repetition(s) successfully launched'.format(num_launched_reps))


dqn_template = 'CUDA_VISIBLE_DEVICES={gpu_card} ' \
               'OPENAI_LOG_FORMAT=log,csv,tensorboard ' \
               'python -m baselines.run ' \
               '--alg={alg} ' \
               '--env={env} ' \
               '--num_timesteps={num_steps} ' \
               '--log_path={log_path} ' \
               '--dueling={dueling} ' \
               '--prioritized_replay={prioritized_replay} ' \
               '--double_q={double_q} ' \
               '--print_freq=10 ' \
               '>/dev/null 2>&1 &'

modified_dqn_template = 'CUDA_VISIBLE_DEVICES={gpu_card} ' \
                        'OPENAI_LOG_FORMAT=log,csv,tensorboard ' \
                        'python -m baselines.run ' \
                        '--alg={alg} ' \
                        '--env={env} ' \
                        '--num_timesteps={num_steps} ' \
                        '--log_path={log_path} ' \
                        '--lambda_={lambda_} ' \
                        '--margin={margin} ' \
                        '--i_before={i_before} ' \
                        '--dueling={dueling} ' \
                        '--prioritized_replay={prioritized_replay} ' \
                        '--double_q={double_q} ' \
                        '--print_freq=10 ' \
                        '>/dev/null 2>&1 &'

parser = ArgumentParser()
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--lambda', dest='lambda_', metavar='LAMBDA', nargs='+', type=float, help='Hyper-parameter Lambda')
parser.add_argument('--margin', nargs='+', type=float, help='Hyper-parameter Margin')
parser.add_argument('--alpha', dest='i_before', nargs='+', type=int, help='Hyper-parameter Alpha')
parser.add_argument('--num_repeat_times', type=int, default=5, help='The number of repeat training times')
parser.add_argument('--double_q', action='store_true', help='Whether to run the Double-DQN version')
parser.add_argument('--dueling', action='store_true', help='Whether to run the Dueling DQN version')
parser.add_argument('--prioritized_replay', action='store_true', help='Whether to run DQN with PER')
gp = parser.add_mutually_exclusive_group()
gp.add_argument('--only_modified_dqn', action='store_true', help='Whether only to run modified dqn experiment')
gp.add_argument('--only_dqn', action='store_true', help='Whether only to run original dqn experiment')


def execute_training(alg, gpu_card, parent_directory, num_reps, config, lambda_=None, margin=None,
                     i_before=None):
    config = config.copy()
    config['alg'] = alg
    config['gpu_card'] = gpu_card
    if alg == 'deepq':
        config['log_path'] = os.path.join(parent_directory, '_'.join([config['env'], alg, str(num_reps)]))
        os.system(dqn_template.format(**config))
    elif alg == 'modified_deepq':
        config['log_path'] = os.path.join(
            parent_directory,
            '_'.join([
                config['env'], alg, str(lambda_), str(margin), str(i_before),
                str(num_reps)
            ])
        )
        config['lambda_'] = lambda_
        config['margin'] = margin
        config['i_before'] = i_before
        os.system(modified_dqn_template.format(**config))


if __name__ == '__main__':
    main()
