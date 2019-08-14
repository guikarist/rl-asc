from argparse import ArgumentParser

import GPUtil
import datetime
import sys
import os


def main():
    args = parser.parse_args()
    if not args.only_dqn and (args.lambda_ is None or args.margin is None or args.i_before is None):
        parser.error('--lambda, --margin and --i are required unless --only_dqn is present.')

    initial_max_load = 0.5
    initial_max_memory = 0.5

    if not args.only_dqn:
        modified_dqn_exps = [(x, y, z) for x in args.lambda_ for y in args.margin for z in args.i_before]
        num_exps_each_epoch = len(modified_dqn_exps) if args.only_modified_dqn else len(modified_dqn_exps) + 1
    else:
        modified_dqn_exps = None
        num_exps_each_epoch = 1
    no_enough_cards = False
    num_launched_epochs = 0

    config = {
        'env': args.env,
        'num_steps': args.num_steps,
        'modified_part': args.modified_part
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

    for i in range(args.num_epochs):
        need_dqn_exp = not args.only_modified_dqn
        need_modified_dqn_exp = not args.only_dqn

        num_committed_exps = 0
        if modified_dqn_exps is not None:
            modified_dqn_exps_to_do = modified_dqn_exps.copy()
        else:
            modified_dqn_exps_to_do = None
        while num_committed_exps < num_exps_each_epoch:
            new_available_gpu = GPUtil.getAvailable(
                order='first', limit=num_exps_each_epoch, maxLoad=initial_max_load, maxMemory=initial_max_memory
            )
            if len(new_available_gpu) == 0:
                if query_yes_no('No enough cards for one epoch (maxLoad={:.3}, maxMemory={:.3}), '
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

        num_launched_epochs += 1

    print('{} training epochs has been successfully launched'.format(num_launched_epochs))


dqn_template = 'CUDA_VISIBLE_DEVICES={gpu_card} ' \
               'OPENAI_LOG_FORMAT=log,csv,tensorboard ' \
               'python -m baselines.run ' \
               '--alg={alg} ' \
               '--env={env} ' \
               '--num_timesteps={num_steps} ' \
               '--log_path={log_path} ' \
               '--dueling=False ' \
               '--prioritized_replay=False ' \
               '--double_q=False ' \
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
                        '--dueling=False ' \
                        '--prioritized_replay=False ' \
                        '--double_q=False ' \
                        '--print_freq=10 ' \
                        '--modified_part={modified_part} ' \
                        '>/dev/null 2>&1 &'

parser = ArgumentParser()
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--lambda', dest='lambda_', metavar='LAMBDA', nargs='+', type=float, help='Hyper-parameter Lambda')
parser.add_argument('--margin', nargs='+', type=float, help='Hyper-parameter Margin')
parser.add_argument('--i', dest='i_before', nargs='+', type=int, help='Hyper-parameter i')
parser.add_argument('--num_epochs', type=int, default=5, help='The number of training epochs')
parser.add_argument('--print_freq', type=int, default=10, help='The frequency of printing logs')
parser.add_argument('--modified_part', type=str, default=None, choices=['before', 'after'],
                    help='The modified part of the whole learning process')
gp = parser.add_mutually_exclusive_group()
gp.add_argument('--only_modified_dqn', action='store_true', help='Whether only to run modified dqn experiment')
gp.add_argument('--only_dqn', action='store_true', help='Whether only to run original dqn experiment')


def execute_training(alg, gpu_card, parent_directory, num_epoch, config, lambda_=None, margin=None,
                     i_before=None):
    config = config.copy()
    config['alg'] = alg
    config['gpu_card'] = gpu_card
    if alg == 'deepq':
        config['log_path'] = os.path.join(parent_directory, '_'.join([config['env'], alg, str(num_epoch)]))
        os.system(dqn_template.format(**config))
    elif alg == 'modified_deepq':
        config['log_path'] = os.path.join(
            parent_directory,
            '_'.join([
                config['env'], alg, str(lambda_), str(margin), str(i_before), str(config['modified_part']),
                str(num_epoch)
            ])
        )
        config['lambda_'] = lambda_
        config['margin'] = margin
        config['i_before'] = i_before
        os.system(modified_dqn_template.format(**config))


def query_yes_no(question, default='yes'):
    """Copied from https://stackoverflow.com/a/3041990
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':
    main()
