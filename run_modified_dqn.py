from argparse import ArgumentParser

import GPUtil
import datetime
import sys
import os


def main():
    args = parser.parse_args()
    initial_max_load = 0.5
    initial_max_memory = 0.5

    modified_dqn_exps = [(x, y) for x in args.lambda_ for y in args.margin]
    num_exps_each_epoch = len(modified_dqn_exps) if args.only_modified_dqn else len(modified_dqn_exps) + 1
    no_enough_cards = False
    num_launched_epochs = 0

    config = {
        'env': args.env,
        'num_steps': args.num_steps
    }

    parent_directory = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print('Logging to {}'.format(parent_directory))
    else:
        raise FileExistsError('The training directory already exists')

    for i in range(args.num_epochs):
        need_dqn_exp = not args.only_modified_dqn
        num_committed_exps = 0
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

            while len(new_available_gpu) > 0 and len(modified_dqn_exps) > 0:
                exp = modified_dqn_exps.pop()
                execute_training(
                    'modified_deepq', new_available_gpu.pop(), parent_directory, i, config, exp[0], exp[1]
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
                        '--dueling=False ' \
                        '--prioritized_replay=False ' \
                        '--print_freq=10 ' \
                        '>/dev/null 2>&1 &'

parser = ArgumentParser()
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--lambda', dest='lambda_', metavar='LAMBDA', nargs='+', type=float, help='Hyper-parameter Lambda',
                    required=True)
parser.add_argument('--margin', nargs='+', type=float, help='Hyper-parameter Margin', required=True)
parser.add_argument('--num_epochs', type=int, default=5, help='The number of training epochs')
parser.add_argument('--print_freq', type=int, default=10, help='The frequency of printing logs')
parser.add_argument('--only_modified_dqn', action='store_true', help='Whether to run original dqn experiment or not')


def execute_training(alg, gpu_card, parent_directory, num_epoch, config, lambda_=None, margin=None):
    config = config.copy()
    config['alg'] = alg
    config['gpu_card'] = gpu_card
    if alg == 'deepq':
        config['log_path'] = os.path.join(parent_directory, '{}_{}'.format(alg, str(num_epoch)))
        os.system(dqn_template.format(**config))
    elif alg == 'modified_deepq':
        config['log_path'] = os.path.join(parent_directory, '{}_{}_{}_{}'.format(alg, lambda_, margin, str(num_epoch)))
        config['lambda_'] = lambda_
        config['margin'] = margin
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