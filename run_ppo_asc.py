from argparse import ArgumentParser

import GPUtil
import datetime
import sys
import os


def main():
    args = parser.parse_args()
    if not args.only_ppo and (args.lambda_ is None or args.margin is None or args.i_before is None):
        parser.error('--lambda, --margin and --alpha are required unless --only_ppo is present.')

    initial_max_load = 0.5
    initial_max_memory = 0.5

    if not args.only_ppo:
        modified_ppo_exps = [(x, y, z) for x in args.lambda_ for y in args.margin for z in args.i_before]
        num_exps_each_rep = len(modified_ppo_exps) if args.only_modified_ppo else len(modified_ppo_exps) + 1
    else:
        modified_ppo_exps = None
        num_exps_each_rep = 1
    no_enough_cards = False
    num_launched_reps = 0

    config = {
        'env': args.env,
        'num_steps': args.num_steps
    }

    parent_directory = os.path.join(
        os.getcwd(),
        '-'.join([args.env, 'ppo', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
    )
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print('Logging to {}'.format(parent_directory))
    else:
        raise FileExistsError('The training directory already exists')

    for i in range(args.num_repeat_times):
        need_ppo_exp = not args.only_modified_ppo
        need_modified_ppo_exp = not args.only_ppo

        num_committed_exps = 0
        if modified_ppo_exps is not None:
            modified_ppo_exps_to_do = modified_ppo_exps.copy()
        else:
            modified_ppo_exps_to_do = None
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

            if need_ppo_exp:
                execute_training('ppo', new_available_gpu.pop(), parent_directory, i, config)
                need_ppo_exp = False
                num_committed_exps += 1

            if need_modified_ppo_exp and modified_ppo_exps_to_do is not None:
                while len(new_available_gpu) > 0 and len(modified_ppo_exps_to_do) > 0:
                    exp = modified_ppo_exps_to_do.pop()
                    execute_training(
                        'modified_ppo', new_available_gpu.pop(), parent_directory, i, config, exp[0], exp[1], exp[2]
                    )
                    num_committed_exps += 1

        if no_enough_cards:
            break

        num_launched_reps += 1

    print('{} training repetition(s) successfully launched'.format(num_launched_reps))


ppo_template = 'CUDA_VISIBLE_DEVICES={gpu_card} ' \
               'OPENAI_LOG_FORMAT=log,csv,tensorboard ' \
               'python -m baselines.run ' \
               '--alg={alg} ' \
               '--env={env} ' \
               '--num_timesteps={num_steps} ' \
               '--log_path={log_path} ' \
               '--log_interval=10 ' \
               '>/dev/null 2>&1 &'

modified_ppo_template = 'CUDA_VISIBLE_DEVICES={gpu_card} ' \
                        'OPENAI_LOG_FORMAT=log,csv,tensorboard ' \
                        'python -m baselines.run ' \
                        '--alg={alg} ' \
                        '--env={env} ' \
                        '--num_timesteps={num_steps} ' \
                        '--log_path={log_path} ' \
                        '--lambda_={lambda_} ' \
                        '--margin={margin} ' \
                        '--i_before={i_before} ' \
                        '--log_interval=10 ' \
                        '>/dev/null 2>&1 &'

parser = ArgumentParser()
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--lambda', dest='lambda_', metavar='LAMBDA', nargs='+', type=float, help='Hyper-parameter Lambda')
parser.add_argument('--margin', nargs='+', type=float, help='Hyper-parameter Margin')
parser.add_argument('--alpha', dest='i_before', nargs='+', type=int, help='Hyper-parameter Alpha')
parser.add_argument('--num_repeat_times', type=int, default=5, help='The number of repeat training times')
gp = parser.add_mutually_exclusive_group()
gp.add_argument('--only_modified_ppo', action='store_true', help='Whether only to run modified ppo experiment')
gp.add_argument('--only_ppo', action='store_true', help='Whether only to run original ppo experiment')


def execute_training(alg, gpu_card, parent_directory, num_reps, config, lambda_=None, margin=None,
                     i_before=None):
    config = config.copy()
    config['alg'] = alg
    config['gpu_card'] = gpu_card
    if alg == 'ppo':
        config['log_path'] = os.path.join(parent_directory, '_'.join([config['env'], 'ppo', str(num_reps)]))
        config['alg'] = 'ppo2'
        os.system(ppo_template.format(**config))
    elif alg == 'modified_ppo':
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
        os.system(modified_ppo_template.format(**config))


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
