import argparse
import subprocess
import os
import logging
import subprocess
import multiprocessing

from python.conformance.model_list import model_list

os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python'))


logger = logging.getLogger('Conformance')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[PID:%(process)d] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def execute_commands(commands):
    def execute_command(command):
        logger.info(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            logger.info(line.strip())
        process.wait()
        return process.returncode

    processes = []

    for cmd in commands:
        process = multiprocessing.Process(target=execute_command, args=(cmd,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def gen_data(partition, device_type, device_num, use_db):
    commands = []
    for model in model_list[:2]:
        cmd = f'srun --job-name {model}_gen_data -p {partition} --gres={device_type}:{device_num} python main.py --mode gen_data --model_name {model}'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path}'
        commands.append(cmd)

    execute_commands(commands)


def gen_case(partition, use_db):
    commands = []
    for model in model_list[:2]:
        cmd = f'srun --job-name {model}_gen_case -p {partition} python main.py --mode gen_case --model_name {model} --case_output_dir ./gencases/{model}_case'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path}'
        commands.append(cmd)

    execute_commands(commands)


def run_test(partition, device_type, device_num, use_db, pytest_args):
    commands = []
    for model in model_list[:2]:
        cmd = f'srun --job-name {model}_run_test -p {partition} --gres={device_type}:{device_num} python main.py --mode run_test --file_or_dir ./gencases/{model}_case --pytest_args "{pytest_args}"'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path}'
        commands.append(cmd)

    execute_commands(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate configuration code.")

    parser.add_argument('--partition', '-p', type=str, default='pat_rd',
                        help='slurm partition')

    parser.add_argument('--device_type', type=str, default='gpu',
                        help='device type')

    parser.add_argument('--device_num', type=str, default='1',
                        help='device num')

    parser.add_argument(
        "--use_db", action="store_true", help="use database to save test data"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="running mode, available options: gen_data, run_test and utest",
    )

    parser.add_argument("--pytest_args", type=str, help="pytest args", default='')

    args = parser.parse_args()

    if args.mode == 'gen_data':
        gen_data(args.partition, args.device_type, args.device_num, args.use_db)
    elif args.mode == 'gen_case':
        gen_case(args.partition, args.use_db)
    elif args.mode == 'run_test':
        run_test(args.partition, args.device_type, args.device_num, args.use_db, args.pytest_args)
