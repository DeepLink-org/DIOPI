import argparse
import subprocess
import os
import logging
import multiprocessing
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.conformance.model_list import model_list    # noqa

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
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            logger.info(line.strip())
        process.wait()
        return process.returncode

    processes = []

    for i, cmd in enumerate(commands):
        process = multiprocessing.Process(target=execute_command, args=(cmd,))
        processes.append(process)
        process.start()
        if i % 5 == 0:
            time.sleep(30)

    for process in processes:
        process.join()


def gen_data(partition, device_type, device_num, use_db, use_slurm):
    commands = []
    for model in model_list:
        cmd = ''
        if use_slurm:
            cmd += f'srun --job-name {model}_gen_data -p {partition} --gres={device_type}:{device_num} '
        cmd += f'python main.py --mode gen_data --model_name {model}'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path}'
        commands.append(cmd)

    execute_commands(commands)


def gen_case(partition, use_db, use_slurm impl_folder):
    commands = []
    for model in model_list:
        cmd = ''
        if use_slurm:
            cmd += f'srun --job-name {model}_gen_case -p {partition} '
        cmd += f'python main.py --mode gen_case --model_name {model} --case_output_dir ./gencases/{model}_case'
        if impl_folder:
            cmd += f' --impl_folder {impl_folder}'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path}'
        commands.append(cmd)

    execute_commands(commands)


def run_test(partition, device_type, device_num, use_db, pytest_args, use_slurm):
    commands = []
    for model in model_list:
        cmd = ''
        if use_slurm:
            cmd += f'srun --job-name {model}_run_test -p {partition} --gres={device_type}:{device_num} '
        cmd += f'python main.py --mode run_test --test_cases_path ./gencases/{model}_case'
        if pytest_args:
            cmd += f' --pytest_args "{pytest_args}"'
        if use_db:
            db_path = f'sqlite:///./cache/{model}_testrecord.db'
            cmd += f' --use_db --db_path {db_path} --excel_path logs/{model}.xlsx'
        cmd += f' 2>&1 | tee logs/{model}.log'
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
        "--use_slurm", action="store_true", help="use slurm to run test"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="running mode, available options: gen_data, run_test and utest",
    )
    
    parser.add_argument(
        "--impl_folder", type=str, default="", help="impl_folder"
    )

    parser.add_argument("--pytest_args", type=str, help="pytest args", default='')

    args = parser.parse_args()

    if args.mode == 'gen_data':
        gen_data(args.partition, args.device_type, args.device_num, args.use_db, args.use_slurm)
    elif args.mode == 'gen_case':
        gen_case(args.partition, args.use_db, args.use_slurm, args.impl_folder)
    elif args.mode == 'run_test':
        run_test(args.partition, args.device_type, args.device_num, args.use_db, args.pytest_args, args.use_slurm)
