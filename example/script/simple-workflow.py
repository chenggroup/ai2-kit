from ai2_kit.core.executor import HpcExecutor
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.util import list_split
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.checkpoint import set_checkpoint_file
from typing import List
import asyncio
import os

# Define remote executable functions
def __export_remote_functions():

    def process_input(n: int, base_dir: str):
        import random

        task_dirs = []
        for i in range(n):
            task_dir = os.path.join(base_dir, str(i))
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, 'input'), 'w') as f:
                f.write(str(random.randint(0, 100)))
            task_dirs.append(task_dir)
        return task_dirs

    def process_output(task_dirs: List[str]):
        outputs = []
        for task_dir in task_dirs:
            with open(os.path.join(task_dir, 'output'), 'r') as f:
                outputs.append(int(f.read().strip()))
        return sum(outputs)

    return (process_input, process_output)


(process_input, process_output) = __export_remote_functions()


# Define workflow
async def workflow(n: int, path_prefix: str, executor: HpcExecutor, script_header: str, concurrency: int = 5):
    # create base_dir to store input and output
    # it is suggested to use a unique path_prefix for each workflow
    base_dir = os.path.join(executor.work_dir, path_prefix)
    executor.mkdir(base_dir)

    # run pre process
    task_dirs = executor.run_python_fn(process_input)(n, base_dir=base_dir)

    # build commands to calculate square and save to output
    steps = [BashStep(cmd='read num < input; echo $(( num * num )) > output',
                      cwd=task_dir) for task_dir in task_dirs]
    # create script according to concurrency limit and submit
    jobs = []
    for group in list_split(steps, concurrency):
        script = BashScript(
            template=BashTemplate(header=script_header),
            steps=group,
        )
        job = executor.submit(script.render(), cwd=base_dir)
        jobs.append(job)

    # wait for all jobs to complete
    await gather_jobs(jobs, max_tries=2, raise_error=True)

    # post process
    result = executor.run_python_fn(process_output)(task_dirs)
    print(result)


def main():
    # config and initialize executor
    executor = HpcExecutor.from_config({
        'ssh': {
            'host': 'user01@login-node',
            'gateway': {  # Optional, use it when you have to use just host to connect to the cluster
                'host': 'user01@jump-host-node',
            }
        },
        'queue_system': {
            'slurm': {}  # Specify queue system
        },
        'work_dir': '/home/user01/work_dir',  # Specify working directory
        'python_cmd': '/home/user01/conda/env/py39/bin/python',  # Specify python command

    }, 'cheng-lab')
    executor.init()

    script_header = '\n'.join([
        '#SBATCH --job-name=square',
        '#SBATCH -N 1',
        '#SBATCH -partition=small',
    ])

    # set checkpoint file so that the workflow can be resumed
    set_checkpoint_file('square-sum-workflow.ckpt')
    # run workflow
    asyncio.run(workflow(n=10,
                         path_prefix='square-sum-workflow',
                         executor=executor,
                         script_header=script_header,
                         concurrency=5,
                         ))


if __name__ == '__main__':
    main()
