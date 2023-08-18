# AI<sup>2</sup>-Kit HPC Executor

## Introduction
`ai2-kit` implements a lightweight HPC executor for submitting and managing jobs on HPC clusters. Compared with other HPC schedulers or workflow frameworks that support HPC (such as DpDispatcher, parsl, DFlow, etc.), `ai2-kit HPC executor` has the following characteristics:

* SSH remote execution and local execution
* Connecting to HPC clusters through jump servers
* Remote execution of Python functions
* Executing simple commands or functions directly on the login node
* More efficient and stable job status polling mechanism
* State recovery mechanism based on Checkpoint
* Synchronous or asynchronous waiting for job execution
* Easy to customize or integrate with other frameworks

If you encounter the following problems when using other solutions, you may wish to try `ai2-kit HPC executor`:
* Too high learning cost
* Unstable connection
* Need to frequently copy data to the local for processing and then submit it back to the cluster for execution

Currently, `ai2-kit HPC executor` only supports the Slurm job system. The support for other job systems depends on actual needs. Welcome to submit Issues or PR.

If you need a more powerful workflow engine, it is recommended to try [DFlow](https://github.com/dptech-corp/dflow), [covalent](https://github.com/AgnostiqHQ/covalent.git), [parsl](https://github.com/Parsl/parsl), [redun](https://github.com/insitro/redun)

## Usage

### Basic Usage
There are two ways to initialize `ai2-kit HPC executor`, using a dictionary configuration or a `Pydantic` object. The configuration items of the two methods are exactly the same. The former is more suitable for direct use, and the latter is more suitable for integration with other frameworks. The following is an example of using a dictionary for configuration.

```python
from ai2_kit.core.executor import HpcExecutor

executor = HpcExecutor.from_config({
    'ssh': {  # Specify ssh connection information, if omitted, local execution is used
        'host': 'user01@hpc-login01',
        'gateway': {  # If you need to connect through a jump server, you can specify this configuration (optional)
          'host': 'user01@jump-host',  
        }
    },
    'queue_system': {
        'slurm': {}  # Specify the job system as Slurm
    },
    'work_dir': '/home/user01/ai2-kit/work_dir',  # Specify the working directory
    'python_cmd': '/home/user01/conda/env/py39/bin/python',  # Specify the Python interpreter
}, 'cheng-lab')  # Specify the cluster name (optional)

executor.init()  # Initialize the executor
executor.run('echo "hello world"')  # Execute command on login node
```

The above example completes the following tasks:
1. Instantiate a `HpcExecutor` object
2. Initialize the `HpcExecutor` object
3. Execute the command `echo "hello world"` on the login node

### Remote execution of Python functions

The usual mode of executing complex computing tasks on HPC is:
* Prepare the input data of the computing task on the login node through the command line or Python
* Submit the job to the queue and wait
* After the job is completed, use the command line or Python on the login node to process the output data of the computing task

In order to meet the above mode, in addition to providing the `run` interface for executing commands on the login node, `ai2-kit HPC executor` also provides the `run_python_fn` interface for directly running Python functions on the login node.

```python
def add(a, b):
    return a + b

result = executor.run_python_fn(add)(1, 2)  # run on login node
```

Note that in order to execute local Python functions on remote nodes, the following conditions must be met:
* The main version of the local Python environment and the remote Python environment must be consistent (such as 3.8.x)
  * The configuration of the remote Python environment can be specified through the `python_cmd` parameter
* The parameters and return values of the function must be serializable (cannot contain unserializable objects such as locks and file handles)
* The software packages on which the function depends must exist in the Python environment of the login node
  * For example, suppose the function executed remotely uses the `numpy` package, then the `numpy` package must exist in the Python environment of the login node
* If the function depends on other **locally implemented methods or classes**, these methods and classes need to be defined in a special way, otherwise `ModuleNotFoundError` will appear remotely
  * This is a limitation of cloudpickle, see: [1](https://stackoverflow.com/a/75293155/3099733)


### Submit jobs

`ai2-kit HPC executor` provides the `submit` interface for submitting jobs to HPC clusters. The following is an example

```python
script = '''\
#! /bin/bash
#SBATCH --job-name=demo
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=cpu-small
#SBATCH --mem=1G

echo "hello world" > output.txt
'''

job = executor.submit(script)  # submit job
``` 

Except for the method of directly writing scripts mentioned above, you can also generate scripts through the tool class provided by `ai2-kit`, as shown below:

```python 
from ai2_kit.core.script import BashScript, BashStep, BashTemplate

header = '''\
#SBATCH --job-name=demo
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=cpu-small
#SBATCH --mem=1G
'''

script = BashScript(
    template=BashTemplate(
        header=header,
    ),
    steps=[
        BashStep(cmd='echo "hello world"'),
    ]
)

job = executor.submit(script.render(), cwd='/path/to/cwd') 
```

Besides, you can specify additional parameters when submitting jobs to meet different needs, such as setting the execution directory and the checkpoint file for error recovery.

### Wait for job completion

There are two ways to wait for the completion of the submitted task, synchronous and asynchronous. The following is an example:

```python
...
state = job.result()  # wait for job completion synchronously
```

```python
async def main():
    ...
    state = await job.result_async()  # wait for job completion asynchronously
```

Synchronous waiting will block the subsequent code execution, so it is not suitable for scenarios where multiple jobs need to be submitted in parallel. At this time, asynchronous waiting can be used.


### Implement simple workflow

Although `ai2-kit` does not provide a workflow scheduling engine, for simple tasks, with the support of Python's asynchronous support and the tool classes provided by `ai2-kit`, it is easy to implement a simple workflow. Next, take the following simple task as an example:
* Pre-processing: Implement a Python function to create n working directories and an input file containing a random number
* Submit n job scripts, each job script reads the number in the input and calculates its square and writes it to the output file
* Post-processing: Implement a Python function to read the output values of all files and sum them after all jobs are completed

The specific code implementation can be found in [simple-workflow](../../example/script/simple-workflow.py).

From the code implementation, it can be seen that the workflow implemented through `ai2-kit` is essentially ordinary Python code, but when it is necessary to execute functions remotely or submit jobs, it will call the interface provided by `ai2-kit HPC executor`. Therefore, it is not difficult for students familiar with Python coding to get started with `ai2-kit`.

More complex workflows are also implemented in the same way, but on this basis, modeling and parsing of configuration files are added, as well as the use of conditional branches and loops for flow control. If you are interested, you can refer to the code in the `ai2-kit.workflow` module.
