# HPC 执行器

## 介绍
`ai2-kit` 实现了一个轻量级的 HPC 执行器用于提交和管理 HPC 集群上的作业。相比于其它的 HPC 调度器或者支持 HPC 的工作流框架 (如 DpDispatcher, parsl, DFlow, etc), `ai2-kit` 的 HPC 执行器具有以下特点:

* 支持 SSH 远程执行和本地执行两种模式
* 支持通过跳板机连接 HPC 集群
* 支持远程执行 Python 函数
* 支持在登录节点直接执行简单命令或者函数
* 更高效稳定的作业状态轮询机制
* 基于 Checkpoint 的状态恢复机制
* 支持同步或者异步等待作业执行
* 易于定制或与其它框架集成

如果您在使用其它方案时遇到以下问题，不妨试试 `ai2-kit` 提供的 HPC Executor:
* 过高的上手和学习成本
* 连接不稳定
* 需要频繁将数据复制到本地进行处理后再提交回集群运行

当前, `ai2-kit HPC executor` 只支持 Slurm 作业系统，对其它作业系统的支持视实际需求而定，欢迎提交 Issue 或 PR。

## 使用方法

### 基本用法

初始化 `ai2-kit HPC executor` 有两种方式，使用字典配置或者 `Pydantic` 对象， 两种方式的配置项完全一致，前者更适合直接使用，后者更适合集成到其它框架。 以下为使用字典进行配置的示例。

```python
from ai2_kit.core.executor import HpcExecutor

executor = HpcExecutor.from_config({
    'ssh': {  # 指定 ssh 连接信息，如果缺省则使用本地执行
        'host': 'user01@hpc-login01',
        'gateway': {  # 如果需要通过跳板机连接,可指定此配置 (可选) 
          'host': 'user01@jump-host',  
        }
    },
    'queue_system': {
        'slurm': {}  # 指定作业系统为 Slurm 
    },
    'work_dir': '/home/user01/ai2-kit/work_dir',  # 指定工作目录
    'python_cmd': '/home/user01/conda/env/py39/bin/python',  # 指定 Python 解释器
}, 'cheng-lab')  # 指定集群名称 （可选）

executor.init()  # 初始化执行器
executor.run('echo "hello world"')  # 在登录节点执行命令
```

上述示例完成如下工作：
1. 实例化一个 `HpcExecutor` 对象
2. 初始化 `HpcExecutor` 对象
3. 在登录节点执行命令 `echo "hello world"`


### 远程执行 Python 函数

在 HPC 上执行复杂的计算任务的通常模式是：
* 在登录节点上通过命令行或者Python准备好计算任务的输入数据
* 提交作业到队列并等待
* 作业完成后在登录节点上使用命令行或者 Python 处理计算任务的输出数据

为了满足上述模式，`ai2-kit HPC executor` 除了提供 `run` 接口用于在登录节点执行命令外，还提供了 `run_python_fn` 用于在登录节点直接运行 Python 函数。 

```python
def add(a, b):
    return a + b

result = executor.run_python_fn(add)(1, 2)  # 在登录节点上运行 add 函数
```

需要注意的是，为了在远程节点执行本地 Python 函数必须满足以下条件：
* 本地 Python 环境与远程 Python 环境的主版本需一致 (如同为 3.8.x)
  * 远程 Python 环境的配置可以通过 `python_cmd` 参数指定
* 函数的参数和返回值必须是可序列化的 (不能包含诸如锁、文件句柄等不可序列化的对象)
* 函数依赖的软件包必须存在于登录节点的 Python 环境中
  * 例如，假设远程执行的函数使用了 `numpy` 包，那么登录节点的 Python 环境中必须存在 `numpy` 包
* 如果函数依赖于其它**本地实现的方法或者类**，这些方法和类需要通过特殊方式定义, 否则会在远程出现 `ModuleNotFoundError`
  * 此为 cloudpickle 的限制，详见： [1](https://stackoverflow.com/a/75293155/3099733)


### 提交作业

`ai2-kit HPC executor` 提供了 `submit` 接口用于提交作业到 HPC 集群。 示例如下

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

job = executor.submit(script)  # 提交作业
``` 

除了上述直接编写脚本的方式外，也可以通过 `ai2-kit` 提供的工具类生成脚本，示例如下：

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

job = executor.submit(script.render(), cwd='/path/to/cwd')  # 提交作业
```

同时，提交作业时可以指定额外的参数满足不同的需求，如设定执行目录和用于错误恢复的checkpoint文件。


### 等待作业完成

提交任务提交后可以有同步和异步两种方式等待其完成。示例如下：

```python
...
state = job.result()  # 同步等待作业完成

async def main():
    ...
    state = await job.result_async()  # 异步等待作业完成
```

同步等待由于会阻塞后续的代码执行，因此对于需要并行提交多个作业的场景不适用，此时可以使用异步等待。


### 实现简单工作流
