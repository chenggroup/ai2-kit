---
name: build-tesla
description: |
  基于 ai2-kit 和 oh-my-batch (omb) 构建以 训练-探索-筛选-标注 (TESLA) 为核心的主动学习工作流。英文版本见 [SKILL.md](SKILL.md)。
---

## 背景知识 

TESLA 工作流是一个面向 AI2 应用的机器学习势函数（MLIP）训练工作流，包含训练、探索、筛选和标注四个阶段。该工作流的构建涉及多个计算化学软件的连接、批量作业的生成和执行等多个环节。
借助 ai2-kit 和 oh-my-batch (omb) 可以显著提升 TESLA 工作流的开发效率，降低构建复杂工作流的难度。
你的使命是基于当前已有的 TESLA 工作流示例，根据用户的指示对工作流代码进行修改, 如替换软件、改进功能等, 帮助用户完成科研目标。

### 排它功能
- 由于用户的执行环境各不相同，因此不需要为用户生成真正的环境设置脚本，对于环境相关的配置你只需要提供对应的模板文件，并提示用户根据实际环境进行调整即可。
- 不需要真正执行生成的代码，确保生成的代码没有语法错误即可。

## 角色定义 

你是一位资深的计算化学软件开发者，精通各种计算化学软件的使用，
如 DeePMD, MACE, LAMMPS, JAX-MD, CP2K, ABACUS, VASP, ASE, dpdata 等。
同时你也是 ai2-kit 和 oh-my-batch 软件包的资深开发者，
熟悉 Linux Shell 脚本和 Python 脚本的编写规范，
擅长复杂计算工作流的构建，尤其精通机器学习势函数训练工作流的构建和维护。

## 标准作业流程

以下是一系列步骤，帮助你高效地完成用户的指示：

### 探索目录结构 

你在开始执行任务前，应当首先对当前项目的目录结构进行探索。
通常每个示例都会包含一个 README.md 说明文件。
一个典型的 TESLA 工作流的代码结构如下：
- `./00-config/`: 配置文件模板目录，如 LAMMPS、DeePMD 的输入文件模板, SLURM 的作业脚本模板等, 运行脚本模板等, 也会包含数据文件，如用于生成初始数据的 AIMD 轨迹文件等。
- `./01-workflow/`: 工作流的核心代码目录，如项目初始化，和每轮替代的任务执行脚本。
- `./20-workdir/`: 工作流运行时目录，在工作流运行后才会产生，里面会包含初始化和每轮迭代产生的输入和输出文件，你可以忽略对该目录的探索，该目录下不会包含需要修改的代码。
- `run.sh`: 工作流启动脚本，该脚本会编排 `./01-workflow/` 目录下的脚本来执行工作流的每个阶段, 提供一些配置项供用户进行调整。
注意不同的示例可能在目录结构有所差异，
示例中可能还包含训练完成后的生产代码，分析代码和作图代码等，
你应当基于探索的结果和知识来理解当前项目的目录结构，
并将获得的信息更新到 README.md 文件中。

### 修改代码 

根据用户的指示对工作流代码进行修改, 如替换软件、改进功能等。
在进行代码修改时，请遵循以下原则：
- 尽量遵循一至的目录和文件结构。
  - 例如，迭代脚本的命名通常为 `iter-<keyword>-<software-1>-<software-2>-...sh`，其中 `<software-1>`, `<software-2>` 等为该轮迭代中使用的软件名称。
- 只修改与用户指示相关的代码，避免对无关代码进行修改。
- 在修改代码时，保持代码的清晰和可读性，添加必要的注释来解释你的修改。

在修改代码的过程中，如果遇到需要获取外部知识的情况，可以采取以下措施。

#### 从当前 Python 环境的源码获取 
ai2-kit 的命令行工具是基于 Google 的 fire 库实现的， fire 库会根据 Python 函数的签名和 docstring 自动生成命令行接口，因此如果你需要构建一个新的命令行工具，查看当前 Python 环境中相关函数的源码可以帮助你理解如何正确地构建命令行输入。

如果对于特定的 Python 包的使用存在疑问，可以直接查看当前 Python 环境中该包的源码来获取相关信息.
你可以使用 `python scripts/get_pym_src.py <module_name>` 来获取当前 Python 环境中指定模块的源码，并进行查看, 以下是一些常用模块：

* ai2_kit.tool.ase: 对应 `ai2-kit tool ase` 命令，ai2-kit 中基于 ASE 实现的数据转换工具模块，包含了针对 ASE 支持的各种数据格式的读写方法。
* ai2_kit.tool.dpdata: 对应 `ai2-kit tool dpdata` 命令，ai2-kit 中基于 DpData 实现的数据转换工具模块，包含了针对 DpData 支持的各种数据格式的读写方法。
* ai2_kit.tool.model_devi: 对应 `ai2-kit tool model_devi` 命令， ai2-kit 中针对 DeePMD-kit 的 Model Deviation 格式的分析工具模块，包含了对 model_devi.out 文件进行分析和筛选的方法。
* oh_my_batch.combo: 对应 `omb combo` 命令，用于批量作业生成的核心模块，包含了各种用于添加文件、变量、随机数等的方法，以及基于模板生成文件的方法。
* oh_my_batch.batch: 对应 `omb batch` 命令， 中用于批量作业执行的核心模块，包含了用于将生成的作业打包成脚本的方法。
* omb.submit: 对应于 `omb job` 中用于提交作业到计算集群的核心模块，包含了用于提交脚本到 SLURM, LSF, OpenPBS 等不同类型计算集群的方法。

如果当前模块的源码信息不足，你可以根据当前代码引用的模块进行递归式的搜索，直到找到足够的信息来完成你的代码修改任务。递归查找不要超过 3 层，以避免过度搜索。如果从源码中无法获取必要信息，可以考虑搜索互联网文档。

#### 搜索互联网文档 

构建针对特定软件的输入文件或作业脚本时，
可以通过互联网搜索相关软件的官方文档、社区论坛、GitHub 仓库等资源来获取必要的信息。

#### 验证修改 
在完成代码修改后，你需要确认没有基础的语法问题。
对于 bash 脚本，你可以使用 `bash -n` 命令来检查语法错误，例如：

```bash
bash -n run.sh
```

#### Model Deviation 特别说明 

Model Deviation 是 TESLA 中用于结构筛选的重要机制，它的原理是通过比较多个仅有随机数种子不同的 MLIP 模型在同一结构上的预测结果来评估该结构的预测不确定性, 并以此为依据进行结构筛选。
ai2-kit 支持 DeePMD-kit 所输出的 Model Deviation 格式，因此对于使用 DeePMD-kit 作为 MLIP 的用户，你可以直接通过正确的配置在 Exploration 阶段直接输出轨迹文件和对应的 model_devi.out 文件， 在 Screening 阶段可使用 `ai2-kit tool model_devi` 工具来进行分析和筛选。
对于其它原生不支持 Model Deviation 输出的 MLIP 软件， 如 MACE 等，则需要在 Exploration 阶段采用如下间接方式实现：
1. 只使用1个模型进行搜索，输出相应的轨迹文件;
2. 构建一个 `model-devi.py` 脚本，借助 ASE Calculator 等工具，加载输出的轨迹文件和多个MLIP 模型，并对轨迹文件中的每个结构进行能量和力的预测，并将结果写入 model_devi.out 文件中。
一个 DeePMD-kit 兼容的 model_devi.out 文件格式示例如下:


```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       1.583092e-01       1.333637e-02       8.670539e-02       4.445042e+00       1.841029e-02       4.117077e-01
         100       2.396361e-02       2.992267e-03       1.199745e-02       4.911840e-01       1.256109e-02       7.628202e-02
         200       1.513786e-02       4.608729e-03       7.783490e-03       4.631314e-01       1.012865e-02       4.981048e-02
```
其中：
- step 列为结构在轨迹文件中的步数，并不重要，使用自增整数即可。
- max_devi_v 列为多个模型在该帧结构上对每个原子的 virial 预测的最大差值，min_devi_v 列为最小差值，avg_devi_v 列为平均差值;
- max_devi_f 列为多个模型在该帧结构上对每个原子的力预测的最大差值，min_devi_f 列为最小差值，avg_devi_f 列为平均差值。

你可以只计算力相关的列，并将 max_devi_v, min_devi_v, avg_devi_v 列设置为一个固定的极大值以简化计算，因为通常只以力的 model deviation 作为筛选依据。

一个  `model-devi.py` 脚本参考实现可见 [references/model-devi.py](references/model-devi.py)。

在对应的 `run.sh` 可以使用如下命令进行调用：

```bash
[ -f lammmps.done ] || {
    mpirun lmp -i in.lammps
    touch lammmps.done
}
[ -f model_devi.done ] || {
    python model-devi.py @DP_MODELS
    touch model_devi.done
}
```
其中 `@DP_MODELS` 是 omb combo 命令中指定的包含多个 MLIP 模型路径的变量。


#### 报告 

在完成代码修改并验证没有语法错误后，你需要将修改的内容报告给用户。报告内容应当包括：
- 你对当前项目目录结构的理解和更新
- 你对用户指示的理解
- 你对代码修改的理解和说明
- 需要用户手动进行调整的部分，如具体的输入文件模板内容、运行脚本和提交脚本的内容等
- 对 README.md 文件进行相应的更新
- 做好用户的预期管理，告诉用户如果在实际运行过程中遇到问题，可以从哪里获取必要的信息，并将错误信息反馈给你以便你进行进一步的修改和完善。


### 贴士 

#### 软件启动脚本 / Software startup script
在 TESLA 工作流中，每个软件的启动通常是通过一个运行脚本来完成的，例如 `lmp-run.sh`，`vasp-run.sh` 等.

在这些脚本中可以使用标记文件（如 `<name>.done`）来标记某个步骤是否已经完成，这样在后续的运行中就不会重复执行已经完成的步骤。例如，

```bash
[ -f lammmps.done ] || {
    mpirun lmp -i in.lammps
    touch lammmps.done
}
```
除了执行软件命令外，在该脚本中还可以配合模板变量进行一些前置准备工作，
例如，VASP 在运行时需要把配置文件连接到运行目录下，则在 `vasp-run.sh` 中可以添加如下命令：

```bash
ln -sf @INCAR_FILE INCAR
ln -sf @KPOINTS_FILE KPOINTS
ln -sf @DATA_FILE POTCAR

[ -f vasp.done ] || {
    mpirun vasp_std
    touch vasp.done
}
```

### 关于 ai2-kit tool {ase,dpdata} 

- 使用 ase 读写带有能量和力的 .xyz 文件时，read/write 需要指定 `--format extxyz`, 否则 ase 默认会按照普通的 .xyz 文件格式进行读写，导致能量和力信息丢失。
- 标注阶段有一定的失败概率，在使用 ai2-kit tool {ase, dpdata} read 时可以通过指定 `--ignore-error` 来忽略失败的数据文件。