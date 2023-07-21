# CLL 势函数训练工作流

```bash
ai2-kit workflow cll-mlp-training
```

## 简介

CLL 工作流通过对 DPGEN 的流程和实现进行改进，以满足更复杂的势函数训练需求以及可持续的代码集成。CLL 工作流采用闭环学习的模式，通过迭代的方式自动训练 MLP 势函数。在每一次迭代中，工作流使用由第一性原理方法生成的标记结构来训练多个 MLP 模型。然后，这些模型被用来探索新的结构作为下一次迭代的训练数据。迭代会一直持续，直到 MLP 模型的质量满足预定的标准。每次迭代的配置可根据训练需要进行更新，以进一步提高训练效率。

![cll-mlp-diagram](../res/cll-mlp-diagram.svg)

CLL 工作流的主要改进包括:
* 更语义化的配置系统，以支持不同软件的选择，以及根据不同体系设定不同的软件配置。
* 更健壮的 checkpoint 机制，以减少执行中断产生的影响。
* 支持远程Python方法执行以避免不必要的数据搬运， 提高在HPC集群上执行的效率和稳定性。

目前 CLL 工作流支持使用以下工具进行势函数训练:
* Label：CP2K, VASP
* Train：DeepMD
* Explore：LAMMPS, LASP
* Select: Model deviation, ASAP (TODO)

当前 CLL 工作流通过 `ai2-kit` 自带的HPC执行器进行作业提交，支持在单一HPC集群完成计算，未来根据需要考虑支持多集群调度，以及支持包括 `dflow` 在内的不同工作流引擎。


## 环境要求
* Workflow 运行环境与 HPC 运行环境的 Python 主版本需要保持一致，否则远程执行会出现问题。
* HPC 的运行环境需要安装 `ai2-kit`, 通常来讲 HPC 上的 `ai2-kit` 与本地的 `ai2-kit` 版本不需要严格相同， 但如果差异过大仍有出现问题的可能，所以建议在条件允许时使用相同版本的 `ai2-kit`。

## 使用说明

以下通过通一个案例来说明 CLL 工作流的使用方法。

### 数据准备
工作流执行所需要的数据需要提前放置在 HPC 集群节点上。开始工作流的执行前你需要准备以下数据：
* 用于势函数训练的初始结构或初始数据集
* 用于结构搜索的初结构

 假设你已经有一段使用 AIMD 生成的轨迹 `h2o_64.aimd.xyz`, 那么你可以通过使用 [`ai2-kit tool ase`](./ase.md) 命令行工具来准备这些数据。

 ```bash
mkdir -p data/explore

# 从0-900帧中抽取训练集, 间隔 5 帧抽取
ai2-kit tool ase read h2o_64.aimd.xyz --index ':900:5' - set_cell "[12.42,12.42,12.42,90,90,90]" - write data/training.xyz

# 从900-帧后抽取验证集，间隔 5 帧抽取
ai2-kit tool ase read h2o_64.aimd.xyz --index '900::5' - set_cell "[12.42,12.42,12.42,90,90,90]" - write data/validation.xyz

# 抽取用于初始结构搜索的数据，间隔 100 帧抽取
ai2-kit tool ase read h2o_64.aimd.xyz --index '::100' - set_cell "[12.42,12.42,12.42,90,90,90]" - write_each_frame "./data/explore/POSCAR-{i:04d}" --format vasp
 ```


### 配置文件准备

CLL 工作流的配置文件采用 YAML 格式，并且支持以任意维度进行分拆，`ai2-kit` 会在执行时自动将其合并。适度的分拆有利于配置文件的维护和重用。通常情况下，我们可以将配置文件拆分为以下几个部分：

* artifact.yml: 用于配置工作流所需的数据
* executor.yml: 用于配置 HPC 执行器的参数
* workflow.yml: 用于配置工作流软件的参数

我们首先从 `artifact.yml` 开始，这个配置文件用于配置工作流所需的数据。在这个例子中，我们需要配置三个数据集，分别是用于训练的数据集，用于验证的数据集，以及用于结构搜索的数据集。这三个数据集的配置如下：

```yaml
.base_dir: &base_dir /home/user01/data/

artifacts:
  h2o_64-train:
    url: !join [*base_dir, training.xyz]
    
  h2o_64-validation:
    url: !join [*base_dir, validation.xyz]
    attrs:
      deepmd:
        validation_data: true  # 指定该数据集为验证集

  h2o_64-explore:
    url: !join [*base_dir, explore]
    includes: POSCAR*
    attrs:  # 如有需要可在这里针对特定体系指定特定的软件配置, 此例无需此配置，因此放空
      lammps:
        plumed_config_file:   
      cp2k:
        input_template_file:
```

这里我们使用 `ai2-kit` 提供的自定义 tag `!join` 来简化数据配置, 相关功能可查看 [TIPS](./tips.md) 文档。

接下来我们配置 `executor.yml` 文件，这个文件用于配置与 HPC 链接相关的参数，以及软件的使用模板

```yaml
executors:
  hpc-cluster01:
    ssh:
      host: user01@login-01  # 登录节点
      gateway:
        host: user01@jump-host  # 跳板机（可选）
    queue_system:
      slurm: {}  # 使用 slurm 作为作业调度系统

    work_dir: /home/user01/ai2-kit/workdir  # 工作目录
    python_cmd: /home/user01/libs/conda/env/py39/bin/python  # 远程 Python 解释器

    context:
      train:
        deepmd:  # 配置 deepmd 作业提交模板
          script_template:
            header: |
              #SBATCH -N 1
              #SBATCH --ntasks-per-node=4
              #SBATCH --job-name=deepmd
              #SBATCH --partition=gpu3
              #SBATCH --gres=gpu:1
              #SBATCH --mem=8G
            setup: |
              set -e
              module load deepmd/2.2
              set +e

      explore:
        lammps:  # 配置 lammps 作业提交模板
          lammps_cmd: lmp_mpi
          concurrency: 5
          script_template:
            header: |
              #SBATCH -N 1
              #SBATCH --ntasks-per-node=4
              #SBATCH --job-name=lammps
              #SBATCH --partition=gpu3
              #SBATCH --gres=gpu:1
              #SBATCH --mem=24G
            setup: |
              set -e
              module load deepmd/2.2
              export OMP_NUM_THREADS=1
              set +e

      label:
        cp2k:  # 配置 cp2k 作业提交模板
          cp2k_cmd: mpiexec.hydra cp2k.popt
          concurrency: 5
          script_template:
            header: |
              #SBATCH -N 1
              #SBATCH --ntasks-per-node=16
              #SBATCH -t 12:00:00
              #SBATCH --job-name=cp2k
              #SBATCH --partition=c52-medium
            setup: |
              set -e
              module load intel/17.5.239 mpi/intel/2017.5.239
              module load gcc/5.5.0
              module load cp2k/7.1
              set +e
```

最后则是对 `workflow.yml` 文件的配置，这个文件用于配置工作流的参数。

```yaml
workflow:
  general:
    type_map: [ H, O ]
    mass_map: [ 1.008, 15.999 ]
    max_iters: 2  # 指定最大迭代次数

  train:
    deepmd:  # deepmd 参数配置
      model_num: 4
      # 此例中使用的数据尚未标注，需要在 label 配置，此处为空，如有现成的已标注 deepmd/npy 数据集可在此处指定
      init_dataset: [ ]  
      input_template:
        model:
          descriptor:
            type: se_a  
            sel:
            - 100
            - 100
            rcut_smth: 0.5
            rcut: 5.0
            neuron:
            - 25
            - 50
            - 100
            resnet_dt: false
            axis_neuron: 16
            seed: 1
          fitting_net:
            neuron:
            - 240
            - 240
            - 240
            resnet_dt: true
            seed: 1
        learning_rate:
          type: exp
          start_lr: 0.001
          decay_steps: 2000
        loss:
          start_pref_e: 0.02
          limit_pref_e: 2
          start_pref_f: 1000
          limit_pref_f: 1
          start_pref_v: 0
          limit_pref_v: 0
        training:
          #numb_steps: 400000
          numb_steps: 5000
          seed: 1
          disp_file: lcurve.out
          disp_freq: 1000
          save_freq: 1000
          save_ckpt: model.ckpt
          disp_training: true
          time_training: true
          profiling: false
          profiling_file: timeline.json

  label:
    cp2k:  # 指定 cp2k 参数配置
      limit: 10
      # 此例中使用的数据尚未标注，因此需要在此配置，如有现成的已标注数据集则此处应为空
      # 当此配置为空时，工作流会自动跳过第1次的label阶段从train开始执行
      init_system_files: [ h2o_64-train, h2o_64-validation ]  
      input_template: |
        &GLOBAL
           PROJECT  DPGEN
        &END
        &FORCE_EVAL
           &DFT
              BASIS_SET_FILE_NAME  /home/user01/data/cp2k/BASIS/BASIS_MOLOPT
              POTENTIAL_FILE_NAME  /home/user01/data/cp2k/POTENTIAL/GTH_POTENTIALS
              CHARGE  0
              UKS  F
              &MGRID
                 CUTOFF  600
                 REL_CUTOFF  60
                 NGRIDS  4
              &END
              &QS
                 EPS_DEFAULT  1.0E-12
              &END
              &SCF
                 SCF_GUESS  RESTART
                 EPS_SCF  3.0E-7
                 MAX_SCF  50
                 &OUTER_SCF
                    EPS_SCF  3.0E-7
                    MAX_SCF  10
                 &END
                 &OT
                    MINIMIZER  DIIS
                    PRECONDITIONER  FULL_SINGLE_INVERSE
                    ENERGY_GAP  0.1
                 &END
              &END
              &LOCALIZE
                 METHOD  CRAZY
                 MAX_ITER  2000
                 &PRINT
                    &WANNIER_CENTERS
                       IONS+CENTERS
                       FILENAME  =64water_wannier.xyz
                    &END
                 &END
              &END
              &XC
                 &XC_FUNCTIONAL PBE
                 &END
                 &vdW_POTENTIAL
                    DISPERSION_FUNCTIONAL  PAIR_POTENTIAL
                    &PAIR_POTENTIAL
                       TYPE  DFTD3
                       PARAMETER_FILE_NAME  dftd3.dat
                       REFERENCE_FUNCTIONAL  PBE
                    &END
                 &END
              &END
           &END
           &SUBSYS
              &KIND O
                 BASIS_SET  DZVP-MOLOPT-SR-GTH
                 POTENTIAL  GTH-PBE-q6
              &END
              &KIND H
                 BASIS_SET  DZVP-MOLOPT-SR-GTH
                 POTENTIAL  GTH-PBE-q1
              &END
           &END
           &PRINT
              &FORCES ON
              &END
           &END
        &END

  explore:
    lammps:
      timestep: 0.0005
      sample_freq: 100
      tau_t: 0.1
      tau_p: 0.5
      nsteps: 2000
      ensemble: nvt

      post_init_section: |
          neighbor 1.0 bin
          box      tilt large

      post_read_data_section: |
          change_box all triclinic

      system_files: [ h2o-64-explore ]
      explore_vars:
          temp: [ 330, 440]
          pres: [1]

  select:
    model_devi:
        f_trust_lo: 0.4
        f_trust_hi: 0.6

  update:
    walkthrough:
      # 可在此处指定从第2次迭代及其后要使用的参数配置
      # 此处配置的参数会覆盖 workflow 一节的任意配置, 可根据训练策略进行调整
      table:
        - train:  # 第二次迭代时训练步数为 10000
            deepmd:
              input_template:
                training:
                  numb_steps: 10000
        - train:  # 第三次迭代时训练步数为 20000
            deepmd:
              input_template:
                training:
                  numb_steps: 20000
```

### 执行工作流

完成配置工作后即可开始工作流的执行

```bash
ai2-kit workflow cll-mlp-training *.yml --executor hpc-cluster01 --path-prefix h2o_64-run-01 --checkpoint run-01.ckpt
```

上述参数中，
* `*.yml` 用于指定配置文件，可以指定多个配置文件，`ai2-kit` 会自动将其合并, 此处使用 `*` 通配符指定;
* `--executor hpc-cluster01` 用于指定要使用的 HPC 执行器，此处使用了上一节中配置的 `hpc-cluster01` 执行器;
* `--path-prefix h2o_64-run-01` 指定远程工作目录，它会在 `work_dir` 下创建一个 `h2o_64-run-01` 的目录用于存放工作流的执行结果; 
* `--checkpoint run-01.cpkt` 会在本地生成一个checkpoint文件，用于保存工作流的执行状态，以便在执行中断后恢复执行。
