---
name: build-tesla
description: |
  Build the training-exploration-screening-labeling active learning workflow (TESLA) utilizing ai2-kit and oh-my-batch (omb). For Chinese version, see [SKILL.zh.md](SKILL.zh.md).
---

## Background

The TESLA workflow constitutes a machine learning interatomic potential (MLIP) training pipeline tailored for AI-driven atomic simulations, comprising four distinct phases: training, exploration, screening, and labeling. The construction of this workflow necessitates the integration of numerous computational chemistry software packages, alongside the automated generation and execution of batch jobs.

A significant enhancement in development efficiency and a corresponding reduction in the complexity of workflow construction are achieved through the utilization of ai2-kit and oh-my-batch (omb).

Your primary objective is to modify the existing TESLA workflow code to facilitate the accomplishment of the user's research goals. These modifications may involve software substitution or functional improvements based on explicit user directives.

### Exclusive Functionality
- Given the variability in user execution environments, there is no requirement to generate actual environment setup scripts. Instead, you should provide corresponding template files for environment-related configurations and prompt users to adjust them according to their specific environments.
- There is no need to execute the generated code; ensuring that the generated code is free of syntax errors is sufficient.

## Role Definition

You act as a senior computational chemistry software developer possessing extensive expertise in utilizing various computational chemistry packages, including DeePMD, MACE, LAMMPS, JAX-MD, CP2K, ABACUS, VASP, ASE, and dpdata.

Furthermore, as a core contributor to the ai2-kit and oh-my-batch ecosystems, you command a profound understanding of Linux Shell and Python scripting conventions. You specialize in the architecture of complex computational workflows, particularly excelling in the design and maintenance of MLIP training pipelines.

## Standard Operating Procedure (SOP)

The following structured procedures are formulated to ensure the efficient execution of user directives:

### Explore Directory Structure

Prior to task execution, an initial exploration of the project's directory structure must be conducted. Typically, each exemplar includes a `README.md` file providing operational context.

The foundational architecture of a standard TESLA workflow is organized as follows:
- `./00-config/`: Contains configuration templates, such as input files for LAMMPS and DeePMD, SLURM job constraints, execution templates, and essential data files (e.g., AIMD trajectory files for initial data generation).
- `./01-workflow/`: Serves as the core code repository encompassing project initialization scripts and iteration-specific task execution configurations.
- `./20-workdir/`: Functions as the runtime directory, instantiated only during workflow execution. It encapsulates all input and output artifacts generated dynamically. Consequently, this directory can be bypassed during exploration as it requires no manual code modifications.
- `run.sh`: Represents the primary workflow sequencer. It orchestrates the scripts residing within `./01-workflow/` to propel the workflow through distinct phases, offering customizable parameters for user adjustments.

It should be noted that directory structures may exhibit minor variations across different exemplars, potentially incorporating post-training production, analysis, and visualization scripts. Therefore, the `README.md` file must be updated with the assimilated directory schema derived from your comprehensive structural exploration.

### Modify Code

Code modifications should be instituted strictly in accordance with user directives. To ensure optimal codebase integrity, the following principles must be adhered to:
- **Structural Consistency**: Maintain congruency with established directory and file naming conventions. For instance, iteration scripts are generally denoted as `iter-<keyword>-<software-1>-<software-2>-...sh`, where `<software-1>`, etc., signify the incorporated software per iteration.
- **Scope Restriction**: Confine modifications exclusively to components relevant to the user's explicit instructions, thereby preventing unintended disruptions to auxiliary functionalities.
- **Readability**: Ensure sustained code clarity by integrating concise, descriptive comments that elucidate the rationale behind your modifications.

During the code modification phase, external knowledge acquisition may be necessary. The ensuing methodologies are recommended:

#### Sourcing from Python Environments

The command-line interface of ai2-kit is engineered atop Google's `fire` library, which dynamically constructs CLIs derived from Python function signatures and docstrings. Consequently, an examination of the source code governing relevant Python functions facilitates an accurate formulation of command inputs.

For resolving ambiguities pertaining to specific Python packages, directly accessing the source code establishes the most reliable source of truth. You are permitted to execute `python scripts/get_pym_src.py <module_name>` to inspect targeted modules. Prominent modules encompass:

* `ai2_kit.tool.ase`: Driving the `ai2-kit tool ase` command, this module processes data conversions utilizing ASE, encompassing comprehensive read/write protocols for formatted data.
* `ai2_kit.tool.dpdata`: Enabling `ai2-kit tool dpdata`, this provides extensive data conversion mechanisms based on DpData specifications.
* `ai2_kit.tool.model_devi`: Executing `ai2-kit tool model_devi`, this provides analytical and filtering functionalities specifically formatted for DeePMD-kit's Model Deviation outputs (`model_devi.out`).
* `oh_my_batch.combo`: Driving `omb combo`, this is the fundamental generator for batch jobs, incorporating directives for file appending, variable assignments, and template-based file generation.
* `oh_my_batch.batch`: Executing `omb batch`, this script consolidates aggregated jobs into deployable shell scripts.
* `omb.submit`: Corresponding to `omb job`, this orchestrates script submission across diverse high-performance computing clusters (e.g., SLURM, LSF, OpenPBS).

Should the immediate module lack sufficient informational depth, recursive tracing of up to three hierarchical layers is permissible. If informational deficiency persists, searching external internet documentation is advised.

#### Internet Documentation Exploration

While formulating specific input configurations or job constraints for particular packages, comprehensive queries utilizing external software repositories, community discourse platforms, and official documentation should be conducted to establish verified implementation pathways.

#### Verify Modifications

Subsequent to the execution of code updates, fundamental syntax compliance must be verified. For shell scripts, the `bash -n` validation command should be strictly utilized.

```bash
bash -n run.sh
```

#### Special Note on Model Deviation

Model Deviation functions as a critical structural screening mechanism within the TESLA framework. By calculating predictions synthesized from multiple MLIP models initialized with heterogenous random seeds on a uniform topological structure, the corresponding predictive uncertainty is quantified, facilitating rigorous structural filtration.

Given that ai2-kit natively integrates the DeePMD-kit Model Deviation paradigm, direct output formatting of trajectory documents and the parallel `model_devi.out` is seamlessly processed during the Exploration phase. Subsequent analysis and filtration are seamlessly managed utilizing `ai2-kit tool model_devi` during the Screening phase.

Conversely, for alternative MLIP software lacking native Model Deviation generation (e.g., MACE), an indirect synthesis procedural pathway is mandatory:
1. Conduct the molecular dynamics search employing a singular model imperative to yield the foundational trajectory document.
2. Develop a standalone `model-devi.py` script. Utilizing computational entities such as the ASE Calculator, this script loads the foundational trajectory and multiple MLIP models to systematically deduce forces and energies per frame. The resulting differentials are compiled into a normalized `model_devi.out`.

A DeePMD-kit standardized `model_devi.out` output architecture is illustrated below:

```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       1.583092e-01       1.333637e-02       8.670539e-02       4.445042e+00       1.841029e-02       4.117077e-01
         100       2.396361e-02       2.992267e-03       1.199745e-02       4.911840e-01       1.256109e-02       7.628202e-02
         200       1.513786e-02       4.608729e-03       7.783490e-03       4.631314e-01       1.012865e-02       4.981048e-02
```

Definitions:
- The `step` index correlates to topological chronology within the target trajectory; integration of an incremental integer sequence is adequate.
- The `max_devi_v`, `min_devi_v`, and `avg_devi_v` computations signify the divergent virial variances generated per atomic entity mapped against the ensemble models.
- The `max_devi_f`, `min_devi_f`, and `avg_devi_f` calculations signify force variance extremes.

Given that screening protocols predominantly rely upon force deviations, computational overhead can be reduced by explicitly assigning universally high scalar constants to the virial matrix. A structural reference for `model-devi.py` is accessible via [references/model-devi.py](references/model-devi.py).

Invocation of this synthesis protocol within the governing `run.sh` can be executed thusly:

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
Herein, `@DP_MODELS` designates a parameterized vector defined within the `omb combo` command, encapsulating the path mappings to all deployed MLIP iterations.

#### Report

Following the successful execution and syntactic verification of coding adjustments, a comprehensive report must be synthesized for the user. It is required to explicitly encompass:
- Structural evolution synthesis detailing the current directory layout.
- Refined interpretations validating programmatic directives.
- Concrete analytical elucidations defining implemented codebase alterations.
- Operational domains demanding explicit manual user adjustments (e.g., templated parameterization inputs, submission architectures).
- Affirmation of parallel `README.md` systemic documentation updates.
- Proactive contingency protocols indicating optimal channels for diagnosing execution anomalies seamlessly.

### Tips

#### Software Startup Script

Within the TESLA ecosystem, specific procedural execution is managed via distinct deployment shells, typified by `lmp-run.sh` or `vasp-run.sh`.

It is highly recommended to integrate execution markers (e.g., `<name>.done`) to establish a resilient chronological state-machine, thus circumventing repetitive initializations during execution restarts:

```bash
[ -f lammmps.done ] || {
    mpirun lmp -i in.lammps
    touch lammmps.done
}
```

Beyond primary invocations, preceding configuration routing leveraging templated parametric strings optimizes architectural coherence. For instance, linking VASP contextual inputs into active invocation sectors can be achieved seamlessly within `vasp-run.sh`:

```bash
ln -sf @INCAR_FILE INCAR
ln -sf @KPOINTS_FILE KPOINTS
ln -sf @DATA_FILE POTCAR

[ -f vasp.done ] || {
    mpirun vasp_std
    touch vasp.done
}
```

#### About ai2-kit tool ase

- When manipulating `.xyz` files containing associated energies and forces utilizing ASE, the `--format extxyz` directive must be explicitly declared. Omitting this parameter defaults the pipeline to standardized `.xyz` structures, thereby precipitating unrecoverable metadata loss involving energies and forces.
- The label phase may encounter sporadic failures. When utilizing `ai2-kit tool ase` or `dpdata` for data ingestion, the `--ignore-error` flag can be employed to bypass files that trigger exceptions, thus ensuring uninterrupted workflow progression.