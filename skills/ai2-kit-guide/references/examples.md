# Versioned example index

Reference revision: `76181f47cbc2e204c8977af1b1d0e1a229bfe022`.
[Supplementary examples](https://github.com/chenggroup/ai2-kit/tree/76181f47cbc2e204c8977af1b1d0e1a229bfe022/example/supplement).
Paths below are relative to the repository root. Check the actual checkout before
editing: local manuscript copies and partial evidence snapshots may differ.

| Goal | Example and entry points | Additional requirements / limits |
| --- | --- | --- |
| A potential for a bulk liquid | `example/supplement/tesla-h2o/run.sh`; `01-workflow/setup.sh`; `01-workflow/iter-classic-dp-lammps-cp2k.sh` within that example | DeePMD/LAMMPS/CP2K TESLA. Replace structures and scientific settings for other liquids; electrolyte accuracy is not established by water results. |
| Potential training with enhanced exploration | `example/supplement/tesla-auco/run.sh`; `01-workflow/iter-classic-dp-lammps-cp2k.sh` | Au–CO2 example, PLUMED inputs and reference templates. Select appropriate collective variables and assess sampling for the new chemistry. |
| Redox free energy | `example/supplement/tesla-redox/run.sh`; `20-workflow/iter-fep-redox-dp-lmp-cp2k.sh` | Separate charge-state datasets/committees, coupled sampling and FEP analysis. More than changing a single-state training parameter. |
| Acid–base free energy | `example/supplement/tesla-pka/run.sh`; `20-workflow/iter-fep-pka-dp-lammps-cp2k.sh` | State definitions, atom removal/charge changes, reference and coupling inputs, free-energy analysis. |
| Electrochemical interface response | `example/supplement/ec-mlp/run.sh`; `02-workdir/iter0-cp2k-mokit-dw-ec-lammps.sh`; `02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh` | Wannier/MOKIT processing, property and potential models, electrode-response settings. Check whether actual files implement constant charge or constant potential. |
| Vibrational spectra | `example/supplement/spectra/README.md`; `run.sh`; `spectra/cal_spectra.py` | IR needs dipoles; Raman needs polarizabilities; the demonstrated SFG analysis uses both. Reference property labels, learned predictions and correlation/Fourier analysis are additional stages beyond energy/force training. Check package/script compatibility, sampling and preprocessing. |
| Substitute MACE/OpenMM/VASP for the water stack | `example/supplement/skill-demo/run.sh`; `01-workflow/iter-classic-mace-openmm-vasp.sh`; `00-config/openmm/model-devi.py` | Engine templates and force-deviation adapter; VASP relabelling rather than assuming CP2K labels are interchangeable. Generated scripts and published numerical results have distinct provenance. |

Each row names one demonstrated starting route. It does not establish an unchanged
solution for every system in that class. The indexed spectroscopy example is not an
excited-state/nonadiabatic surface-hopping engine; such a task needs new scientific
interfaces and workflow integration.

For generic operations inspect `ai2_kit/tool/ase.py`, `ai2_kit/tool/dpdata.py`, and
`ai2_kit/tool/model_devi.py`. The application scripts specify stage connections;
oh-my-batch supplies template/job preparation and execution services. Do not invent
an Ai2-Kit command when the installed version exposes a different interface.
