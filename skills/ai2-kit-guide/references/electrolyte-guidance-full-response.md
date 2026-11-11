# Exact user prompt

I want an ML potential for a liquid electrolyte. I have CP2K-labelled AIMD and use LAMMPS. Which example should I start from and what must I change? Please guide me only. If I later ask you to create the files, do I have to invoke another skill?

# Response

Start from **`tesla-h2o`**, the bulk-liquid TESLA example. Its DeePMD training → LAMMPS exploration → committee-deviation screening → CP2K labelling → dataset update loop matches your existing data and simulation engine. Applying it to your electrolyte is an adaptation; the water example does not establish electrolyte accuracy.

The useful entry points in this paper checkout are `tesla-h2o/run.sh`, `01-workflow/setup.sh`, and `01-workflow/iter-classic-dp-lammps-cp2k.sh`. The README mentions an `iter-basic-...` filename, but the actual entry script calls `iter-classic-dp-lammps-cp2k.sh`.

You can reuse the stage connections, dataset accumulation, structure conversion and job-template machinery. Change these parts before using the example:

- **Initial data:** `setup.sh` expects a labelled `00-config/aimd.xyz`, samples 50 training frames and two MD starting structures. Adapt ingestion to your actual CP2K output format; a coordinate-only XYZ is insufficient. Verify that energies, forces, cells, atom order, species and units survive conversion. Choose representative sampling for your composition and thermodynamic conditions, and reserve independent evaluation data outside the training pool.
- **Species and physical model:** replace `[O,H]` in `run.sh`, the hard-coded DeePMD `type_map` in `00-config/deepmd/input.json`, and the O/H masses and type ordering in `00-config/lammps/lammps.in`. Keep these mappings consistent with the converted datasets and CP2K element definitions. Establish composition, concentration, total charge and intended state range. Assess whether the example's local potential representation is adequate for the electrolyte's electrostatics and transferability; additional long-range or charge-response modelling would require more than a species substitution.
- **Reference labels:** adapt `00-config/cp2k/cp2k.inp` to the reference level used for your AIMD, including functional, dispersion, basis sets, pseudopotentials, charge/spin, periodicity and converged numerical settings. The supplied template is neutral, closed-shell BLYP-D3 water with O/H definitions. Avoid silently mixing incompatible reference levels. If stress is needed, plan its labelling and training explicitly; the supplied training template assigns zero virial loss weight.
- **Training and exploration:** choose training budgets, model ensemble, temperatures, timestep, sampling frequency, label budget and screening thresholds for your system. The LAMMPS template runs NVT; changing its pressure variable alone does not make it NPT. Water's deviation thresholds are starting values, and committee agreement is not an independent prediction-error measurement.
- **Execution setup:** adapt engine launch commands, paths, package versions and Slurm headers to your machine. `run.sh` installs packages and the iteration script submits jobs, so it is an execution entry point, not a harmless configuration preview.

The smallest next step is to inspect one representative labelled AIMD input and its CP2K settings, then define the species mapping and target state range. For that, the missing details are your electrolyte composition/concentration, dataset format/location and intended temperature/density or pressure range. Independent energy/force errors, DFT convergence, sampling convergence and agreement with experiment should be assessed separately.

**You do not need to invoke another skill later.** Ai2-Kit guide includes TESLA development internally. If you ask me to prepare the files, I can carry this context into its build module; asking for files does not itself request training, MD or DFT execution. This answer is guidance only: no workflow files were edited and no calculations were run.

Source scope: I inspected the local `tesla-h2o` example in paper checkout `afd54157d2960a9cf1636eabe93e560eddf81b85` and corresponding preserved source files. The skill index references upstream `76181f47cbc2e204c8977af1b1d0e1a229bfe022`, while `revision/evidence/source_manifest.json` records snapshot commit `507cd0f8db92587dd29270633b3c39d92fda0c3a`. These are not one verified checkout; the local iteration script also differs from the snapshot. I have not tested installed-package compatibility or inspected your electrolyte data.
