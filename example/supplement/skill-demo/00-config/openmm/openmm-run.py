#!/usr/bin/env python
"""
OpenMM NVT MD exploration script using MACE potential via openmmml.

Usage:
    python openmm-run.py <structure_file> <mace_model> <n_steps> <temperature>
                         <sample_freq> <output_traj> <seed>

Arguments:
    structure_file  Path to initial structure file (extxyz format)
    mace_model      Path to MACE model file (.model / .pt)
    n_steps         Total number of MD steps
    temperature     Temperature in Kelvin
    sample_freq     Save one frame every N steps
    output_traj     Output trajectory file path (extxyz format)
    seed            Integer random seed for velocity initialisation
"""
import sys
import numpy as np
import ase
import ase.io
import openmm
from openmm import app, unit
from openmmml import MLPotential

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
ANGSTROM_TO_NM    = 0.1          # Å  → nm
NM_TO_ANGSTROM    = 10.0         # nm → Å
KJ_MOL_NM_TO_EV_ANG = 0.0010364 # kJ mol⁻¹ nm⁻¹ → eV Å⁻¹
KJ_MOL_TO_EV      = 0.010364     # kJ mol⁻¹ → eV


def atoms_to_openmm(atoms):
    """Convert ASE Atoms to an OpenMM Topology and positions array (nm)."""
    topology = app.Topology()
    chain    = topology.addChain()
    residue  = topology.addResidue('MOL', chain)

    for sym in atoms.get_chemical_symbols():
        element = app.Element.getBySymbol(sym)
        topology.addAtom(sym, element, residue)

    # Periodic box vectors: Å → nm
    # setPeriodicBoxVectors takes a single sequence of 3 Vec3 objects
    cell = atoms.get_cell() * ANGSTROM_TO_NM  # (3,3) array in nm
    topology.setPeriodicBoxVectors([
        openmm.Vec3(*cell[0]) * unit.nanometer,
        openmm.Vec3(*cell[1]) * unit.nanometer,
        openmm.Vec3(*cell[2]) * unit.nanometer,
    ])

    positions_nm = atoms.get_positions() * ANGSTROM_TO_NM
    return topology, positions_nm


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        sys.exit(1)

    structure_file = sys.argv[1]
    mace_model     = sys.argv[2]
    n_steps        = int(sys.argv[3])
    temperature    = float(sys.argv[4])  # Kelvin
    sample_freq    = int(sys.argv[5])
    output_traj    = sys.argv[6]
    seed           = int(sys.argv[7])

    print(f"Structure   : {structure_file}")
    print(f"MACE model  : {mace_model}")
    print(f"Steps       : {n_steps}  Temp: {temperature} K  Sample every: {sample_freq} steps")

    # ------------------------------------------------------------------
    # Load initial structure
    # ------------------------------------------------------------------
    init_atoms = ase.io.read(structure_file, format='extxyz')
    topology, positions_nm = atoms_to_openmm(init_atoms)

    # ------------------------------------------------------------------
    # Build OpenMM System with MACE potential via openmmml
    # ------------------------------------------------------------------
    potential = MLPotential('mace', modelPath=mace_model)
    system    = potential.createSystem(topology)

    # Langevin NVT integrator with 0.5 fs timestep
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        0.5 * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)

    # Prefer CUDA; fall back to CPU automatically
    try:
        platform   = openmm.Platform.getPlatformByName('CUDA')
        simulation = app.Simulation(topology, system, integrator, platform)
        print("Platform    : CUDA")
    except Exception:
        simulation = app.Simulation(topology, system, integrator)
        print("Platform    : CPU (CUDA unavailable)")

    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)

    # Quick energy minimisation before production
    print("Minimising energy...")
    simulation.minimizeEnergy()

    # ------------------------------------------------------------------
    # Production MD: collect one frame every sample_freq steps
    # ------------------------------------------------------------------
    n_frames = n_steps // sample_freq
    traj_frames = []

    print(f"Running MD ({n_frames} frames to save)...")
    for i in range(n_frames):
        simulation.step(sample_freq)
        state = simulation.context.getState(
            getPositions=True,
            getForces=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )

        pos_nm   = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces   = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        pe_kj    = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        frame = ase.Atoms(
            symbols=init_atoms.get_chemical_symbols(),
            positions=pos_nm * NM_TO_ANGSTROM,
            cell=init_atoms.get_cell(),
            pbc=init_atoms.get_pbc(),
        )
        # Store forces (eV/Å) and energy (eV) in the extxyz frame
        frame.arrays['forces'] = forces * KJ_MOL_NM_TO_EV_ANG
        frame.info['energy']   = pe_kj * KJ_MOL_TO_EV
        traj_frames.append(frame)

    ase.io.write(output_traj, traj_frames, format='extxyz')
    print(f"Saved {len(traj_frames)} frames to {output_traj}")


if __name__ == '__main__':
    main()
