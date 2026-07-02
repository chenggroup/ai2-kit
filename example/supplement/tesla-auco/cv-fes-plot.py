import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("agg")

LW = 1.8
FS = 18
TICKFS = 14

FES_RE = re.compile(r"fes_(\d+)\.dat$")


def read_temperature(job_dir: str) -> float:
    temp_file = os.path.join(job_dir, "TEMP")
    with open(temp_file, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def extract_fes_index(path: str) -> int:
    match = FES_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Invalid FES filename: {path}")
    return int(match.group(1))


def read_last_fes(job_dir: str) -> tuple[np.ndarray, np.ndarray]:
    fes_files = glob.glob(os.path.join(job_dir, "out", "fes_*.dat"))
    if not fes_files:
        raise FileNotFoundError(f"No fes_*.dat found in {job_dir}/out")

    last_fes = sorted(fes_files, key=extract_fes_index)[-1]
    data = np.loadtxt(last_fes, usecols=(0, 1))
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Invalid FES data in {last_fes}")
    return data[:, 0], data[:, 1]


def collect_curves(in_dir_pattern: str) -> list[dict]:
    job_dirs = sorted(d for d in glob.glob(in_dir_pattern) if os.path.isdir(d))
    if not job_dirs:
        raise FileNotFoundError(f"No directories found matching {in_dir_pattern!r}")

    curves = []
    for job_dir in job_dirs:
        temp = read_temperature(job_dir)
        cv, fes = read_last_fes(job_dir)
        curves.append(
            {
                "job_dir": job_dir,
                "temp": temp,
                "cv": cv,
                # Shift each curve so the minimum free energy is zero, matching the reference style.
                "fes": fes - np.min(fes),
            }
        )

    curves.sort(key=lambda item: item["temp"])
    return curves


def plot_curves(curves: list[dict], out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)

    cmap = plt.get_cmap("coolwarm")
    n_curves = len(curves)
    color_positions = np.linspace(0.05, 0.95, n_curves) if n_curves > 1 else [0.1]

    for curve, color_pos in zip(curves, color_positions):
        ax.plot(
            curve["cv"],
            curve["fes"],
            color=cmap(color_pos),
            linewidth=LW,
            label=f'{int(round(curve["temp"]))} K',
        )

    ax.set_xlabel("Distance (Å)", fontsize=FS)
    ax.set_ylabel(r"Free Energy ($kJ\ mol^{-1}$)", fontsize=FS)
    ax.tick_params(axis="both", which="major", direction="out", labelsize=TICKFS)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(
        ncol=min(5, n_curves),
        fontsize=11,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=0.8,
        columnspacing=1.4,
    )

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def run(in_dir_pattern: str, out_file: str = "cv-fes.png") -> None:
    curves = collect_curves(in_dir_pattern)
    plot_curves(curves, out_file)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir_pattern")
    parser.add_argument("--out_file", default="cv-fes.png")
    args = parser.parse_args()
    run(args.in_dir_pattern, out_file=args.out_file)
