import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

LW = 2.0
FS = 18
TICKFS = 14
TIME_PER_FES_NS = 0.000500 * 100 * 100 / 1000
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


def compute_delta_g(fes_path: str, ts: float) -> float:
    cv_fes = np.loadtxt(fes_path, usecols=(0, 1))
    if cv_fes.ndim != 2 or cv_fes.shape[1] != 2:
        raise ValueError(f"Invalid FES data in {fes_path}")

    cv = cv_fes[:, 0]
    fes = cv_fes[:, 1]

    is_mask = cv < ts
    fs_mask = cv > ts
    if not np.any(is_mask) or not np.any(fs_mask):
        raise ValueError(f"Cannot identify IS/FS around ts={ts} in {fes_path}")

    abs_fes_is = np.min(fes[is_mask])
    abs_fes_fs = np.min(fes[fs_mask])
    return abs_fes_fs - abs_fes_is


def cumulative_average(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.cumsum(array) / np.arange(1, len(array) + 1)


def collect_curves(in_dir_pattern: str, skip_fes: int, ts: float) -> list[dict]:
    job_dirs = sorted(d for d in glob.glob(in_dir_pattern) if os.path.isdir(d))
    if not job_dirs:
        raise FileNotFoundError(f"No directories found matching {in_dir_pattern!r}")

    curves = []
    for job_dir in job_dirs:
        temp = read_temperature(job_dir)
        fes_files = glob.glob(os.path.join(job_dir, "out", "fes_*.dat"))
        if not fes_files:
            raise FileNotFoundError(f"No fes_*.dat found in {job_dir}/out")

        fes_files = sorted(fes_files, key=extract_fes_index)
        if skip_fes > 0:
            fes_files = fes_files[skip_fes:]
        if not fes_files:
            raise ValueError(f"No FES slices left after skip_fes={skip_fes} for {job_dir}")

        times = []
        delta_g = []
        for fes_path in fes_files:
            fes_idx = extract_fes_index(fes_path)
            times.append(fes_idx * TIME_PER_FES_NS)
            delta_g.append(compute_delta_g(fes_path, ts))

        curves.append(
            {
                "job_dir": job_dir,
                "temp": temp,
                "time_ns": np.array(times),
                "delta_g": cumulative_average(delta_g),
            }
        )

    curves.sort(key=lambda item: item["temp"])
    return curves


def plot_curves(curves: list[dict], out_file: str, y_min: float, y_max: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    cmap = plt.get_cmap("coolwarm")
    n_curves = len(curves)
    color_positions = np.linspace(0.08, 0.92, n_curves) if n_curves > 1 else [0.1]

    for curve, color_pos in zip(curves, color_positions):
        ax.plot(
            curve["time_ns"],
            curve["delta_g"],
            color=cmap(color_pos),
            linewidth=LW,
            label=f'{int(round(curve["temp"]))} K',
        )

    ax.set_xlabel("Time (ns)", fontsize=FS, fontstyle="italic")
    ax.set_ylabel(r"$\Delta_r\ G$ ($kJ/mol$)", fontsize=FS)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", which="major", direction="out", labelsize=TICKFS)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    ax.legend(
        frameon=False,
        fontsize=12,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def run(
    in_dir_pattern: str,
    out_file: str = "delta-g.png",
    skip_fes: int = 0,
    ts: float = 1.9,
    y_min: float = 100,
    y_max: float = 400,
) -> None:
    curves = collect_curves(in_dir_pattern, skip_fes=skip_fes, ts=ts)
    plot_curves(curves, out_file, y_min=y_min, y_max=y_max)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir_pattern")
    parser.add_argument("--out_file", default="delta-g.png")
    parser.add_argument("--skip_fes", type=int, default=0)
    parser.add_argument("--ts", type=float, default=1.9)
    parser.add_argument("--y_min", type=float, default=100)
    parser.add_argument("--y_max", type=float, default=400)
    args = parser.parse_args()
    run(
        args.in_dir_pattern,
        out_file=args.out_file,
        skip_fes=args.skip_fes,
        ts=args.ts,
        y_min=args.y_min,
        y_max=args.y_max,
    )
