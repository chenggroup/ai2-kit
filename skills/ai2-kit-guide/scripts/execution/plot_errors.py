#!/usr/bin/env python3
"""Plot paired reference/prediction CSV values; no unit conversion or qualification."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot_errors(csv, quantity, unit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.genfromtxt(csv, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if values.dtype.names is None or not {"reference", "prediction"} <= set(values.dtype.names):
        raise ValueError("CSV requires reference,prediction header columns")
    reference = np.atleast_1d(values["reference"])
    prediction = np.atleast_1d(values["prediction"])
    if not reference.size or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("CSV must contain nonempty finite numeric pairs")
    error = prediction - reference
    metrics = {"quantity": quantity, "unit": unit, "samples": int(reference.size),
               "rmse": float(np.sqrt(np.mean(error ** 2))),
               "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
               "error_definition": "prediction - reference", "source_csv": str(Path(csv).resolve())}
    prefix = Path(output).expanduser().resolve()
    targets = [prefix.parent / f"{prefix.name}{suffix}" for suffix in (".png", ".pdf", ".metrics.json")]
    if any(path.exists() for path in targets):
        raise ValueError("Output already exists; choose a new output prefix")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(reference, prediction, s=8, alpha=0.4, rasterized=True)
    lo = min(reference.min(), prediction.min())
    hi = max(reference.max(), prediction.max())
    pad = (hi - lo) * 0.04 if hi > lo else max(abs(lo) * 0.04, 0.01)
    limits = (lo - pad, hi + pad)
    axes[0].plot(limits, limits, color="black", linewidth=1)
    axes[0].set(xlabel=f"Reference ({unit})", ylabel=f"Prediction ({unit})", xlim=limits, ylim=limits)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(f"{quantity}: n={reference.size}\nRMSE={metrics['rmse']:.4g} {unit}")
    axes[1].hist(error, bins=min(60, max(1, int(np.sqrt(reference.size)))), color="#2878a8")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel=f"Prediction − reference ({unit})", ylabel="Count", title="Residual distribution")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[0], dpi=200)
    fig.savefig(targets[1])
    plt.close(fig)
    targets[2].write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--quantity", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", required=True, help="Output prefix for PNG, PDF and metrics JSON")
    args = parser.parse_args()
    try:
        print(json.dumps(plot_errors(args.csv, args.quantity, args.unit, args.output), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
