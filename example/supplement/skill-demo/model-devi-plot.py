import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import fire
import matplotlib
import pandas as pd
from typing import List

matplotlib.use('agg')

# Constants for styling reference from dp-test.py (if applicable)
LW = 2
FS = 18
TICKFS = 14

def parse_tsv(file_path: str):
    """
    Parse the tsv file.
    Aggregate values from all entries except the SUMMARY row.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None

            header = lines[0].split()
            try:
                idx_file = header.index('file')
                idx_total = header.index('total')
                idx_good = header.index('good')
                idx_decent = header.index('decent')
                idx_poor = header.index('poor')
            except ValueError as e:
                print(f"Missing required column in {file_path}: {e}")
                return None

            data_rows = []
            for line in lines[1:]:
                parts = line.split()
                if not parts or parts[idx_file] == 'SUMMARY':
                    continue

                data_rows.append({
                    'total': int(parts[idx_total]),
                    'good': int(parts[idx_good]),
                    'decent': int(parts[idx_decent]),
                    'poor': int(parts[idx_poor])
                })

            if not data_rows:
                return None

            total = sum(r['total'] for r in data_rows)
            good = sum(r['good'] for r in data_rows)
            decent = sum(r['decent'] for r in data_rows)
            poor = sum(r['poor'] for r in data_rows)

        return {
            'file': file_path,
            'total': total,
            'good': good,
            'decent': decent,
            'poor': poor
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def run(*files, out_file="./model-devi-stats.png"):
    """
    CLI tool to plot model-devi stats.
    files: glob patterns for tsv files.
    """
    all_files = []
    for pattern in files:
        all_files.extend(glob.glob(pattern))

    all_files = sorted(list(set(all_files)))

    if not all_files:
        print("No files found.")
        return

    results = []
    for f in all_files:
        res = parse_tsv(f)
        if res:
            results.append(res)

    if not results:
        print("No valid data parsed.")
        return

    # Prepare data for plotting
    df_plot = pd.DataFrame(results)

    # Calculate percentages
    df_plot['good_ratio'] = df_plot['good'] / df_plot['total'] * 100
    df_plot['decent_ratio'] = df_plot['decent'] / df_plot['total'] * 100
    df_plot['poor_ratio'] = df_plot['poor'] / df_plot['total'] * 100

    # Use iteration number from 1 to N
    indices = np.arange(len(df_plot))
    labels = [str(i + 1) for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    width = 0.6

    # Stacked bar plot (good, decent, poor)
    ax.bar(indices, df_plot['good_ratio'], width, label='Accurate', color='#8cbfde', edgecolor='gray', linewidth=0.5)
    ax.bar(indices, df_plot['decent_ratio'], width, bottom=df_plot['good_ratio'], label='Candidate', color='#fda072', edgecolor='gray', linewidth=0.5)
    ax.bar(indices, df_plot['poor_ratio'], width, bottom=df_plot['good_ratio'] + df_plot['decent_ratio'], label='Failed', color='#e67d7d', edgecolor='gray', linewidth=0.5)

    ax.set_xlabel('Iteration', fontsize=FS)
    ax.set_ylabel('Percentage (%)', fontsize=FS)
    ax.set_title('Model Deviation Statistics', fontsize=FS)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=TICKFS)
    ax.tick_params(axis='y', labelsize=TICKFS)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    ax.legend(loc='lower right', fontsize=TICKFS)

    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    fire.Fire(run)
