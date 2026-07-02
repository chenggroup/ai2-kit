import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os
import glob
import fire
import matplotlib
matplotlib.use('agg')

# 设置参数
LW2 = 2
LW = 2
FS = 18
TICKFS = 14

def calculate_metrics(y_true, y_pred):
    """计算统计指标"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return rmse, mae, r2

def get_symmetric_limit(*arrays, pad_ratio=0.08, min_limit=1e-3):
    """根据数据自动生成对称坐标范围，并留出边界。"""
    max_abs = 0.0
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        arr_max = np.nanmax(np.abs(arr))
        if np.isfinite(arr_max):
            max_abs = max(max_abs, arr_max)

    if max_abs == 0:
        max_abs = min_limit

    return max(max_abs * (1.0 + pad_ratio), min_limit)

def plot_comparison_with_error_dist(all_dft_energy, all_mlp_energy, all_dft_forces, all_mlp_forces, output="val-dp-test.png"):
    """绘制能量和力的对比图及其误差分布"""
    rmse_energy, mae_energy, r2_energy = calculate_metrics(all_dft_energy, all_mlp_energy)
    rmse_forces, mae_forces, r2_forces = calculate_metrics(all_dft_forces, all_mlp_forces)

    # 创建主图
    fig = plt.figure(figsize=(18.5, 3.5), dpi=300)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.35, hspace=0)

    # 计算误差
    energy_errors = all_mlp_energy - all_dft_energy
    force_errors = all_mlp_forces - all_dft_forces

    # === 能量散点图 ===
    ax1 = plt.subplot(gs[0])
    energy_limit = get_symmetric_limit(all_dft_energy, all_mlp_energy, pad_ratio=0.08, min_limit=0.01)
    ax1.scatter(all_dft_energy, all_mlp_energy, s=20, color='#fda072', alpha=0.7)

    # 在左下角添加误差分布直方图
    bbox1 = ax1.get_position()
    hist_width = 0.053
    hist_height = 0.22
    hist_left = (bbox1.x0 + bbox1.x1) * 0.585
    hist_bottom = (bbox1.y0 + bbox1.y1) * 0.238
    ax1_hist = fig.add_axes([hist_left, hist_bottom, hist_width, hist_height])

    data = energy_errors * 1000
    max_err = np.max(np.abs(data))
    weights = np.ones_like(data) * (100.0 / len(data))
    bins = np.linspace(-max_err, max_err, 31)

    ax1_hist.hist(data, bins=bins, weights=weights, color='#fda072', alpha=0.7, edgecolor='gray', linewidth=0.8)
    ax1_hist.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=7, length=2)
    ax1_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    ax1_hist.set_xlabel('Error (meV/atom)', fontsize=9)
    ax1_hist.set_title('Error Distribution', fontsize=9)
    ax1_hist.set_xlim(-max_err, max_err)
    ax1_hist.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    ax1_hist.tick_params(axis='x', labelsize=7, length=2)

    # === 力散点图 ===
    ax2 = plt.subplot(gs[1])
    force_limit = get_symmetric_limit(all_dft_forces, all_mlp_forces, pad_ratio=0.08, min_limit=0.5)
    ax2.scatter(all_dft_forces, all_mlp_forces, s=20, color='#8cbfde', alpha=0.7)

    bbox2 = ax2.get_position()
    hist_left = (bbox2.x0 + bbox2.x1) * 0.54
    hist_bottom = (bbox2.y0 + bbox2.y1) * 0.239
    hist_width = 0.053
    hist_height = 0.22

    data = force_errors
    max_err_f = np.max(np.abs(data))
    weights = np.ones_like(data) * (100.0 / len(data))
    bins = np.linspace(-max_err_f, max_err_f, 41)

    ax2_hist = fig.add_axes([hist_left, hist_bottom, hist_width, hist_height])
    ax2_hist.hist(data, bins=bins, weights=weights, color='#8cbfde', alpha=0.9, edgecolor='gray', linewidth=0.8)
    ax2_hist.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=7, length=2)
    ax2_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    ax2_hist.set_xlabel('Error (eV/Å)', fontsize=9)
    ax2_hist.set_title('Error Distribution', fontsize=9)
    ax2_hist.set_xlim(-max_err_f, max_err_f)
    ax2_hist.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    ax2_hist.tick_params(axis='x', labelsize=7, length=2)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(LW2)
        ax.tick_params(which='major', direction='out', width=LW2, labelsize=FS-6)

    ax1.plot([-energy_limit, energy_limit], [-energy_limit, energy_limit], 'k--', linewidth=LW2)
    ax2.plot([-force_limit, force_limit], [-force_limit, force_limit], 'k--', linewidth=LW2)

    ax1.set_xlim(-energy_limit, energy_limit)
    ax1.set_ylim(-energy_limit, energy_limit)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax1.set_xlabel(r'$\rm E_{PBE-D3}$ (eV/atom)', fontsize=FS, labelpad=6)
    ax1.set_ylabel(r'$\rm E_{MLP}$ (eV/atom)', fontsize=FS)

    ax2.set_xlim(-force_limit, force_limit)
    ax2.set_ylim(-force_limit, force_limit)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax2.set_xlabel(r'$\rm F^i_{PBE-D3}$ (eV/Å)', fontsize=FS)
    ax2.set_ylabel(r'$\rm F^i_{MLP}$ (eV/Å)', fontsize=FS, labelpad=4)

    stats_text1 = (f"RMSE: {rmse_energy:.3e} eV/atom\n"
                   f"MAE: {mae_energy:.3e} eV/atom\n"
                   f"R² = {r2_energy:.3f}")
    ax1.text(0.03, 0.97, stats_text1, transform=ax1.transAxes, fontsize=FS-6, va="top", color="black")

    stats_text2 = (f"RMSE: {rmse_forces:.3e} eV/Å\n"
                   f"MAE: {mae_forces:.3e} eV/Å\n"
                   f"R² = {r2_forces:.3f}")
    ax2.text(0.03, 0.97, stats_text2, transform=ax2.transAxes, fontsize=FS-6, va="top", color="black")

    plt.savefig(output, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Results saved to {output}")

def run(result_prefix, output="val-dp-test.png"):
    """
    Run the comparison of DFT and MLP results based on result prefix.

    Usage Examples:
    1. Default search for prefixes like '000' in current directory:
       python3 dp-test.py

    2. Specify a prefix:
       python3 dp-test.py --result_prefix="./test_data/test"

    3. Use glob in prefix to match multiple runs:
       python3 dp-test.py --result_prefix="30-dp-test/*"

    :param result_prefix: Glob pattern for the result prefix (e.g., '**/000' or '30-dp-test/test').
    :param output: Output plot filename.
    """
    # 查找所有的能量文件来确定 prefix 分支
    energy_pattern = result_prefix + ".e_peratom.out"
    energy_files = sorted(glob.glob(energy_pattern, recursive=True))

    if not energy_files:
        print(f"Error: No files found matching prefix pattern: {energy_pattern}")
        return

    print(f"Found {len(energy_files)} result sets matching prefix.")

    all_dft_energy = []
    all_mlp_energy = []
    all_dft_forces = []
    all_mlp_forces = []

    for energy_file in energy_files:
        # 推导对应的力文件路径
        prefix = energy_file.replace(".e_peratom.out", "")
        force_file = prefix + ".f.out"

        if not os.path.exists(force_file):
            print(f"Warning: Force file not found for {energy_file}, skipping.")
            continue

        try:
            energy_data = np.loadtxt(energy_file)
            dft_energy = energy_data[:, 0]
            mlp_energy = energy_data[:, 1]
            all_dft_energy.append(dft_energy - np.mean(dft_energy))
            all_mlp_energy.append(mlp_energy - np.mean(dft_energy))

            force_data = np.loadtxt(force_file)
            dft_forces = force_data[:, :3].flatten()
            mlp_forces = force_data[:, 3:].flatten()
            all_dft_forces.append(dft_forces)
            all_mlp_forces.append(mlp_forces)
        except Exception as e:
            print(f"Error loading data from {prefix}: {e}")

    if not all_dft_energy:
        print("No data loaded.")
        return

    all_dft_energy = np.concatenate(all_dft_energy)
    all_mlp_energy = np.concatenate(all_mlp_energy)
    all_dft_forces = np.concatenate(all_dft_forces)
    all_mlp_forces = np.concatenate(all_mlp_forces)

    print(f"Total data points - Energy: {len(all_dft_energy)}, Forces: {len(all_dft_forces)}")

    plot_comparison_with_error_dist(all_dft_energy, all_mlp_energy, all_dft_forces, all_mlp_forces, output=output)

if __name__ == '__main__':
    fire.Fire(run)
