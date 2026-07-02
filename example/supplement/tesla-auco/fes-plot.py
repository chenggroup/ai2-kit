import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import fire
import subprocess
import matplotlib
matplotlib.use('agg')

# Plotting parameters from dp-test.py for professional look
LW2 = 2
LW = 2
FS = 18
TICKFS = 14

def aver_sum(array):
    total = 0
    averages = []
    for i, value in enumerate(array):
        total += value
        average = total / (i + 1)
        averages.append(average)
    return np.array(averages)

def run(in_dir_pattern: str, plumed_cmd="plumed sum_hills --bin 99 --min 1.0 --max 5.5 --hills ../HILLS --stride 100", out_prefix="", skip_fes=200, ts=1.9, y_min=100, y_max=400, td_y_min=100, td_y_max=300):
    """
    Run plumed sum_hills and analyze the generated FES files to yield reaction and activation free energies.

    :param in_dir_pattern: Glob pattern for directory/directories where the FES files will be generated and processed.
    :param plumed_cmd: The PLUMED command to execute before analysis.
    :param out_prefix: Output prefix for the plot files.
    :param y_min: Minimum value for the y-axis of all plots (default: 100).
    :param y_max: Maximum value for the y-axis of all plots (default: 400).
    :param td_y_min: Minimum y-axis for the temperature dependence plot (default: 100).
    :param td_y_max: Maximum y-axis for the temperature dependence plot (default: 300).
    """
    matched_paths = sorted([d for d in glob.glob(in_dir_pattern) if os.path.isdir(d)])
    if not matched_paths:
        print(f"Error: No directories found matching '{in_dir_pattern}'.")
        return

    results = []

    for d in matched_paths:
        out_dir = os.path.join(d, "out")
        os.makedirs(out_dir, exist_ok=True)

        time_fes = glob.glob(os.path.join(out_dir, 'fes_*.dat'))
        if not time_fes:
            print(f"Executing PLUMED command in '{out_dir}': {plumed_cmd}")
            try:
                subprocess.run(plumed_cmd, shell=True, cwd=out_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing PLUMED command in {out_dir}: {e}")
                continue
            time_fes = glob.glob(os.path.join(out_dir, 'fes_*.dat'))
        else:
            print(f"Skipping PLUMED execution for '{d}', data already exists in '{out_dir}'.")

        if not time_fes:
            print(f"Warning: No 'fes_*.dat' files generated in '{out_dir}'.")
            continue

        def extract_idx(f):
            base = os.path.basename(f)
            try:
                val = base.replace('fes_', '').replace('.dat', '')
                return int(val)
            except ValueError:
                return -1

        time_fes = sorted(time_fes, key=extract_idx)

        # 示例代码指出：必须跳过初期（如前200个）的 FES 输出
        # 因为初始阶段采样不足，直接放入 aver_sum 会导致平均值出现严重偏差
        time_fes = time_fes[skip_fes:]

        fes_is = []
        fes_fs = []
        fes_ts = []

        loaded_files = 0
        for f in time_fes:
            try:
                # 读取计算好的一维 FES 数据 (cv, fes)
                cv_fes = np.loadtxt(f)
                if cv_fes.size == 0 or len(cv_fes.shape) < 2:
                    continue

                cv_ts = ts
                # 寻找反应物状态 (is, Initial State) 的自由能最小值，条件：cv < ts
                abs_fes_is = np.min(cv_fes[:, 1][(cv_fes[:, 0] < cv_ts)])

                # 寻找过渡态 (ts, Transition State) 的自由能最大值，条件：在 ts 附近 (ts-0.5 到 ts+0.5)
                abs_fes_ts = np.max(cv_fes[:, 1][(cv_fes[:, 0] > cv_ts - 0.5) & (cv_fes[:, 0] < cv_ts + 0.5)])

                # 寻找产物状态 (fs, Final State) 的自由能最小值，条件：cv > ts
                abs_fes_fs = np.min(cv_fes[:, 1][(cv_fes[:, 0] > cv_ts)])

                # 更新 cv_ts 为实际的过渡态 cv 坐标 (示例代码中有这步，可选)
                # cv_ts = cv_fes[:,0][cv_fes[:,1]==abs_fes_ts]

                fes_fs.append(abs_fes_fs)
                fes_ts.append(abs_fes_ts)
                fes_is.append(abs_fes_is)
                loaded_files += 1
            except Exception as e:
                pass

        if loaded_files == 0:
            print(f"Error: No valid FES data could be loaded for '{d}'. (May be due to skip_fes={skip_fes} being too large)")
            continue

        # 计算反应自由能 deA (Final - Initial)
        deA = np.array(fes_fs) - np.array(fes_is)
        # 计算活化自由能 acA (Transition - Initial)
        acA = np.array(fes_ts) - np.array(fes_is)

        # 计算滚动平均值
        av_deA = aver_sum(deA)
        av_acA = aver_sum(acA)

        # 将结果保存到 dat 文件
        dat_out = os.path.join(out_dir, f"dA{skip_fes}.dat")
        np.savetxt(dat_out, np.vstack((deA, acA, av_deA, av_acA)).T,
                   header='Raw_dG\tRaw_barrier\tCumAvg_dG\tCumAvg_barrier')

        # Read TEMP file
        temp_file = os.path.join(d, 'TEMP')
        temperature = None
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r') as tf:
                    temperature = float(tf.read().strip())
            except Exception:
                pass

        n_stable = max(len(deA) // 2, 1)
        results.append({
            'dir': d,
            'temp': temperature,
            'deA': deA, 'acA': acA,
            'av_deA': av_deA, 'av_acA': av_acA,
            'final_dG': np.mean(deA[-n_stable:]),
            'error_dG': np.std(deA[-n_stable:]),
            'final_barrier': np.mean(acA[-n_stable:]),
            'error_barrier': np.std(acA[-n_stable:]),
            'n_stable': n_stable,
        })

    if not results:
        print("Error: No analysis results were produced.")
        return

    # Sort results by temperature
    results.sort(key=lambda x: x['temp'] if x['temp'] is not None else 0)

    if len(results) == 1:
        # Single directory plotting mode
        res = results[0]
        frames = np.arange(len(res['deA']))
        std_deA = np.std(res['deA'])
        std_acA = np.std(res['acA'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

        # Top-left: ΔG convergence with ±1σ
        axes[0, 0].plot(frames, res['av_deA'], 'b-', linewidth=LW2, label='Cumulative average')
        axes[0, 0].fill_between(frames,
                                res['av_deA'] - std_deA,
                                res['av_deA'] + std_deA,
                                alpha=0.3, label=r'$\pm 1\sigma$')
        axes[0, 0].axhline(y=res['av_deA'][-1], color='r', linestyle='--', label='Converged')
        axes[0, 0].text(0.97, 0.05,
                        f"Converged: {res['av_deA'][-1]:.2f} kJ/mol",
                        transform=axes[0, 0].transAxes, fontsize=FS-2, color='black',
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))
        axes[0, 0].set_xlabel('Frame', fontsize=FS)
        axes[0, 0].set_ylabel(r'$\Delta G$ (kJ/mol)', fontsize=FS)
        axes[0, 0].set_title('Reaction Free Energy Convergence', fontsize=FS)
        axes[0, 0].legend(fontsize=FS-6)
        axes[0, 0].grid(True)
        axes[0, 0].tick_params(which='major', direction='out', width=LW2, labelsize=TICKFS)
        axes[0, 0].set_ylim(y_min, y_max)

        # Top-right: ΔG‡ convergence with ±1σ
        axes[0, 1].plot(frames, res['av_acA'], 'g-', linewidth=LW2, label='Cumulative average')
        axes[0, 1].fill_between(frames,
                                res['av_acA'] - std_acA,
                                res['av_acA'] + std_acA,
                                alpha=0.3, label=r'$\pm 1\sigma$')
        axes[0, 1].axhline(y=res['av_acA'][-1], color='r', linestyle='--', label='Converged')
        axes[0, 1].text(0.97, 0.05,
                        f"Converged: {res['av_acA'][-1]:.2f} kJ/mol",
                        transform=axes[0, 1].transAxes, fontsize=FS-2, color='black',
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))
        axes[0, 1].set_xlabel('Frame', fontsize=FS)
        axes[0, 1].set_ylabel(r'$\Delta G^\ddagger$ (kJ/mol)', fontsize=FS)
        axes[0, 1].set_title('Activation Free Energy Convergence', fontsize=FS)
        axes[0, 1].legend(fontsize=FS-6)
        axes[0, 1].grid(True)
        axes[0, 1].tick_params(which='major', direction='out', width=LW2, labelsize=TICKFS)
        axes[0, 1].set_ylim(y_min, y_max)

        # Bottom-left: Raw vs cumulative (ΔG)
        axes[1, 0].plot(frames, res['deA'], 'b-', alpha=0.5, linewidth=0.5, label='Raw data')
        axes[1, 0].plot(frames, res['av_deA'], 'r-', linewidth=LW2, label='Cumulative average')
        _y_raw10 = res['deA'][-1]
        _y_cum10 = res['av_deA'][-1]
        axes[1, 0].text(0.97, 0.05,
                        f"Avg: {_y_cum10:.2f} kJ/mol\nRaw (last): {_y_raw10:.2f} kJ/mol",
                        transform=axes[1, 0].transAxes, fontsize=FS-2, color='black',
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))
        axes[1, 0].set_xlabel('Frame', fontsize=FS)
        axes[1, 0].set_ylabel(r'$\Delta G$ (kJ/mol)', fontsize=FS)
        axes[1, 0].set_title(r'Raw Data vs Cumulative Average ($\Delta G$)', fontsize=FS)
        axes[1, 0].legend(fontsize=FS-6)
        axes[1, 0].grid(True)
        axes[1, 0].tick_params(which='major', direction='out', width=LW2, labelsize=TICKFS)
        axes[1, 0].set_ylim(y_min, y_max)

        # Bottom-right: Raw vs cumulative (ΔG‡)
        axes[1, 1].plot(frames, res['acA'], 'g-', alpha=0.5, linewidth=0.5, label='Raw data')
        axes[1, 1].plot(frames, res['av_acA'], 'r-', linewidth=LW2, label='Cumulative average')
        _y_raw11 = res['acA'][-1]
        _y_cum11 = res['av_acA'][-1]
        axes[1, 1].text(0.97, 0.05,
                        f"Avg: {_y_cum11:.2f} kJ/mol\nRaw (last): {_y_raw11:.2f} kJ/mol",
                        transform=axes[1, 1].transAxes, fontsize=FS-2, color='black',
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))
        axes[1, 1].set_xlabel('Frame', fontsize=FS)
        axes[1, 1].set_ylabel(r'$\Delta G^\ddagger$ (kJ/mol)', fontsize=FS)
        axes[1, 1].set_title(r'Raw Data vs Cumulative Average ($\Delta G^\ddagger$)', fontsize=FS)
        axes[1, 1].legend(fontsize=FS-6)
        axes[1, 1].grid(True)
        axes[1, 1].tick_params(which='major', direction='out', width=LW2, labelsize=TICKFS)
        axes[1, 1].set_ylim(y_min, y_max)

        title = f"Temperature: {res['temp']} K" if res['temp'] is not None else "Free Energy Convergence"
        fig.suptitle(title, fontsize=FS)
        plt.tight_layout()

        out_png = f"{out_prefix}converge-{skip_fes}.png"
        plt.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"FES analysis plot saved to {out_png}")

        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Number of frames used: {len(res['deA'])}")
        print(f"Stable region (last {res['n_stable']} frames):")
        print(f"  Reaction Free Energy (\u0394G): {res['final_dG']:.2f} \u00b1 {res['error_dG']:.2f} kJ/mol")
        print(f"  Activation Free Energy (\u0394G\u2021): {res['final_barrier']:.2f} \u00b1 {res['error_barrier']:.2f} kJ/mol")
        print("="*50)

    else:
        # Multiple directories plotting mode
        n = len(results)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # 1. Grid Plot
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=300)
        axes_flat = axes.flatten() if n > 1 else [axes]

        for idx, ax in enumerate(axes_flat):
            if idx < n:
                res = results[idx]
                frames = np.arange(len(res['deA']))
                ax.plot(frames, res['deA'], color='#fda072', alpha=0.4, linewidth=LW)
                ax.plot(frames, res['acA'], color='#8cbfde', alpha=0.4, linewidth=LW)
                ax.plot(frames, res['av_deA'], color='#d95f02', linewidth=LW2+1, label=r'$\Delta G$')
                ax.plot(frames, res['av_acA'], color='#3182bd', linewidth=LW2+1, label=r'$\Delta G^\ddagger$')

                for spine in ax.spines.values(): spine.set_linewidth(LW2)
                ax.tick_params(which='major', direction='out', width=LW2, labelsize=FS-8)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('Frame', fontsize=FS-4)
                ax.set_ylabel(r'$\Delta G$ (kJ/mol)', fontsize=FS-4)

                title = f"{res['temp']} K" if res['temp'] is not None else res['dir']
                ax.set_title(title, fontsize=FS-2)

                _y_dG = res['av_deA'][-1]
                _y_bar = res['av_acA'][-1]
                ax.text(0.97, 0.05,
                        f"\u0394G: {res['final_dG']:.2f}\u00b1{res['error_dG']:.2f}\n"
                        f"\u0394G\u2021: {res['final_barrier']:.2f}\u00b1{res['error_barrier']:.2f}",
                        transform=ax.transAxes, fontsize=FS-2, color='black',
                        va='bottom', ha='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))
                if idx == 0:
                    ax.legend(fontsize=FS-8, loc='best')
            else:
                ax.axis('off')

        plt.tight_layout()
        grid_out = f"{out_prefix}convergence_grid.png"
        plt.savefig(grid_out, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Convergence grid plot saved to {grid_out}")

        # 2. Temperature Dependence Plot
        valid_temps = [r for r in results if r['temp'] is not None]
        if valid_temps:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            temps = [r['temp'] for r in valid_temps]
            deA_vals = [r['final_dG'] for r in valid_temps]
            acA_vals = [r['final_barrier'] for r in valid_temps]

            ax.plot(temps, deA_vals, 'o-', color='#d95f02', linewidth=LW2+1, markersize=8, label=r'Reaction $\Delta G$')
            ax.plot(temps, acA_vals, 's-', color='#3182bd', linewidth=LW2+1, markersize=8, label=r'Activation $\Delta G^\ddagger$')

            for spine in ax.spines.values(): spine.set_linewidth(LW2)
            ax.tick_params(which='major', direction='out', width=LW2, labelsize=FS-6)

            ax.set_xlabel('Temperature (K)', fontsize=FS, labelpad=6)
            ax.set_ylabel(r'$\Delta G$ (kJ/mol)', fontsize=FS)
            ax.set_title(r'$\Delta G$ vs Temperature', fontsize=FS)
            ax.set_ylim(td_y_min, td_y_max)
            ax.legend(fontsize=FS-4, loc='best')

            temp_out = f"{out_prefix}temp_dependence.png"
            plt.savefig(temp_out, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Temperature dependence plot saved to {temp_out}")

if __name__ == '__main__':
    fire.Fire(run)
