import os, json, numpy as np, matplotlib.pyplot as plt


# project root (one level up from scripts/)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = "."

def summarize_activation_sweeps(results_dir=results_dir):
    sweep_folders = [f for f in os.listdir(results_dir) if f.startswith("sweep_activation_")]
    print(f"\n🔍 Found {len(sweep_folders)} sweep folders.")

    all_metrics = []
    for sweep in sweep_folders:
        sweep_path = os.path.join(results_dir, sweep)
        print(f"\n📁 Processing sweep folder: {sweep_path}")

        for run in os.listdir(sweep_path):
            act = run.replace("act_", "")
            run_path = os.path.join(sweep_path, run)
            metrics_path = os.path.join(run_path, "fold_metrics.json")

            print(f"   🔸 Found activation run: {run}")
            if os.path.exists(metrics_path):
                print(f"      ✅ Loading metrics from: {metrics_path}")
                with open(metrics_path) as f:
                    fold_metrics = json.load(f)
                all_metrics.append({
                    "activation": act,
                    "r2_nu": np.mean([m["r2_nu"] for m in fold_metrics]),
                    "r2_dp": np.mean([m["r2_dp"] for m in fold_metrics]),
                    "rmse_nu": np.mean([m["rmse_nu"] for m in fold_metrics]),
                    "rmse_dp": np.mean([m["rmse_dp"] for m in fold_metrics])
                })
            else:
                print(f"      ⚠️ Missing fold_metrics.json in: {run_path}")

    # Sort for consistent plotting
    all_metrics.sort(key=lambda x: x["activation"])

    print("\n📊 Activation Sweep Summary:")
    for m in all_metrics:
        print(f"{m['activation']:>10} | R² Nu: {m['r2_nu']:.4f}, RMSE Nu: {m['rmse_nu']:.4f} | R² DP: {m['r2_dp']:.4f}, RMSE DP: {m['rmse_dp']:.4f}")

    # Save summary
    summary_path = os.path.join(results_dir, "activation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_path}")

    # Plot
    acts = [m["activation"] for m in all_metrics]
    x = np.arange(len(acts))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, [m["r2_nu"] for m in all_metrics], width, label="R² Nu")
    ax.bar(x + width/2, [m["rmse_nu"] for m in all_metrics], width, label="RMSE Nu", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(acts, rotation=45)
    ax.set_title("Activation Function Performance")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "activation_summary_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"\n📈 Plot saved to: {plot_path}")
