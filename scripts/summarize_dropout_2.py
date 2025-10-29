import os, json, numpy as np, matplotlib.pyplot as plt

# 🔍 Locate results directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_root, "results")
print(f"\n🔍 Looking in: {results_dir}")

# 🔎 Find dropout sweep folders
sweep_folders = [f for f in os.listdir(results_dir) if f.startswith("sweep_dropout_2_")]
print(f"📁 Found dropout sweep folders: {sweep_folders}")

all_metrics = []
for sweep in sweep_folders:
    sweep_path = os.path.join(results_dir, sweep)
    print(f"\n📂 Checking sweep folder: {sweep_path}")

    for run in os.listdir(sweep_path):
        dropout = run.replace("dropout_", "")
        run_path = os.path.join(sweep_path, run)
        metrics_path = os.path.join(run_path, "fold_metrics.json")

        print(f"   🔸 Found dropout run: {run}")
        print(f"   📄 Looking for: {metrics_path}")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                fold_metrics = json.load(f)
            all_metrics.append({
                "dropout": float(dropout),
                "r2_nu": np.mean([m["r2_nu"] for m in fold_metrics]),
                "r2_dp": np.mean([m["r2_dp"] for m in fold_metrics]),
                "rmse_nu": np.mean([m["rmse_nu"] for m in fold_metrics]),
                "rmse_dp": np.mean([m["rmse_dp"] for m in fold_metrics])
            })
        else:
            print(f"      ⚠️ Missing fold_metrics.json in: {run_path}")

# 📊 Print summary
print("\n📊 Dropout Sweep 2 Summary:")
for m in all_metrics:
    print(f"Dropout {m['dropout']:.2f} | R² Nu: {m['r2_nu']:.4f}, RMSE Nu: {m['rmse_nu']:.4f} | R² DP: {m['r2_dp']:.4f}, RMSE DP: {m['rmse_dp']:.4f}")

# 💾 Save summary
summary_path = os.path.join(results_dir, "dropout_2_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n✅ Summary saved to: {summary_path}")

# 📈 Plot
dropouts = [m["dropout"] for m in all_metrics]
x = np.arange(len(dropouts))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, [m["r2_nu"] for m in all_metrics], width, label="R² Nu")
ax.bar(x + width/2, [m["rmse_nu"] for m in all_metrics], width, label="RMSE Nu", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels([f"{d:.2f}" for d in dropouts])
ax.set_title("Dropout Performance")
ax.legend()
plt.tight_layout()

plot_path = os.path.join(results_dir, "dropout_2_summary_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"\n📈 Plot saved to: {plot_path}")
