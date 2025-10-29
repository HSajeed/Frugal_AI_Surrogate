import os, json, numpy as np

def summarize_layer_sweeps(results_dir="results"):
    sweep_folders = [f for f in os.listdir(results_dir) if f.startswith("sweep_")]

    all_metrics = []
    for sweep in sweep_folders:
        sweep_path = os.path.join(results_dir, sweep)
        for run in os.listdir(sweep_path):
            if "act_" in run: continue  # skip activation sweeps
            metrics_path = os.path.join(sweep_path, run, "fold_metrics.json")
            if os.path.exists(metrics_path):
                depth = int(run.split("_")[1])
                width = int(run.split("_")[3])
                with open(metrics_path) as f:
                    fold_metrics = json.load(f)
                all_metrics.append({
                    "depth": depth,
                    "width": width,
                    "r2_nu": np.mean([m["r2_nu"] for m in fold_metrics]),
                    "r2_dp": np.mean([m["r2_dp"] for m in fold_metrics]),
                    "rmse_nu": np.mean([m["rmse_nu"] for m in fold_metrics]),
                    "rmse_dp": np.mean([m["rmse_dp"] for m in fold_metrics])
                })

    print("\n📊 Layer Sweep Summary:")
    for m in all_metrics:
        print(f"Depth={m['depth']}, Width={m['width']} | R² Nu: {m['r2_nu']:.4f}, RMSE Nu: {m['rmse_nu']:.4f} | R² DP: {m['r2_dp']:.4f}, RMSE DP: {m['rmse_dp']:.4f}")
