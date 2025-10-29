import yaml
import time
import os
from scripts.crossval import run_crossval

def main():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create unique run folder
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"results/run_{run_id}"
    os.makedirs(run_folder, exist_ok=True)

    # Run cross-validation
    fold_metrics = run_crossval(config, n_splits=5, run_folder=run_folder)

    # Print summary
    print("\n📊 Cross-Validation Summary:")
    for m in fold_metrics:
        print(f"Fold {m['fold']} | Nu_avg R²: {m['r2_nu']:.4f}, RMSE: {m['rmse_nu']:.4f} | DeltaP R²: {m['r2_dp']:.4f}, RMSE: {m['rmse_dp']:.4f}")

    r2_nu_avg = sum(m["r2_nu"] for m in fold_metrics) / len(fold_metrics)
    r2_dp_avg = sum(m["r2_dp"] for m in fold_metrics) / len(fold_metrics)
    rmse_nu_avg = sum(m["rmse_nu"] for m in fold_metrics) / len(fold_metrics)
    rmse_dp_avg = sum(m["rmse_dp"] for m in fold_metrics) / len(fold_metrics)

    print(f"\n✅ Mean Nu_avg R²: {r2_nu_avg:.4f}, RMSE: {rmse_nu_avg:.4f}")
    print(f"✅ Mean DeltaP R²: {r2_dp_avg:.4f}, RMSE: {rmse_dp_avg:.4f}")

if __name__ == "__main__":
    main()
