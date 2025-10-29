import yaml, time, os
from scripts.crossval import run_crossval

def sweep_dropout():
    dropout_values = [0.0, 0.1, 0.2, 0.3]
    with open("config/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    sweep_id = time.strftime("%Y%m%d_%H%M%S")
    sweep_folder = f"results/sweep_dropout_{sweep_id}"
    os.makedirs(sweep_folder, exist_ok=True)

    for d in dropout_values:
        print(f"\n🧪 Testing dropout: {d}")
        config = base_config.copy()
        config["model"]["dropout"] = d

        run_folder = os.path.join(sweep_folder, f"dropout_{d}")
        os.makedirs(run_folder, exist_ok=True)

        fold_metrics = run_crossval(config, n_splits=5, run_folder=run_folder)

        print(f"\n📊 Dropout {d} Summary:")
        for m in fold_metrics:
            print(f"Fold {m['fold']} | R² Nu: {m['r2_nu']:.4f}, RMSE Nu: {m['rmse_nu']:.4f} | R² DP: {m['r2_dp']:.4f}, RMSE DP: {m['rmse_dp']:.4f}")

if __name__ == "__main__":
    sweep_dropout()
