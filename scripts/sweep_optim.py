import yaml
import time
import os
import json
from scripts.crossval import run_crossval

def sweep_optimizers(config_path="config/config.yaml", optimizers=["adam", "sgd", "rmsprop", "adagrad"]):
    # Load base config
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Create sweep folder
    sweep_id = time.strftime("%Y%m%d_%H%M%S")
    sweep_folder = f"results/sweep_optim_{sweep_id}"
    os.makedirs(sweep_folder, exist_ok=True)

    for opt in optimizers:
        print(f"\n🧪 Testing optimizer: {opt}")

        # Patch config
        config = base_config.copy()
        config["training"]["optimizer"] = opt

        # Create run folder
        run_name = f"opt_{opt}"
        run_folder = os.path.join(sweep_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)

        # Run cross-validation
        fold_metrics = run_crossval(config, n_splits=5, run_folder=run_folder)

        # Save config and metrics
        with open(os.path.join(run_folder, "config_used.yaml"), "w") as f:
            yaml.dump(config, f)
        with open(os.path.join(run_folder, "fold_metrics.json"), "w") as f:
            json.dump(fold_metrics, f, indent=2)

if __name__ == "__main__":
    sweep_optimizers()
