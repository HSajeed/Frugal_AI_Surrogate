import yaml
import time
import os
from scripts.crossval import run_crossval

def sweep_architectures(config_path="config/config.yaml", depths=[1, 2, 3], widths=[32, 64, 128]):
    # Load base config
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Create sweep folder
    sweep_id = time.strftime("%Y%m%d_%H%M%S")
    sweep_folder = f"results/sweep_{sweep_id}"
    os.makedirs(sweep_folder, exist_ok=True)

    for depth in depths:
        for width in widths:
            print(f"\n🧪 Testing depth={depth}, width={width}")

            # Patch config
            config = base_config.copy()
            config["model"]["hidden_layers"] = [width] * depth

            # Create run folder
            run_name = f"depth_{depth}_width_{width}"
            run_folder = os.path.join(sweep_folder, run_name)
            os.makedirs(run_folder, exist_ok=True)

            # Run cross-validation
            fold_metrics = run_crossval(config, n_splits=5, run_folder=run_folder)

            # Save config and metrics
            with open(os.path.join(run_folder, "config_used.yaml"), "w") as f:
                yaml.dump(config, f)
            with open(os.path.join(run_folder, "fold_metrics.json"), "w") as f:
                import json
                json.dump(fold_metrics, f, indent=2)

if __name__ == "__main__":
    sweep_architectures()
