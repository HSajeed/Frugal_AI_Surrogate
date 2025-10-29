import torch
import yaml
from scripts.preprocess import load_and_preprocess
from scripts.train import train_model
from scripts.evaluate import (
    inverse_transform_and_evaluate,
    plot_loss_curves,
    plot_predictions
)

def main():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Preprocess data
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = load_and_preprocess(config)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Train model
    model, train_losses, val_losses, val_output = train_model(X_train, y_train, X_val, y_val, config)

    # Evaluate
    y_val_true, y_val_pred, r2_nu, r2_dp, rmse_nu, rmse_dp = inverse_transform_and_evaluate(y_val, val_output, scaler_y)

    # Save plots
    plot_loss_curves(train_losses, val_losses, "plots/loss_curve_single.png")
    plot_predictions(y_val_true[:, 0], y_val_pred[:, 0], "Nu_avg", "plots/Nu_avg_prediction_single.png")
    plot_predictions(y_val_true[:, 1], y_val_pred[:, 1], "DeltaP", "plots/DeltaP_prediction_single.png")

    # Print summary
    print("\n📊 Final Metrics:")
    print(f"✅ Nu_avg | R²: {r2_nu:.4f}, RMSE: {rmse_nu:.4f}")
    print(f"✅ DeltaP | R²: {r2_dp:.4f}, RMSE: {rmse_dp:.4f}")

if __name__ == "__main__":
    main()
