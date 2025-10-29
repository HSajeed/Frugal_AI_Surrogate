import torch
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from scripts.preprocess import load_and_preprocess
from scripts.train import train_model
from scripts.evaluate import (
    inverse_transform_and_evaluate,
    plot_loss_curves,
    plot_predictions
)

def run_crossval(config, n_splits=5, run_folder="results/run_default"):
    # Load and preprocess
    X_train, _, y_train, _, scaler_X, scaler_y = load_and_preprocess(config)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config["data"]["random_seed"])
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n🔁 Fold {fold+1}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model, train_losses, val_losses, val_output = train_model(X_tr, y_tr, X_val, y_val, config)

        y_val_true, y_val_pred, r2_nu, r2_dp, rmse_nu, rmse_dp = inverse_transform_and_evaluate(
            y_val, val_output, scaler_y
        )

        fold_metrics.append({
            "fold": fold+1,
            "r2_nu": r2_nu,
            "r2_dp": r2_dp,
            "rmse_nu": rmse_nu,
            "rmse_dp": rmse_dp
        })

        # Save plots
        plot_loss_curves(train_losses, val_losses, f"{run_folder}/fold{fold+1}_loss_curve.png")
        plot_predictions(y_val_true[:, 0], y_val_pred[:, 0], "Nu_avg", f"{run_folder}/fold{fold+1}_Nu_avg_prediction.png")
        plot_predictions(y_val_true[:, 1], y_val_pred[:, 1], "DeltaP", f"{run_folder}/fold{fold+1}_DeltaP_prediction.png")

        # Save predictions to CSV
        df_pred = pd.DataFrame({
            "True_Nu_avg": y_val_true[:, 0],
            "Pred_Nu_avg": y_val_pred[:, 0],
            "True_DeltaP": y_val_true[:, 1],
            "Pred_DeltaP": y_val_pred[:, 1]
        })
        df_pred.to_csv(f"{run_folder}/fold{fold+1}_predictions.csv", index=False)

    # Save metrics to JSON
    metrics_path = f"{run_folder}/fold_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"\n Metrics saved to: {metrics_path}")

    return fold_metrics
