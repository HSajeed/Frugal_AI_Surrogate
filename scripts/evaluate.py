import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os

def inverse_transform_and_evaluate(y_val, val_output, scaler_y):
    y_val_pred = scaler_y.inverse_transform(val_output.numpy())
    y_val_true = scaler_y.inverse_transform(y_val.numpy())

    r2_nu = r2_score(y_val_true[:, 0], y_val_pred[:, 0])
    r2_dp = r2_score(y_val_true[:, 1], y_val_pred[:, 1])
    rmse_nu = np.sqrt(mean_squared_error(y_val_true[:, 0], y_val_pred[:, 0]))
    rmse_dp = np.sqrt(mean_squared_error(y_val_true[:, 1], y_val_pred[:, 1]))

    print(f"\n✅ Nu_avg | R²: {r2_nu:.4f}, RMSE: {rmse_nu:.4f}")
    print(f"✅ DeltaP | R²: {r2_dp:.4f}, RMSE: {rmse_dp:.4f}")

    return y_val_true, y_val_pred, r2_nu, r2_dp, rmse_nu, rmse_dp

def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, label, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, c='blue', label='Predicted')
    plt.plot(y_true, y_true, 'r--', label='Ideal')
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"{label} Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
