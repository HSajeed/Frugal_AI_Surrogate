import torch
import torch.nn as nn
import torch.optim as optim
from models.model import build_mlp

def train_model(X_train, y_train, X_val, y_val, config):
    input_dim = X_train.shape[1]
    output_dim = config["model"]["output_dim"]
    hidden_layers = config["model"]["hidden_layers"]

    model = build_mlp(input_dim, output_dim, hidden_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    best_val_loss = float('inf')
    patience = config["training"]["patience"]
    wait = 0
    train_losses = []
    val_losses = []

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break

    return model, train_losses, val_losses, val_output
