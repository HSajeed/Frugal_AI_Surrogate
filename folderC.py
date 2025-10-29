import os

folders = [
    "data", "models", "scripts", "config",
    "results/metrics", "results/predictions", "results/logs", "plots"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Project folders created.")
