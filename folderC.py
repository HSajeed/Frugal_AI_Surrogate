import os

folders = [
    "data", "models", "scripts", "config",
    "results", "plots"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created.")
