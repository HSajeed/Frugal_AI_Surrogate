import torch.nn as nn

class MLPDropout(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = getattr(nn, activation.capitalize())()
        layers = []

        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_layers[i-1], h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
