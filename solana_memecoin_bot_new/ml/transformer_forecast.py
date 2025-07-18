import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np

class PumpTransformer(nn.Module):
    """
    PyTorch Transformer for forecasting next 5-15min returns based on sequence (10-30min features).
    Input: [batch, seq_len, features] incl. volume/imbalance.
    Output: Binary pump prediction (>50% return) or regression.
    """
    def __init__(self, input_dim: int, seq_len: int = 10, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, 1)  # Regression for returns; sigmoid for binary
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))  # Last timestep prediction (binary pump)

def prepare_sequences(df: pl.DataFrame, seq_len: int = 10, features: list = None) -> tuple:
    """
    Preps sequences per token/archetype; prevents leakage with train/test splits.
    """
    if features is None:
        features = ["scaled_returns", "ma_returns_5", "vol_std_5", "imbalance", "avg_volume_5m", "liquidity"]
    sequences = []
    labels = []
    for token in df["token_id"].unique():
        token_df = df.filter(pl.col("token_id") == token).sort("timestamp")
        data = token_df.select(features).to_numpy()
        future_returns = token_df["returns"].shift(-1).to_numpy()  # Next return as label
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            labels.append(1 if future_returns[i+seq_len-1] > 0.5 else 0)  # Binary pump
    return np.array(sequences), np.array(labels)

def train_transformer_per_archetype(df: pl.DataFrame, epochs: int = 10, lr: float = 0.001):
    """
    Trains separate Transformers per archetype; uses volume features for early pump detection.
    Targets >40% F1; Bayesian opt stub for hypers.
    """
    models = {}
    for arch in df["archetype"].unique():
        subset = df.filter(pl.col("archetype") == arch)
        train_subset = subset.filter(pl.col("split") == "train")
        test_subset = subset.filter(pl.col("split") == "test")
        
        X_train, y_train = prepare_sequences(train_subset)
        X_test, y_test = prepare_sequences(test_subset)
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        input_dim = X_train.shape[2]
        model = PumpTransformer(input_dim)
        criterion = nn.BCELoss()  # Binary cross-entropy
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            print(f"Archetype {arch}, Epoch {epoch}: Loss {loss.item():.4f}")
        
        # Eval
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().round().numpy()
            f1 = f1_score(y_test, preds)
            print(f"Archetype {arch} Test F1: {f1:.2%} (target >40%)")
        
        models[arch] = model
    return models

# Usage
df = pl.read_parquet("data/processed/processed_features.parquet")
transformer_models = train_transformer_per_archetype(df)