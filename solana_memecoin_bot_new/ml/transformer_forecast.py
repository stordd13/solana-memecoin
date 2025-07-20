# ml/transformer_forecast.py (Walk-forward training per token/archetype; PyTorch for seq forecasting)
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
import time

from utils import setup_logger
logger = setup_logger(__name__)

class PumpTransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 10, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, 1)  # Binary pump
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))

def prepare_sequences(df: pl.DataFrame, seq_len: int = 10, features: list = None) -> tuple:
    if features is None:
        features = ["scaled_returns", "ma_5", "vol_std_5", "imbalance", "avg_volume_5m", "liquidity"]
    sequences = []
    labels = []
    for token in df["token_id"].unique():
        token_df = df.filter(pl.col("token_id") == token).sort("datetime")
        data = token_df.select(features).to_numpy()
        future_returns = token_df["returns"].shift(-1).to_numpy()
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            labels.append(1 if future_returns[i+seq_len-1] > 0.5 else 0)
    return np.array(sequences), np.array(labels)

def train_transformer_per_archetype(df: pl.DataFrame, epochs: int = 10, lr: float = 0.001, walk_steps: int = 5) -> dict:
    models = {}
    for arch in df["archetype"].unique():
        subset = df.filter(pl.col("archetype") == arch)
        arch_metrics = []
        for step in range(walk_steps):
            step_start = time.time()
            # Walk-forward: Rolling split per step (e.g., train 0-70%, val 70-85%, test 85-100%; shift by 20%)
            shift = int(len(subset) * 0.2 * step)
            train_end = shift + int(len(subset) * 0.7)
            val_end = train_end + int(len(subset) * 0.15)
            train_df = subset.slice(shift, train_end - shift)
            val_df = subset.slice(train_end, val_end - train_end)
            test_df = subset.slice(val_end, len(subset) - val_end)
            
            X_train, y_train = prepare_sequences(train_df)
            X_val, y_val = prepare_sequences(val_df)
            X_test, y_test = prepare_sequences(test_df)
            
            if len(X_train) == 0: continue
            
            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            
            input_dim = X_train.shape[2]
            model = PumpTransformer(input_dim)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Eval on test
            with torch.no_grad():
                preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().round().numpy()
                f1 = f1_score(y_test, preds)
                arch_metrics.append(f1)
                logger.info(f"Archetype {arch}, Walk Step {step}: Test F1 {f1:.2%}, Time {time.time() - step_start:.2f}s")
        
        models[arch] = model  # Last model, or ensemble
        avg_f1 = np.mean(arch_metrics)
        logger.info(f"Archetype {arch} Avg Walk-Forward F1: {avg_f1:.2%}")
    return models

# Usage
df = pl.read_parquet("processed_features_5m.parquet")
transformer_models = train_transformer_per_archetype(df)