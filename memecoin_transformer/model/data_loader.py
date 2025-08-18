# model/data_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class MemecoinsDataset(Dataset):
    """Dataset pour les séquences de memecoins"""
    
    def __init__(
        self,
        sequences_path: str,
        mode: str = 'train',
        transform: Optional[callable] = None
    ):
        """
        Args:
            sequences_path: Chemin vers le fichier .npz
            mode: 'train' ou 'val'
            transform: Transformations additionnelles
        """
        self.mode = mode
        self.transform = transform
        
        # Charger les données
        data = np.load(sequences_path)
        
        if mode == 'train':
            self.inputs = data['train_inputs']
            self.targets = data['train_targets']
        else:
            self.inputs = data['val_inputs']
            self.targets = data['val_targets']
        
        # Charger les métadonnées si disponibles
        self.feature_names = data.get('feature_names', None)
        self.sequence_length = int(data.get('sequence_length', 15))
        self.forecast_steps = int(data.get('forecast_steps', 5))
        
        # Generate risk scores based on volatility patterns in the input sequences
        # High volatility in recent timesteps = higher risk
        self.risk_scores = self._generate_risk_scores()
        
        print(f"✅ Loaded {mode} dataset: {len(self)} sequences")
        print(f"   Input shape: {self.inputs.shape}")
        print(f"   Target shape: {self.targets.shape}")

        self.debug_features()
    
    def _generate_risk_scores(self) -> np.ndarray:
        """Generate risk scores based on volatility and price patterns"""
        # Simple heuristic: use volatility from last 5 timesteps
        # Assuming volatility is feature index 5 (you can adjust based on actual feature order)
        risk_scores = []
        
        for i in range(len(self.inputs)):
            sequence = self.inputs[i]
            # Calculate risk based on recent volatility and price changes
            # This is a simple heuristic - you can make it more sophisticated
            
            # If we have volatility feature (usually around index 5)
            if sequence.shape[1] > 5:
                recent_volatility = np.mean(sequence[-5:, 5])  # Last 5 timesteps, volatility feature
                # Normalize to 0-1 range
                risk = np.clip(recent_volatility * 2, 0, 1)  # Scale and clip
            else:
                # Default low risk if we can't calculate
                risk = 0.1
            
            risk_scores.append(risk)
        
        return np.array(risk_scores, dtype=np.float32).reshape(-1, 1)
    
    def debug_features(self):
        """Analyse les features pour détecter les anomalies"""
        print(f"\n🔍 Feature analysis for {self.mode} dataset:")
        
        if self.feature_names is not None:
            for i, feat_name in enumerate(self.feature_names):
                feat_values = self.inputs[:, :, i].flatten()
                valid_values = feat_values[~np.isnan(feat_values)]
                
                if len(valid_values) > 0:
                    print(f"{i}. {feat_name:20s} : min={valid_values.min():8.2f}, "
                          f"max={valid_values.max():8.2f}, "
                          f"mean={valid_values.mean():8.2f}, "
                          f"std={valid_values.std():8.2f}")
                    
                    # Alerte si valeurs extrêmes
                    if valid_values.min() < -100 or valid_values.max() > 100:
                        print(f"   ⚠️  VALEURS EXTRÊMES DÉTECTÉES !")

    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Retourne une séquence et ses targets"""
        
        # Input sequence
        x = self.inputs[idx]  # [seq_len, features]
        
        # Target prices (the actual target from the file)
        target_prices = self.targets[idx]  # [forecast_steps]
        
        # Risk score for this sequence
        target_risk = self.risk_scores[idx]  # [1]
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
        
        # Convert to tensors
        x = torch.FloatTensor(x)
        target_prices = torch.FloatTensor(target_prices)
        target_risk = torch.FloatTensor(target_risk)
        
        # Créer le dict de targets
        targets = {
            'prices': target_prices,
            'risk': target_risk  # Risk score based on volatility
            # Direction will be computed in the loss function from prices
        }
        
        return x, targets


def create_data_loaders(
    sequences_path: str,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = False,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders optimisés pour M4 Max
    """
    
    # Datasets
    train_dataset = MemecoinsDataset(sequences_path, mode='train')
    val_dataset = MemecoinsDataset(sequences_path, mode='val')
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Plus grand pour validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader
