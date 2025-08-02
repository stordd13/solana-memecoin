# training/data_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import json

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
        
        print(f"✅ Loaded {mode} dataset: {len(self)} sequences")
        print(f"   Shape: {self.inputs.shape}")
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Retourne une séquence et ses targets"""
        
        # Input sequence
        x = self.inputs[idx]  # [seq_len, features]
        
        # Targets
        target_prices = self.targets[idx]  # [forecast_steps]
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
        
        # Convert to tensors
        x = torch.FloatTensor(x)
        target_prices = torch.FloatTensor(target_prices)
        
        # Créer le dict de targets
        targets = {
            'prices': target_prices,
            # La direction sera calculée dans la loss
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