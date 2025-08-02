# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class MemecoinsLoss(nn.Module):
    """
    Loss multi-tâches pour la prédiction de memecoins
    Combine price prediction, direction, et consistency
    """
    
    def __init__(
        self,
        price_weight: float = 1.0,
        direction_weight: float = 0.5,
        consistency_weight: float = 0.2,
        uncertainty_weight: float = 0.1,
        use_uncertainty_weighting: bool = True
    ):
        super().__init__()
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.consistency_weight = consistency_weight
        self.uncertainty_weight = uncertainty_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Smooth L1 pour la robustesse aux outliers
        self.price_criterion = nn.SmoothL1Loss(reduction='none')
        self.direction_criterion = nn.BCELoss(reduction='none')
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule la loss totale et les composantes individuelles
        
        Args:
            predictions: Dict avec 'prices', 'directions', 'uncertainty', 'volatility'
            targets: Dict avec 'prices', 'directions'
        """
        
        batch_size = predictions['prices'].shape[0]
        device = predictions['prices'].device
        
        # 1. Price prediction loss (Smooth L1 / Huber)
        price_loss = self.price_criterion(
            predictions['prices'], 
            targets['prices']
        )
        
        # Pondération par l'incertitude si activée
        if self.use_uncertainty_weighting and 'uncertainty' in predictions:
            # Utilise l'incertitude prédite pour pondérer la loss
            uncertainty = predictions['uncertainty']
            price_loss = price_loss / (2 * uncertainty.pow(2)) + uncertainty.log()
        
        price_loss = price_loss.mean()
        
        # 2. Direction prediction loss
        # Créer les labels de direction à partir des prix futurs
        current_price = targets.get('current_price', None)
        if current_price is not None:
            # Direction = 1 si prix monte, 0 sinon
            direction_targets = (targets['prices'] > current_price.unsqueeze(-1)).float()
        else:
            # Utiliser la différence entre timesteps consécutifs
            price_diff = targets['prices'][:, 1:] - targets['prices'][:, :-1]
            first_direction = (targets['prices'][:, 0] > 0).float().unsqueeze(1)
            direction_targets = torch.cat([
                first_direction,
                (price_diff > 0).float()
            ], dim=1)
        
        direction_loss = self.direction_criterion(
            predictions['directions'], 
            direction_targets
        ).mean()
        
        # 3. Consistency loss
        # Les prédictions de prix et direction doivent être cohérentes
        # Si on prédit une hausse, le prix prédit devrait augmenter
        predicted_price_direction = torch.sigmoid(
            predictions['prices'] * 10  # Scale pour avoir des probas nettes
        )
        consistency_loss = F.mse_loss(
            predictions['directions'],
            predicted_price_direction
        )
        
        # 4. Uncertainty regularization
        # Éviter l'overconfidence : l'incertitude ne doit pas être trop faible
        uncertainty_reg = 0.0
        if 'uncertainty' in predictions:
            # Pénaliser les incertitudes trop faibles
            min_uncertainty = 0.01
            uncertainty_reg = F.relu(min_uncertainty - predictions['uncertainty'].mean())
        
        # 5. Volatility prediction loss (optionnel)
        volatility_loss = 0.0
        if 'volatility' in predictions and 'volatility' in targets:
            volatility_loss = F.mse_loss(
                predictions['volatility'],
                targets['volatility']
            )
        
        # Calcul de la loss totale
        total_loss = (
            self.price_weight * price_loss +
            self.direction_weight * direction_loss +
            self.consistency_weight * consistency_loss +
            self.uncertainty_weight * uncertainty_reg +
            0.1 * volatility_loss  # Poids fixe pour la volatilité
        )
        
        # Métriques pour le logging
        metrics = {
            'loss_total': total_loss.item(),
            'loss_price': price_loss.item(),
            'loss_direction': direction_loss.item(),
            'loss_consistency': consistency_loss.item(),
            'loss_uncertainty': uncertainty_reg.item() if isinstance(uncertainty_reg, torch.Tensor) else uncertainty_reg,
            'loss_volatility': volatility_loss.item() if isinstance(volatility_loss, torch.Tensor) else volatility_loss
        }
        
        return total_loss, metrics


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre dans la classification de direction
    Utile si on a beaucoup plus de 'non-pumps' que de 'pumps'
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Prédictions (après sigmoid)
            targets: Labels binaires
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if t=1 else 1-p
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()