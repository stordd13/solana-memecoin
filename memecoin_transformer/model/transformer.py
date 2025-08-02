# models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding adapté aux séries temporelles financières"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        # Encodage sinusoidal classique
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable scaling factor pour l'importance du positional encoding
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, seq_len, d_model]"""
        return x + self.scale * self.pe[:, :x.size(1)]


class FeatureEmbedding(nn.Module):
    """Embedding layer spécialisé pour nos features de trading"""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Projection non-linéaire pour capturer les interactions
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Feature-wise scaling learnable
        self.feature_scales = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, seq_len, input_dim]"""
        # Scale features avant projection
        x = x * self.feature_scales.unsqueeze(0).unsqueeze(0)
        return self.projection(x)


class TemporalAttentionBlock(nn.Module):
    """Block d'attention spécialisé pour les patterns temporels"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention avec bias temporel
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Temporal convolution pour capturer les patterns locaux
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, 
            kernel_size=3, 
            padding=1, 
            groups=d_model  # Depthwise
        )
        
        # Feed-forward amélioré
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [batch_size, seq_len, d_model]"""
        
        # 1. Self-attention avec residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        
        # 2. Temporal convolution
        # [batch, seq, d_model] -> [batch, d_model, seq]
        conv_in = x.transpose(1, 2)
        conv_out = self.temporal_conv(conv_in).transpose(1, 2)
        x = self.ln2(x + self.dropout(conv_out))
        
        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln3(x + self.dropout(ffn_out))
        
        return x


class MemecoinsTransformer(nn.Module):
    """Transformer optimisé pour la prédiction de memecoins"""
    
    def __init__(
        self,
        input_dim: int = 10,
        seq_len: int = 15,
        forecast_len: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.d_model = d_model
        
        # 1. Feature embedding
        self.feature_embedding = FeatureEmbedding(input_dim, d_model, dropout)
        
        # 2. Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Aggregation mechanism
        self.aggregation_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # 5. Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # last + mean + attended
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 6. Prediction heads
        self._build_prediction_heads(d_model * 2, forecast_len, dropout)
        
        # 7. Initialize weights
        self._init_weights()
        
    def _build_prediction_heads(self, hidden_dim: int, forecast_len: int, dropout: float):
        """Construit les têtes de prédiction multi-tâches"""
        
        # Price prediction head (régression)
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_len)
        )
        
        # Direction prediction head (classification binaire)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, forecast_len)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, forecast_len)
        )
        
        # Volatility prediction head (bonus)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, forecast_len)
        )
        
    def _init_weights(self):
        """Initialisation des poids Xavier/He"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Crée un masque causal pour l'attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            Dict avec keys: 'prices', 'directions', 'uncertainty', 'volatility'
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. Feature embedding
        x = self.feature_embedding(x)  # [batch, seq, d_model]
        
        # 2. Add positional encoding
        x = self.pos_encoding(x)
        
        # 3. Pass through transformer blocks
        # Pas de masque causal car on veut voir tout l'historique
        for block in self.transformer_blocks:
            x = block(x)
        
        # 4. Sophisticated aggregation
        # a) Last token (contient le plus de contexte)
        last_token = x[:, -1, :]  # [batch, d_model]
        
        # b) Mean pooling (vision globale)
        mean_pooled = x.mean(dim=1)  # [batch, d_model]
        
        # c) Attention-weighted aggregation
        query = last_token.unsqueeze(1)  # [batch, 1, d_model]
        attended, attention_weights = self.aggregation_attention(query, x, x)
        attended = attended.squeeze(1)  # [batch, d_model]
        
        # 5. Combine all representations
        combined = torch.cat([last_token, mean_pooled, attended], dim=-1)
        combined = self.context_fusion(combined)  # [batch, d_model*2]
        
        # 6. Multi-task predictions
        price_pred = self.price_head(combined)  # [batch, forecast_len]
        direction_logits = self.direction_head(combined)  # [batch, forecast_len]
        uncertainty = self.uncertainty_head(combined)  # [batch, forecast_len]
        volatility = self.volatility_head(combined)  # [batch, forecast_len]
        
        # 7. Activations finales
        direction_probs = torch.sigmoid(direction_logits)
        uncertainty = F.softplus(uncertainty) + 1e-6  # Toujours positif
        volatility = F.softplus(volatility) + 1e-6
        
        return {
            'prices': price_pred,              # Log price predictions
            'directions': direction_probs,     # P(price up)
            'uncertainty': uncertainty,        # Epistemic uncertainty
            'volatility': volatility,         # Predicted volatility
            'attention_weights': attention_weights.detach()  # Pour visualisation
        }
    
    def get_param_groups(self, lr: float = 1e-3) -> list:
        """Retourne les groupes de paramètres avec learning rates différents"""
        # Learning rate différent pour les embeddings
        embed_params = list(self.feature_embedding.parameters()) + \
                      list(self.pos_encoding.parameters())
        
        # Transformer blocks
        transformer_params = []
        for block in self.transformer_blocks:
            transformer_params.extend(block.parameters())
        
        # Heads
        head_params = list(self.price_head.parameters()) + \
                     list(self.direction_head.parameters()) + \
                     list(self.uncertainty_head.parameters()) + \
                     list(self.volatility_head.parameters())
        
        return [
            {'params': embed_params, 'lr': lr * 0.1},  # Lower LR for embeddings
            {'params': transformer_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 2}  # Higher LR for heads
        ]


# Fonction helper pour créer le modèle
def create_memecoin_transformer(
    input_dim: int = 10,
    seq_len: int = 15,
    forecast_len: int = 5,
    d_model: int = 256,
    **kwargs
) -> MemecoinsTransformer:
    """Factory function pour créer le modèle avec config par défaut"""
    
    default_config = {
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    config = {**default_config, **kwargs}
    
    model = MemecoinsTransformer(
        input_dim=input_dim,
        seq_len=seq_len,
        forecast_len=forecast_len,
        d_model=d_model,
        **config
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Created MemecoinsTransformer:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_memecoin_transformer()
    
    # Test input
    batch_size = 32
    x = torch.randn(batch_size, 15, 10)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\n📊 Output shapes:")
    for key, value in outputs.items():
        if key != 'attention_weights':
            print(f"   - {key}: {value.shape}")