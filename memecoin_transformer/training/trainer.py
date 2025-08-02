# training/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

class MemecoinsTrainer:
    """Trainer optimisé pour M4 Max avec mixed precision et monitoring"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'mps',
        use_mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 50
    ):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device in ['cuda', 'mps']
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        
        # Mixed precision setup
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(
        self,
        train_loader,
        criterion,
        optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Entraîne une époque"""
        
        self.model.train()
        epoch_metrics = {
            'loss_total': 0,
            'loss_price': 0,
            'loss_direction': 0,
            'loss_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast(device_type=self.device):
                    predictions = self.model(inputs)
                    loss, metrics = criterion(predictions, targets)
            else:
                predictions = self.model(inputs)
                loss, metrics = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()
            
            # Update metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss_total']:.4f}",
                    'price': f"{metrics['loss_price']:.4f}",
                    'dir': f"{metrics['loss_direction']:.4f}"
                })
        
        # Average metrics
        n_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def validate(
        self,
        val_loader,
        criterion
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Évalue sur le validation set"""
        
        self.model.eval()
        val_metrics = {
            'loss_total': 0,
            'loss_price': 0,
            'loss_direction': 0,
            'loss_consistency': 0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward
                predictions = self.model(inputs)
                loss, metrics = criterion(predictions, targets)
                
                # Update metrics
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key] += metrics[key]
                
                # Store predictions for analysis
                all_predictions.append({
                    k: v.cpu().numpy() for k, v in predictions.items()
                    if k != 'attention_weights'
                })
                all_targets.append({
                    k: v.cpu().numpy() for k, v in targets.items()
                })
        
        # Average metrics
        n_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= n_batches
        
        # Concatenate predictions
        predictions_concat = {}
        targets_concat = {}
        
        for key in all_predictions[0].keys():
            predictions_concat[key] = np.concatenate([p[key] for p in all_predictions])
        
        for key in all_targets[0].keys():
            targets_concat[key] = np.concatenate([t[key] for t in all_targets])
        
        # Calculer des métriques supplémentaires
        val_metrics.update(self._compute_extra_metrics(predictions_concat, targets_concat))
        
        return val_metrics, predictions_concat
    
    def _compute_extra_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calcule des métriques supplémentaires"""
        
        metrics = {}
        
        # MAE sur les prix
        price_mae = np.mean(np.abs(predictions['prices'] - targets['prices']))
        metrics['price_mae'] = float(price_mae)
        
        # Accuracy sur la direction
        if 'directions' in predictions:
            # Créer les vrais labels de direction
            if len(targets['prices'].shape) == 2:
                # Multi-step
                true_directions = (targets['prices'] > 0).astype(float)
            else:
                true_directions = (targets['prices'] > 0).astype(float)
            
            pred_directions = (predictions['directions'] > 0.5).astype(float)
            direction_acc = np.mean(pred_directions == true_directions)
            metrics['direction_accuracy'] = float(direction_acc)
        
        # R² score pour les prix
        from sklearn.metrics import r2_score
        r2 = r2_score(
            targets['prices'].flatten(),
            predictions['prices'].flatten()
        )
        metrics['price_r2'] = float(r2)
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs: int,
        save_dir: str = './checkpoints'
    ):
        """Boucle d'entraînement complète"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 Starting training on {self.device}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Gradient clipping: {self.gradient_clip}")
        print(f"   Epochs: {epochs}")
        
        for epoch in range(1, epochs + 1):
            # Train
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics, predictions = self.validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss_total'])
                else:
                    scheduler.step()
            
            # Track history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Save best model
            if val_metrics['loss_total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss_total']
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': self.best_val_loss,
                    'val_metrics': val_metrics,
                    'config': {
                        'model_config': self.model.__dict__,
                        'device': self.device,
                        'mixed_precision': self.use_mixed_precision
                    }
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
            
            # Log
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch}/{epochs} ({elapsed:.1f}s)")
            print(f"   Train Loss: {train_metrics['loss_total']:.4f}")
            print(f"   Val Loss: {val_metrics['loss_total']:.4f} (best: {self.best_val_loss:.4f})")
            print(f"   Price MAE: {val_metrics.get('price_mae', 0):.4f}")
            print(f"   Direction Acc: {val_metrics.get('direction_accuracy', 0):.2%}")
            print(f"   Price R²: {val_metrics.get('price_r2', 0):.3f}")
            
            # Early stopping
            patience = 10
            if epoch > patience:
                recent_losses = [h['loss_total'] for h in self.val_history[-patience:]]
                if all(loss >= self.best_val_loss for loss in recent_losses):
                    print(f"\n⚠️  Early stopping at epoch {epoch}")
                    break
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_loss': float(self.best_val_loss),
            'final_epoch': epoch
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Training complete! Best val loss: {self.best_val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return history