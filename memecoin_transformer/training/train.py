# train.py

import sys
from pathlib import Path

# Ajouter le dossier parent au path Python
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import argparse
import json

from model.transformer import create_memecoin_transformer
from model.losses import MemecoinsLoss
from model.data_loader import create_data_loaders
from training.trainer import MemecoinsTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Memecoin Transformer')
    parser.add_argument('--data-path', type=str, required=True, help='Path to sequences_scaled.npz')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size (512 optimized for M4 Max)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/mps/cuda/cpu)')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            print("🚀 Using Apple Silicon GPU (M4 Max optimized!)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("🚀 Using CUDA GPU")
        else:
            device = 'cpu'
            print("💻 Using CPU")
    else:
        device = args.device
    
    # Load metadata
    metadata_path = Path(args.data_path).parent / 'sequences_metadata_normal_tokens.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Dataset info:")
    print(f"   Features: {len(metadata['feature_names'])}")
    print(f"   Sequence length: {metadata['sequence_length']}")
    print(f"   Forecast steps: {metadata['forecast_steps']}")
    print(f"   Train sequences: {metadata['n_train_sequences']:,}")
    print(f"   Val sequences: {metadata['n_val_sequences']:,}")
    
    # Create model
    model = create_memecoin_transformer(
        input_dim=len(metadata['feature_names']),
        seq_len=metadata['sequence_length'],
        forecast_len=metadata['forecast_steps'],
        d_model=args.d_model,
        num_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    # Dans train.py, après le premier batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx == 0:
            print("\n🔍 First batch analysis:")
            print(f"Input shape: {inputs.shape}")
            print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            
            # Analyser CHAQUE feature individuellement
            print("\n📊 Feature-by-feature analysis:")
            if hasattr(train_loader.dataset, 'feature_names'):
                feature_names = train_loader.dataset.feature_names
            else:
                feature_names = [f'feat_{i}' for i in range(inputs.shape[2])]
                
            for i in range(inputs.shape[2]):
                feat_values = inputs[:, :, i]
                print(f"{i}. {feature_names[i]:20s}: min={feat_values.min():8.2f}, "
                    f"max={feat_values.max():8.2f}, "
                    f"mean={feat_values.mean():8.2f}, "
                    f"std={feat_values.std():8.2f}")
                
                # Identifier les valeurs extrêmes
                if feat_values.min() < -10 or feat_values.max() > 10:
                    print(f"   ⚠️  PROBLÈME DE SCALING DÉTECTÉ!")
            
            print(f"\nTarget shape: {targets['prices'].shape}")
            print(f"Target range: [{targets['prices'].min():.3f}, {targets['prices'].max():.3f}]")
            break
    # Loss function
    criterion = MemecoinsLoss(
        price_weight=1.0,
        direction_weight=0.5,
        consistency_weight=0.2,
        uncertainty_weight=0.1,
        use_uncertainty_weighting=False
    )
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW(
        model.get_param_groups(lr=args.lr),
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Trainer
    trainer = MemecoinsTrainer(
        model=model,
        device=device,
        use_mixed_precision=args.mixed_precision,
        gradient_clip=1.0
    )
    
    # Train!
    print(f"\n🚀 Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best model saved to: {args.save_dir}/best_model.pth")
    print(f"   Training history saved to: {args.save_dir}/training_history.json")

if __name__ == "__main__":
    main()