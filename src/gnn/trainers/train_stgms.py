#!/usr/bin/env python3
"""
train_stgms.py
--------------
STGMS model eğitim scripti

STGMS (Spatio-Temporal Graph Neural Network with Multi-timeScale) modelini
eğitmek için tasarlanmış training pipeline.

Özellikler:
- STGMSDataset kullanımı (multi-timescale decomposition)
- Decomposed features (F * (m+1)) ile eğitim
- Train/val/test split
- Chebyshev basis hesaplama
- Early stopping ve checkpoint kaydetme

Kullanım:
    python train_stgms.py --epochs 100 --batch_size 32 --periods 96 16 4
    python train_stgms.py --use_last_n_days 7 --lr 0.001

Kritik Notlar:
    1. Model in_channels = dataset.num_features_decomposed kullanmalı
    2. Target (y) orijinal feature boyutunda (F_original)
    3. Normalizasyon sadece train set üzerinden yapılır
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

from src.gnn.dataset_stgms import STGMSDataset
from src.gnn.models.stgms import STGMS, count_parameters


class STGMSTrainer:
    """STGMS model trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        patience: int = 10,
        checkpoint_dir: str = 'outputs/models/stgms'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }
    
    def train_epoch(self) -> tuple:
        """Bir epoch training"""
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # Data to device
            x = batch['x'].to(self.device)  # (B, T_in, N, F_decomposed)
            y = batch['y'].to(self.device)  # (B, T_out, N, F_original)
            # Graph is same for all samples, take first one
            edge_index = batch['edge_index'][0].to(self.device)  # (2, E)
            edge_attr = batch['edge_attr'][0].to(self.device).squeeze(-1)  # (E,)
            
            # Forward pass
            # STGMS output: (B, T_out, N, F_original)
            pred = self.model(x, edge_index, edge_attr)
            
            # Loss hesapla
            loss = self.criterion(pred, y)
            mae = torch.abs(pred - y).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            # Graph is same for all samples, take first one
            edge_index = batch['edge_index'][0].to(self.device)  # (2, E)
            edge_attr = batch['edge_attr'][0].to(self.device).squeeze(-1)  # (E,)
            
            pred = self.model(x, edge_index, edge_attr)
            
            loss = self.criterion(pred, y)
            mae = torch.abs(pred - y).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self, epochs: int):
        """Training loop"""
        print(f"\n{'='*70}")
        print(f"🚀 STGMS Training başlıyor...")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # History kaydet
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Log
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ New best model! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    print(f"\n⚠️  Early stopping! No improvement for {self.patience} epochs.")
                    break
            
            # Save checkpoint her 10 epoch
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print()
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ Training tamamlandı!")
        print(f"  - Total time: {total_time/60:.1f} dakika")
        print(f"  - Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Model checkpoint kaydet"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            filename = 'best_model_stgms.pt'
        else:
            filename = f'checkpoint_stgms_epoch_{epoch}.pt'
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"    💾 Saved: {filepath}")
    
    def save_history(self):
        """Training history kaydet (JSON)"""
        history_file = self.checkpoint_dir / 'training_history_stgms.json'
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"📊 Training history saved: {history_file}")


def main():
    """Main training script"""
    
    # Argument parser
    parser = argparse.ArgumentParser(description='STGMS Training')
    
    # Dataset args
    parser.add_argument('--window_size', type=int, default=12,
                        help='Input time window (default: 12)')
    parser.add_argument('--prediction_horizon', type=int, default=3,
                        help='Prediction horizon (default: 3)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Window stride (default: 1)')
    parser.add_argument('--periods', type=int, nargs='+', default=[96, 16, 4],
                        help='Decomposition periods (default: [96, 16, 4])')
    
    # Data filtering
    parser.add_argument('--start_time', type=str, default=None,
                        help='Start timestamp (ISO format)')
    parser.add_argument('--end_time', type=str, default=None,
                        help='End timestamp (ISO format)')
    parser.add_argument('--use_last_n_days', type=int, default=None,
                        help='Use only last N days of data')
    
    # Model args
    parser.add_argument('--k_order', type=int, default=3,
                        help='Chebyshev polynomial order (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--sigma', type=float, default=50.0,
                        help='Gaussian kernel sigma (meters, default: 50.0)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu, default: cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/models/stgms',
                        help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    
    # Checkpoint loading
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (.pt file)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune mode (reset optimizer)')
    
    args = parser.parse_args()
    
    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*70}")
    print(f"STGMS Training")
    print(f"{'='*70}\n")
    
    # Config yazdır
    print(f"📋 Configuration:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print()
    
    # 1. Dataset oluştur (STGMSDataset)
    print(f"📦 Loading STGMS dataset...")
    print(f"  🧩 Multi-timescale periods: {args.periods}")
    
    if args.use_last_n_days:
        print(f"  🕒 Sadece son {args.use_last_n_days} gün verisi kullanılacak")
    elif args.start_time or args.end_time:
        print(f"  📅 Tarih aralığı: {args.start_time or 'başlangıç'} → {args.end_time or 'şimdi'}")
    else:
        print(f"  📊 TÜM Neo4j verisi kullanılacak")
    
    dataset = STGMSDataset(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        stride=args.stride,
        periods=args.periods,
        start_time=args.start_time,
        end_time=args.end_time,
        use_last_n_days=args.use_last_n_days
    )
    
    print(f"\n  📊 Dataset Summary:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Nodes: {dataset.num_nodes}")
    print(f"  - Original features: {dataset.num_features}")
    print(f"  - Decomposed features: {dataset.num_features_decomposed}")
    print(f"  - Feature expansion: {dataset.num_features_decomposed / dataset.num_features:.1f}x")
    
    # 2. Train/val/test split
    print(f"\n📊 Splitting dataset...")
    train_size = int(args.train_ratio * len(dataset))
    val_size = int(args.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    # 2.5 CRITICAL: Normalization (ONLY on train set)
    print(f"\n📐 Computing normalization statistics from TRAIN set only...")
    
    # Collect all train samples (decomposed X)
    train_x_list = []
    for idx in train_dataset.indices:
        sample = dataset[idx]
        train_x_list.append(sample['x'])  # (window_size, N, F_decomposed)
    
    # Stack: (train_samples, window_size, N, F_decomposed)
    train_x_all = torch.stack(train_x_list, dim=0)
    
    # Reshape to (train_samples * window_size * N, F_decomposed)
    train_x_flat = train_x_all.reshape(-1, dataset.num_features_decomposed)
    
    # Compute mean and std
    feature_mean = train_x_flat.mean(dim=0, keepdim=True)  # (1, F_decomposed)
    feature_std = train_x_flat.std(dim=0, keepdim=True)    # (1, F_decomposed)
    
    # Prevent division by zero
    feature_std = torch.where(
        feature_std < 1e-6,
        torch.ones_like(feature_std),
        feature_std
    )
    
    print(f"  ✓ Feature mean shape: {feature_mean.shape}")
    print(f"  ✓ Feature std shape: {feature_std.shape}")
    
    # Normalize decomposed_x using train statistics
    dataset.decomposed_x = (dataset.decomposed_x - feature_mean) / feature_std
    print(f"  ✓ Decomposed features normalized (no data leakage)")
    
    # 3. DataLoader
    print(f"\n🔄 Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # 4. Edge weights hesapla (Gaussian Kernel)
    # Not: ChebConv kendi içinde Chebyshev polinomlarını hesaplar,
    # pre-computed basis'e ihtiyaç yok
    print(f"\n🔄 Preparing edge weights with Gaussian Kernel...")
    sample = dataset[0]
    edge_index = sample['edge_index']
    edge_dist = sample['edge_attr'].squeeze()  # (E,)
    
    # Gaussian Kernel: w_ij = exp(-d_ij^2 / sigma^2)
    sigma = args.sigma
    edge_weight = torch.exp(- (edge_dist ** 2) / (sigma ** 2))
    
    # Threshold small weights
    edge_weight[edge_weight < 0.1] = 0.0
    
    print(f"  ✓ Gaussian kernel applied (sigma={sigma}m)")
    print(f"    Edge weights: min={edge_weight.min():.4f}, max={edge_weight.max():.4f}, mean={edge_weight.mean():.4f}")
    print(f"  ℹ️  ChebConv will compute Chebyshev polynomials internally")
    
    # 5. Model oluştur
    print(f"\n🏗️  Building STGMS model...")
    
    # CRITICAL: in_channels = dataset.num_features_decomposed
    model = STGMS(
        num_nodes=dataset.num_nodes,
        in_channels=dataset.num_features_decomposed,  # Decomposed features!
        out_channels=dataset.num_features,             # Original features (target)
        window_size=args.window_size,
        horizon=args.prediction_horizon,
        k_order=args.k_order,
        dropout=args.dropout
    )
    
    print(f"  - Model: STGMS")
    print(f"  - Input channels: {dataset.num_features_decomposed} (decomposed)")
    print(f"  - Output channels: {dataset.num_features} (original)")
    print(f"  - Parameters: {count_parameters(model):,}")
    print(f"  - Device: {args.device}")
    
    # 5.5. Checkpoint yükle (eğer varsa)
    start_epoch = 1
    if args.resume:
        print(f"\n💾 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Model weights loaded (epoch {checkpoint.get('epoch', '?')})")
        
        if not args.fine_tune:
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"  ✓ Resume mode: epoch {start_epoch}'den devam edecek")
        else:
            print(f"  ✓ Fine-tune mode: optimizer sıfırlandı")
    
    # 6. Trainer oluştur
    print(f"\n🎯 Creating STGMS trainer...")
    trainer = STGMSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Checkpoint'ten optimizer/scheduler yükle
    if args.resume and not args.fine_tune:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.history = checkpoint.get('history', trainer.history)
            print(f"  ✓ Optimizer/Scheduler state loaded")
            print(f"  ✓ Best val loss: {trainer.best_val_loss:.4f}")
        except Exception as e:
            print(f"  ⚠️  Optimizer yüklenemedi: {e}")
    
    # 7. Training başlat
    try:
        trainer.train(epochs=args.epochs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        trainer.save_checkpoint(epoch=0, is_best=False)
        print("💾 Checkpoint saved.")
    
    # 8. Test evaluation (eğer test set varsa)
    if test_size > 0:
        print(f"\n{'='*70}")
        print(f"🧪 Test Set Evaluation")
        print(f"{'='*70}\n")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # En iyi modeli yükle
        best_model_path = Path(args.checkpoint_dir) / "best_model_stgms.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📦 Best model loaded (epoch {checkpoint.get('epoch', '?')})")
        
        # Test evaluation
        model.eval()
        test_loss = 0
        test_mae = 0
        test_rmse = 0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(args.device)
                y = batch['y'].to(args.device)
                edge_index = batch['edge_index'][0].to(args.device)
                edge_attr = batch['edge_attr'][0].to(args.device).squeeze(-1)
                
                pred = model(x, edge_index, edge_attr)
                
                loss = torch.nn.functional.mse_loss(pred, y)
                mae = torch.abs(pred - y).mean()
                rmse = torch.sqrt(((pred - y) ** 2).mean())
                
                test_loss += loss.item()
                test_mae += mae.item()
                test_rmse += rmse.item()
                num_batches += 1
                
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())
        
        # Compute final metrics
        test_loss /= num_batches
        test_mae /= num_batches
        test_rmse /= num_batches
        
        # R² score
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        ss_res = ((all_targets - all_preds) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        # MAPE
        mape = (torch.abs((all_targets - all_preds) / (all_targets + 1e-8)).mean() * 100).item()
        
        print(f"\n📊 Test Set Results:")
        print(f"  - MSE Loss: {test_loss:.6f}")
        print(f"  - MAE:      {test_mae:.6f}")
        print(f"  - RMSE:     {test_rmse:.6f}")
        print(f"  - MAPE:     {mape:.2f}%")
        print(f"  - R²:       {r2.item():.6f}")
        
        # Save test metrics
        test_metrics = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': mape,
            'test_r2': r2.item()
        }
        
        metrics_file = Path(args.checkpoint_dir) / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"\n💾 Test metrics saved: {metrics_file}")
        print(f"\n{'='*70}\n")
    
    print(f"\n✅ Training script tamamlandı!")
    print(f"📁 Model checkpoints: {args.checkpoint_dir}")
    print(f"\n💡 Detaylı evaluation için:")
    print(f"   python src/gnn/evaluators/evaluate_stgms.py --checkpoint {args.checkpoint_dir}/best_model_stgms.pt\n")


if __name__ == "__main__":
    main()
