#!/usr/bin/env python3
"""
evaluate.py
-----------
STA-GCN model değerlendirme ve test scripti

Fonksiyonlar:
- Model test (loss, MAE, RMSE, MAPE)
- Visualizasyon (prediction vs actual)
- Segment-level analiz
- Time-series prediction grafikleri
- Baseline model karşılaştırması

Kullanım:
    # Best model'i test et
    python src/gnn/evaluate.py
    
    # Belirli checkpoint'i test et
    python src/gnn/evaluate.py --checkpoint outputs/models/best_model.pt
    
    # Baseline modellerle karşılaştır
    python src/gnn/evaluate.py --compare-baselines
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import json

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_sta import TrafficDataset
from models.sta_gcn import STAGCN
from graph_utils import precompute_cheb_basis


class ModelEvaluator:
    """STA-GCN model değerlendirici"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        cheb_basis: list,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.cheb_basis = [cb.to(device) for cb in cheb_basis]
        self.device = device
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Test set üzerinde değerlendirme"""
        
        print("\n" + "="*70)
        print("🧪 Model Evaluation")
        print("="*70 + "\n")
        
        all_predictions = []
        all_targets = []
        all_errors = []
        
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        num_batches = 0
        
        criterion = nn.MSELoss()
        
        for batch_idx, batch in enumerate(self.test_loader):
            x = batch['x'].to(self.device)  # (B, T_in, N, F)
            y = batch['y'].to(self.device)  # (B, T_out, N, F)
            
            # Prediction
            pred = self.model(x, self.cheb_basis)
            
            # Son T_out adımı
            if pred.shape[1] != y.shape[1]:
                pred = pred[:, -y.shape[1]:, :, :]
            
            # Metrics
            loss = criterion(pred, y)
            mae = torch.abs(pred - y).mean()
            rmse = torch.sqrt(((pred - y) ** 2).mean())
            
            # MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            epsilon = 1e-8
            mape = (torch.abs((y - pred) / (y + epsilon)) * 100).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_mape += mape.item()
            num_batches += 1
            
            # Kaydet (visualization için)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_errors.append((pred - y).abs().cpu().numpy())
        
        # Ortalamaları hesapla
        metrics = {
            'test_loss': total_loss / num_batches,
            'test_mae': total_mae / num_batches,
            'test_rmse': total_rmse / num_batches,
            'test_mape': total_mape / num_batches,
            'num_samples': len(self.test_loader.dataset),
            'num_batches': num_batches
        }
        
        # Numpy arrays
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        errors = np.concatenate(all_errors, axis=0)
        
        print("📊 Test Metrikleri:")
        print(f"  - Loss (MSE): {metrics['test_loss']:.4f}")
        print(f"  - MAE: {metrics['test_mae']:.4f}")
        print(f"  - RMSE: {metrics['test_rmse']:.4f}")
        print(f"  - MAPE: {metrics['test_mape']:.2f}%")
        print(f"  - Samples: {metrics['num_samples']}")
        
        return metrics, predictions, targets, errors
    
    def visualize_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str = "outputs/evaluation"
    ):
        """Prediction vs actual visualizasyonu"""
        
        print("\n📈 Visualization oluşturuluyor...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # predictions: (B, T_out, N, F)
        B, T_out, N, F = predictions.shape
        
        # 1. Scatter plot: Predicted vs Actual (ilk feature - speed_norm)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        pred_flat = predictions[:, :, :, 0].flatten()
        target_flat = targets[:, :, :, 0].flatten()
        
        # Perfect prediction line
        axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        axes[0].scatter(target_flat, pred_flat, alpha=0.3, s=1)
        axes[0].set_xlabel('Actual Speed (normalized)')
        axes[0].set_ylabel('Predicted Speed (normalized)')
        axes[0].set_title('Prediction vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = pred_flat - target_flat
        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_scatter.png', dpi=150)
        print(f"  ✓ Saved: {save_dir / 'prediction_scatter.png'}")
        plt.close()
        
        # 2. Time-series plot (rastgele bir segment seç)
        sample_idx = 0
        segment_idx = np.random.randint(0, N)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        time_steps = range(T_out)
        pred_series = predictions[sample_idx, :, segment_idx, 0]
        target_series = targets[sample_idx, :, segment_idx, 0]
        
        ax.plot(time_steps, target_series, 'o-', label='Actual', linewidth=2)
        ax.plot(time_steps, pred_series, 's--', label='Predicted', linewidth=2)
        ax.fill_between(time_steps, target_series, pred_series, alpha=0.3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Speed (normalized)')
        ax.set_title(f'Time-series Prediction (Sample {sample_idx}, Segment {segment_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'timeseries_sample.png', dpi=150)
        print(f"  ✓ Saved: {save_dir / 'timeseries_sample.png'}")
        plt.close()
        
        # 3. Feature-wise MAE
        feature_names = ['speed_norm', 'jf_norm', 'conf', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend']
        feature_mae = []
        
        for f_idx in range(F):
            mae = np.abs(predictions[:, :, :, f_idx] - targets[:, :, :, f_idx]).mean()
            feature_mae.append(mae)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(F), feature_mae)
        ax.set_xticks(range(F))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('MAE')
        ax.set_title('MAE per Feature')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'feature_mae.png', dpi=150)
        print(f"  ✓ Saved: {save_dir / 'feature_mae.png'}")
        plt.close()
        
        print(f"\n✅ Tüm grafikler kaydedildi: {save_dir}")


class BaselineModels:
    """Baseline model karşılaştırması için"""
    
    @staticmethod
    def historical_average(test_loader: DataLoader) -> Dict:
        """Historical Average baseline"""
        print("\n🔵 Baseline: Historical Average")
        
        total_mae = 0
        total_rmse = 0
        num_batches = 0
        
        for batch in test_loader:
            x = batch['x']  # (B, T_in, N, F)
            y = batch['y']  # (B, T_out, N, F)
            
            # Son 3 timestep'in ortalaması
            pred = x.mean(dim=1, keepdim=True).repeat(1, y.shape[1], 1, 1)
            
            mae = torch.abs(pred - y).mean()
            rmse = torch.sqrt(((pred - y) ** 2).mean())
            
            total_mae += mae.item()
            total_rmse += rmse.item()
            num_batches += 1
        
        metrics = {
            'model': 'Historical Average',
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches
        }
        
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    @staticmethod
    def last_value(test_loader: DataLoader) -> Dict:
        """Last Value (persistence) baseline"""
        print("\n🔵 Baseline: Last Value (Persistence)")
        
        total_mae = 0
        total_rmse = 0
        num_batches = 0
        
        for batch in test_loader:
            x = batch['x']  # (B, T_in, N, F)
            y = batch['y']  # (B, T_out, N, F)
            
            # Son timestep'i tekrarla
            pred = x[:, -1:, :, :].repeat(1, y.shape[1], 1, 1)
            
            mae = torch.abs(pred - y).mean()
            rmse = torch.sqrt(((pred - y) ** 2).mean())
            
            total_mae += mae.item()
            total_rmse += rmse.item()
            num_batches += 1
        
        metrics = {
            'model': 'Last Value',
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches
        }
        
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    @staticmethod
    def linear_regression(test_loader: DataLoader) -> Dict:
        """Simple Linear Regression baseline"""
        print("\n🔵 Baseline: Linear Regression")
        
        from sklearn.linear_model import LinearRegression
        
        # Train linear model
        all_x = []
        all_y = []
        
        for batch in test_loader:
            x = batch['x']  # (B, T_in, N, F)
            y = batch['y']  # (B, T_out, N, F)
            
            # Flatten to (B*N, T_in*F) and (B*N, T_out*F)
            B, T_in, N, F = x.shape
            T_out = y.shape[1]
            
            x_flat = x.permute(0, 2, 1, 3).reshape(-1, T_in * F)  # (B*N, T_in*F)
            y_flat = y.permute(0, 2, 1, 3).reshape(-1, T_out * F)  # (B*N, T_out*F)
            
            all_x.append(x_flat.numpy())
            all_y.append(y_flat.numpy())
        
        X_train = np.concatenate(all_x, axis=0)
        Y_train = np.concatenate(all_y, axis=0)
        
        # Fit
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        
        # Predict
        Y_pred = lr.predict(X_train)
        
        mae = np.abs(Y_pred - Y_train).mean()
        rmse = np.sqrt(((Y_pred - Y_train) ** 2).mean())
        
        metrics = {
            'model': 'Linear Regression',
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='STA-GCN Model Evaluation')
    
    # Model
    parser.add_argument('--checkpoint', type=str, default='outputs/models/best_model.pt',
                        help='Model checkpoint path')
    
    # Dataset
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--prediction_horizon', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=50.0,
                        help='Gaussian kernel sigma (meters, default: 50.0)')
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--compare_baselines', action='store_true',
                        help='Compare with baseline models')
    parser.add_argument('--save_dir', type=str, default='outputs/evaluation',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 STA-GCN Model Evaluation")
    print("="*70 + "\n")
    
    # Checkpoint kontrol
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        print("\n💡 Önce model eğitin:")
        print("   python src/gnn/train.py --epochs 20")
        sys.exit(1)
    
    print(f"📦 Checkpoint: {checkpoint_path}")
    
    # 1. Dataset yükle
    print(f"\n📊 Dataset yükleniyor...")
    dataset = TrafficDataset(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        stride=args.stride
    )
    
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Nodes: {dataset.num_nodes}")
    print(f"  - Features: {dataset.num_features}")
    
    # 2. Train/val/test split (aynı seed ile)
    from torch.utils.data import random_split
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n  - Test: {len(test_dataset)} samples")
    
    # 3. DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 4. Chebyshev basis (with Gaussian Kernel)
    print(f"\n🔄 Chebyshev basis hesaplanıyor (Gaussian Kernel)...")
    sample = dataset[0]
    edge_index = sample['edge_index']
    edge_dist = sample['edge_attr'].squeeze()
    
    # Apply Gaussian kernel
    sigma = args.sigma
    edge_weight = torch.exp(- (edge_dist ** 2) / (sigma ** 2))
    edge_weight[edge_weight < 0.1] = 0.0
    
    print(f"  ✓ Gaussian kernel (sigma={sigma}m)")
    
    cheb_basis = precompute_cheb_basis(
        edge_index=edge_index,
        num_nodes=dataset.num_nodes,
        k_order=3,
        edge_weight=edge_weight
    )
    
    # 5. Model yükle
    print(f"\n🏗️  Model yükleniyor...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    model = STAGCN(
        num_nodes=dataset.num_nodes,
        in_channels=dataset.num_features,
        hidden_channels=[64, 64, 32],
        out_channels=dataset.num_features,
        k_order=3,
        kernel_size=3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    
    # 6. Evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        cheb_basis=cheb_basis,
        device=args.device
    )
    
    metrics, predictions, targets, errors = evaluator.evaluate()
    
    # 7. Visualization
    evaluator.visualize_predictions(predictions, targets, args.save_dir)
    
    # 8. Baseline karşılaştırma
    if args.compare_baselines:
        print("\n" + "="*70)
        print("🔍 Baseline Model Karşılaştırması")
        print("="*70)
        
        baseline_results = []
        
        # Historical Average
        baseline_results.append(BaselineModels.historical_average(test_loader))
        
        # Last Value
        baseline_results.append(BaselineModels.last_value(test_loader))
        
        # Linear Regression
        baseline_results.append(BaselineModels.linear_regression(test_loader))
        
        # STA-GCN (bizim model)
        baseline_results.append({
            'model': 'STA-GCN (Ours)',
            'mae': metrics['test_mae'],
            'rmse': metrics['test_rmse']
        })
        
        # Comparison table
        print("\n📊 Model Karşılaştırması:")
        print("-" * 70)
        print(f"{'Model':<25} {'MAE':<15} {'RMSE':<15} {'Improvement':<15}")
        print("-" * 70)
        
        baseline_mae = baseline_results[0]['mae']  # Historical average
        
        for result in baseline_results:
            improvement = ((baseline_mae - result['mae']) / baseline_mae * 100)
            print(f"{result['model']:<25} {result['mae']:<15.4f} {result['rmse']:<15.4f} {improvement:>+13.1f}%")
        
        print("-" * 70)
        
        # Save comparison
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        baseline_results_serializable = []
        for result in baseline_results:
            baseline_results_serializable.append({
                'model': result['model'],
                'mae': float(result['mae']),
                'rmse': float(result['rmse'])
            })
        
        with open(save_dir / 'comparison.json', 'w') as f:
            json.dump(baseline_results_serializable, f, indent=2)
        
        print(f"\n✅ Karşılaştırma kaydedildi: {save_dir / 'comparison.json'}")
    
    # 9. Final summary
    print("\n" + "="*70)
    print("✅ Evaluation Tamamlandı!")
    print("="*70)
    print(f"\n📁 Sonuçlar: {args.save_dir}")
    print(f"  - prediction_scatter.png")
    print(f"  - timeseries_sample.png")
    print(f"  - feature_mae.png")
    if args.compare_baselines:
        print(f"  - comparison.json")
    print()


if __name__ == "__main__":
    main()
