#!/usr/bin/env python3
"""
evaluate_stgms.py
-----------------
STGMS model değerlendirme ve test scripti

Fonksiyonlar:
- Model test (loss, MAE, RMSE, MAPE, R²)
- Multi-timescale feature decomposition ile test
- Visualizasyon (prediction vs actual)
- Segment-level analiz

Kullanım:
    # Best model'i test et
    python src/gnn/evaluators/evaluate_stgms.py
    
    # Belirli checkpoint'i test et
    python src/gnn/evaluators/evaluate_stgms.py --checkpoint outputs/models/stgms/best_model_stgms.pt
    
    # Detaylı analiz
    python src/gnn/evaluators/evaluate_stgms.py --detailed
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

# Path ayarla
sys.path.append(str(Path(__file__).parent.parent))

from src.gnn.dataset_stgms import STGMSDataset
from src.gnn.models.stgms import STGMS
from src.gnn.graph_utils import precompute_cheb_basis


class STGMSEvaluator:
    """STGMS model değerlendirici"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, verbose: bool = True) -> Dict[str, float]:
        """Model performansını değerlendir"""
        
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        criterion = nn.MSELoss()
        
        if verbose:
            print("\n🔍 STGMS Evaluation başlıyor...")
        
        for batch in self.test_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'][0].to(self.device)
            edge_attr = batch['edge_attr'][0].to(self.device).squeeze(-1)
            
            # Prediction
            pred = self.model(x, edge_index, edge_attr)
            
            # Metrics
            loss = criterion(pred, y)
            mae = torch.abs(pred - y).mean()
            rmse = torch.sqrt(((pred - y) ** 2).mean())
            
            # MAPE (avoid division by zero)
            mape = torch.abs((y - pred) / (y + 1e-8)).mean() * 100
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_mape += mape.item()
            num_batches += 1
            
            # Collect for R²
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = total_rmse / num_batches
        avg_mape = total_mape / num_batches
        
        # R² score
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        ss_res = ((all_targets - all_predictions) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'mape': avg_mape,
            'r2': r2.item()
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Metrikleri yazdır"""
        print("\n" + "="*70)
        print("📊 STGMS Model Performansı")
        print("="*70)
        print(f"\n  MSE Loss:  {metrics['loss']:.6f}")
        print(f"  MAE:       {metrics['mae']:.6f}")
        print(f"  RMSE:      {metrics['rmse']:.6f}")
        print(f"  MAPE:      {metrics['mape']:.2f}%")
        print(f"  R² Score:  {metrics['r2']:.6f}")
        print("\n" + "="*70 + "\n")
    
    def detailed_analysis(self) -> Dict[str, any]:
        """Detaylı analiz - segment ve zaman bazlı"""
        
        print("\n🔬 Detaylı Analiz başlıyor...")
        
        all_predictions = []
        all_targets = []
        all_timestamps = []
        
        for batch in self.test_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            edge_index = batch['edge_index'][0].to(self.device)
            edge_attr = batch['edge_attr'][0].to(self.device).squeeze(-1)
            
            pred = self.model(x, edge_index, edge_attr)
            
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())
            
            if 'timestamp' in batch:
                all_timestamps.extend(batch['timestamp'])
        
        # Concatenate
        predictions = torch.cat(all_predictions, dim=0)  # (B, H, N, F)
        targets = torch.cat(all_targets, dim=0)  # (B, H, N, F)
        
        B, H, N, F = predictions.shape
        
        # Segment-level analysis
        segment_errors = torch.abs(predictions - targets).mean(dim=(0, 1, 3))  # (N,)
        
        # Time-level analysis (horizon)
        horizon_errors = torch.abs(predictions - targets).mean(dim=(0, 2, 3))  # (H,)
        
        # Feature-level analysis
        feature_errors = torch.abs(predictions - targets).mean(dim=(0, 1, 2))  # (F,)
        
        analysis = {
            'segment_mae': segment_errors.numpy(),
            'horizon_mae': horizon_errors.numpy(),
            'feature_mae': feature_errors.numpy(),
            'best_segments': segment_errors.argsort()[:10].tolist(),
            'worst_segments': segment_errors.argsort()[-10:].tolist()
        }
        
        print(f"\n📊 Detaylı İstatistikler:")
        print(f"  - En iyi 10 segment MAE ortalaması: {segment_errors[analysis['best_segments']].mean():.6f}")
        print(f"  - En kötü 10 segment MAE ortalaması: {segment_errors[analysis['worst_segments']].mean():.6f}")
        print(f"  - Horizon adımları MAE: {horizon_errors.numpy()}")
        
        return analysis
    
    def visualize_predictions(self, num_samples: int = 5, save_path: str = None):
        """Tahmin vs gerçek değerleri görselleştir"""
        
        print(f"\n📈 Visualization oluşturuluyor ({num_samples} sample)...")
        
        # İlk batch'i al
        batch = next(iter(self.test_loader))
        x = batch['x'][:num_samples].to(self.device)
        y = batch['y'][:num_samples].to(self.device)
        edge_index = batch['edge_index'][0].to(self.device)
        edge_attr = batch['edge_attr'][0].to(self.device).squeeze(-1)
        
        pred = self.model(x, edge_index, edge_attr)
        
        # CPU'ya al
        pred_np = pred.cpu().numpy()  # (num_samples, H, N, F)
        target_np = y.cpu().numpy()
        
        # İlk birkaç segment için plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(min(6, num_samples)):
            ax = axes[i]
            
            # İlk segment, ilk feature
            pred_sample = pred_np[i, :, 0, 0]  # (H,)
            target_sample = target_np[i, :, 0, 0]
            
            ax.plot(pred_sample, 'o-', label='Prediction', color='red', alpha=0.7)
            ax.plot(target_sample, 's-', label='Actual', color='blue', alpha=0.7)
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Horizon Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Saved: {save_path}")
        else:
            plt.savefig('outputs/evaluation/stgms/predictions_plot.png', dpi=150, bbox_inches='tight')
            print(f"  💾 Saved: outputs/evaluation/stgms/predictions_plot.png")
        
        plt.close()


def main():
    """Ana evaluation fonksiyonu"""
    
    parser = argparse.ArgumentParser(description='STGMS Model Evaluation')
    
    # Dataset args
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--prediction_horizon', type=int, default=3)
    parser.add_argument('--periods', type=int, nargs='+', default=[96, 16, 4])
    
    # Model args
    parser.add_argument('--k_order', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/models/stgms/best_model_stgms.pt',
                        help='Model checkpoint path')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--detailed', action='store_true',
                        help='Detailed analysis')
    
    args = parser.parse_args()
    
    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("\n" + "="*70)
    print("🎯 STGMS Model Evaluation")
    print("="*70)
    
    # 1. Dataset yükle
    print(f"\n📦 Loading STGMS dataset...")
    dataset = STGMSDataset(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        periods=args.periods
    )
    
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Nodes: {dataset.num_nodes}")
    print(f"  - Features (decomposed): {dataset.num_features_decomposed}")
    
    # Test split (son %15)
    test_size = int(0.15 * len(dataset))
    test_indices = list(range(len(dataset) - test_size, len(dataset)))
    test_dataset = Subset(dataset, test_indices)
    
    print(f"  - Test samples: {len(test_dataset)}")
    
    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 2. Model oluştur
    print(f"\n🏗️  Building STGMS model...")
    model = STGMS(
        num_nodes=dataset.num_nodes,
        in_channels=dataset.num_features_decomposed,
        out_channels=dataset.num_features,
        window_size=args.window_size,
        horizon=args.prediction_horizon,
        k_order=args.k_order,
        dropout=args.dropout
    )
    
    # 3. Checkpoint yükle
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint bulunamadı: {checkpoint_path}")
        print(f"   Önce model eğitimi yapın:")
        print(f"   python src/gnn/trainers/train_stgms.py --epochs 100")
        return
    
    print(f"\n💾 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  - Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    # 4. Evaluation
    evaluator = STGMSEvaluator(model, test_loader, args.device)
    metrics = evaluator.evaluate()
    
    # 5. Sonuçları göster
    evaluator.print_metrics(metrics)
    
    # 6. Detaylı analiz (opsiyonel)
    if args.detailed:
        print("\n" + "="*70)
        print("🔬 Detaylı Analiz")
        print("="*70)
        
        analysis = evaluator.detailed_analysis()
        
        # Visualization
        results_dir = Path("outputs/evaluation/stgms")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.visualize_predictions(
            num_samples=6,
            save_path=str(results_dir / "predictions_plot.png")
        )
        
        # Analiz kaydet
        analysis_file = results_dir / "detailed_analysis.json"
        
        # NumPy arrays'i list'e çevir (JSON serialization için)
        analysis_json = {
            'segment_mae': analysis['segment_mae'].tolist(),
            'horizon_mae': analysis['horizon_mae'].tolist(),
            'feature_mae': analysis['feature_mae'].tolist(),
            'best_segments': analysis['best_segments'],
            'worst_segments': analysis['worst_segments']
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_json, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved: {analysis_file}")
    
    # 7. Metrics kaydet
    results_dir = Path("outputs/evaluation/stgms")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "test_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved: {results_file}")
    
    print("\n✅ Evaluation tamamlandı!")
    print(f"\n📊 Sonuçlar: {results_dir.absolute()}\n")


if __name__ == "__main__":
    main()
