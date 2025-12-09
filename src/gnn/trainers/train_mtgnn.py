#!/usr/bin/env python3
"""
train_mtgnn.py
--------------
MTGNN Model Eğitim Scripti.

STGMS trainer yapısını korur ancak MTGNN'e özgü veri hazırlığı
ve model parametrelerini içerir.

Kullanım:
    python src/gnn/trainers/train_mtgnn.py --epochs 100 --batch_size 32
"""

import argparse
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Local imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.gnn.dataset_mtgnn import MTGNNDataset
from src.gnn.models.mtgnn import MTGNN


def edge_index_to_dense(edge_index, edge_attr, num_nodes, device):
    """
    Neo4j sparse edge_index'i dense ve NORMALIZE EDİLMİŞ adjacency matrisine çevirir.
    
    Args:
        edge_index: (2, E) - Edge bağlantıları
        edge_attr: (E, 1) veya (E,) - Edge ağırlıkları (Gaussian Kernel vb.)
        num_nodes: Node sayısı
        device: Torch device
    
    Returns:
        adj: (N, N) - Row-normalized weighted adjacency matrix
    """
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    
    # 1. Ağırlıkları yerleştir (Binary yerine edge_attr kullan)
    # edge_attr genelde (E, 1) gelir, squeeze ile (E,) yapıyoruz.
    if edge_attr is not None:
        adj[edge_index[0], edge_index[1]] = edge_attr.squeeze().to(device)
    else:
        adj[edge_index[0], edge_index[1]] = 1.0
        
    # 2. Row-wise Normalization (D^-1 * A)
    # Her satırın toplamını bul
    row_sum = adj.sum(dim=1)
    # Sıfıra bölme hatasını önle
    row_sum[row_sum == 0] = 1.0
    
    # Broadcasting ile bölme
    adj = adj / row_sum.unsqueeze(1)
    
    return adj


class MTGNNTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr, weight_decay, patience, checkpoint_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.L1Loss()  # MTGNN genelde MAE (L1) kullanır
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            x = batch['x'].to(self.device)  # (B, T, N, F)
            y = batch['y'].to(self.device)  # (B, T_out, N, F)
            
            # MTGNN tek adım (single step) veya sequence output verebilir.
            # Bu implementasyonda horizon=1 varsayıyoruz veya son adımı alıyoruz.
            if y.shape[1] > 1:
                y = y[:, -1:, :, :]  # Sadece son adımı tahmin et (Many-to-One)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            # edge_index model içinde opsiyonel, çünkü Graph Learning var.
            pred = self.model(x)
            
            loss = self.criterion(pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                if y.shape[1] > 1:
                    y = y[:, -1:, :, :]
                
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def train(self, epochs):
        print(f"\n🚀 MTGNN Training Başlıyor...")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-start:.1f}s")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model_mtgnn.pt')
                print("  ✅ Model kaydedildi.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("⚠️ Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=1)  # MTGNN genelde 1 adım tahmin eder
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_static_graph', action='store_true', help="Neo4j grafını kullan")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset
    print("📦 Dataset yükleniyor...")
    dataset = MTGNNDataset(window_size=args.window_size, prediction_horizon=args.horizon)
    
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_ds, val_ds, _ = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Static Graph Hazırlığı (Hibrit Mod)
    predefined_adj = None
    if args.use_static_graph:
        print("🔗 Neo4j Statik Grafı hazırlanıyor...")
        # Dataset'ten edge_index ve edge_attr'ı alıp dense matrise çeviriyoruz
        edge_index = dataset.edge_index.to(device)
        edge_attr = dataset.edge_attr if hasattr(dataset, 'edge_attr') else None
        predefined_adj = edge_index_to_dense(edge_index, edge_attr, dataset.num_nodes, device)
        print(f"   ✅ Adjacency matrix oluşturuldu: {predefined_adj.shape}, normalized & weighted")
    
    # 3. Model
    print("🏗️ MTGNN Modeli oluşturuluyor...")
    model = MTGNN(
        gcn_true=True,
        build_adj=True,  # Graph Learning AKTİF
        gcn_depth=2,
        num_nodes=dataset.num_nodes,
        device=device,
        predefined_adj=predefined_adj,  # Hibrit yapı için statik graf
        in_dim=dataset.num_features,
        out_dim=dataset.num_features,
        seq_length=args.window_size,
        layers=3
    )
    
    # 4. Trainer
    trainer = MTGNNTrainer(
        model, train_loader, val_loader, device, 
        lr=0.001, weight_decay=1e-5, patience=10, 
        checkpoint_dir='outputs/models/mtgnn'
    )
    
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
