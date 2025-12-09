#!/usr/bin/env python3
"""
dataset_mtgnn.py
----------------
MTGNN Modeli için Dataset Sınıfı.

STGMSDataset yapısını baz alır ancak MTGNN'in ihtiyaç duyduğu
ham (raw) zaman serisi formatını sağlar. MTGNN, feature decomposition
yerine ham veriyi ve kendi içindeki 'Dilated Inception' katmanlarını kullanır.

Kullanım:
    from src.gnn.dataset_mtgnn import MTGNNDataset
    dataset = MTGNNDataset(window_size=12, prediction_horizon=12)
"""

import torch
from typing import Dict, List

# TrafficDataset'i import et (dataset_stgms.py ile aynı yol)
from src.gnn.dataset_sta import TrafficDataset


class MTGNNDataset(TrafficDataset):
    """
    MTGNN için özelleştirilmiş Dataset.
    
    Args:
        **kwargs: TrafficDataset argümanları (window_size, prediction_horizon, vb.)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # MTGNN ham veri ile çalışır, decomposition yapmıyoruz.
        # Ancak STGMS trainer ile uyumluluk için num_features_decomposed
        # değişkenini orijinal feature sayısına eşitliyoruz.
        self.num_features_decomposed = self.num_features
        
        print(f"\n📦 MTGNN Dataset Hazır")
        print(f"   - Window Size: {self.window_size}")
        print(f"   - Horizon: {self.prediction_horizon}")
        print(f"   - Nodes: {self.num_nodes}")
        print(f"   - Features: {self.num_features}")
        print()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Bir sample döndür.
        
        Returns:
            {
                'x': (window_size, N, F) - Geçmiş verisi
                'y': (prediction_horizon, N, F) - Hedef verisi
                'edge_index': (2, E) - Statik graf (Neo4j'den)
                'edge_attr': (E, 1) - Edge ağırlıkları
            }
        """
        start_idx = self.window_starts[idx]
        
        # Geçmiş Penceresi (Input)
        x_window = self.x[start_idx : start_idx + self.window_size]
        
        # Gelecek Penceresi (Target)
        y_window = self.x[
            start_idx + self.window_size : start_idx + self.window_size + self.prediction_horizon
        ]
        
        return {
            'x': x_window,          # (T_in, N, F)
            'y': y_window,          # (T_out, N, F)
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'timestamp': self.timestamps[start_idx]
        }
