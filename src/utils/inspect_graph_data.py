#!/usr/bin/env python3
"""
inspect_graph_data.py
---------------------
Edge List ve Feature Matrix'i görselleştir ve dışa aktar.

Kullanım:
    python src/utils/inspect_graph_data.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import pandas as pd
from src.gnn.dataset_sta import TrafficDataset


def main():
    print("📊 Graf Verisi Yükleniyor...\n")
    
    # Dataset yükle
    dataset = TrafficDataset(window_size=12, prediction_horizon=12)
    
    # ========== DR-21: Edge List ==========
    print("=" * 60)
    print("DR-21: Edge List (sourceNodeIndex, targetNodeIndex)")
    print("=" * 60)
    
    edge_index = dataset.edge_index.numpy()
    edge_attr = dataset.edge_attr.squeeze().numpy()
    
    edge_df = pd.DataFrame({
        'sourceNodeIndex': edge_index[0],
        'targetNodeIndex': edge_index[1],
        'weight': edge_attr
    })
    
    print(f"\n✅ Toplam Edge Sayısı: {len(edge_df):,}")
    print(f"✅ Format: (sourceNodeIndex, targetNodeIndex, weight)")
    print(f"\nİlk 10 Edge:\n{edge_df.head(10)}")
    
    # CSV'ye kaydet
    output_dir = Path('outputs/graph_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(output_dir / 'edge_list.csv', index=False)
    print(f"\n💾 Kaydedildi: outputs/graph_data/edge_list.csv")
    
    # ========== DR-22: Node → SegmentId Eşlemesi ==========
    print("\n" + "=" * 60)
    print("DR-22: Node Index → segmentId Eşlemesi")
    print("=" * 60)
    
    # dataset.segment_ids: List[str] - sıralı segment ID'leri
    # dataset.sid_to_idx: Dict[str, int] - segment ID -> node index
    node_mapping_df = pd.DataFrame({
        'nodeIndex': list(range(len(dataset.segment_ids))),
        'segmentId': dataset.segment_ids
    })
    
    print(f"\n✅ Toplam Node Sayısı: {len(node_mapping_df):,}")
    print(f"\nİlk 10 Node:\n{node_mapping_df.head(10)}")
    
    node_mapping_df.to_csv(output_dir / 'node_mapping.csv', index=False)
    print(f"\n💾 Kaydedildi: outputs/graph_data/node_mapping.csv")
    
    # ========== DR-23: Feature Matrix ==========
    print("\n" + "=" * 60)
    print("DR-23: Node Feature Matrix (T × N × F)")
    print("=" * 60)
    
    x = dataset.x  # (T, N, F)
    T, N, F = x.shape
    
    print(f"\n✅ Boyut: ({T:,} × {N:,} × {F})")
    print(f"   T (Zaman Adımları): {T:,}")
    print(f"   N (Node/Segment): {N:,}")
    print(f"   F (Özellik Sayısı): {F}")
    
    # TrafficDataset'teki actual feature isimleri (8 feature)
    feature_names = ['speed_norm', 'jf_norm', 'conf', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend']
    print(f"\n✅ Özellikler: {feature_names}")
    
    # İlk zaman adımı için örnek
    sample_t0 = x[0].numpy()  # (N, F)
    sample_df = pd.DataFrame(sample_t0, columns=feature_names)
    sample_df['nodeIndex'] = range(N)
    sample_df = sample_df[['nodeIndex'] + feature_names]
    
    print(f"\nİlk Zaman Adımı (t=0) için İlk 10 Node:\n{sample_df.head(10)}")
    
    sample_df.to_csv(output_dir / 'feature_matrix_sample_t0.csv', index=False)
    print(f"\n💾 Kaydedildi: outputs/graph_data/feature_matrix_sample_t0.csv")
    
    # ========== DR-24: İstatistikler ==========
    print("\n" + "=" * 60)
    print("DR-24: Feature İstatistikleri (GCN/GNN Uyumluluğu)")
    print("=" * 60)
    
    # Tüm zaman adımları için istatistikler
    stats = {
        'Feature': feature_names,
        'Mean': x.mean(dim=[0, 1]).tolist(),
        'Std': x.std(dim=[0, 1]).tolist(),
        'Min': x.min(dim=0)[0].min(dim=0)[0].tolist(),
        'Max': x.max(dim=0)[0].max(dim=0)[0].tolist()
    }
    
    stats_df = pd.DataFrame(stats)
    print(f"\n{stats_df}")
    
    stats_df.to_csv(output_dir / 'feature_statistics.csv', index=False)
    print(f"\n💾 Kaydedildi: outputs/graph_data/feature_statistics.csv")
    
    # ========== Özet ==========
    print("\n" + "=" * 60)
    print("📋 ÖZET")
    print("=" * 60)
    print(f"✅ DR-21: Edge List → edge_list.csv ({len(edge_df):,} edge)")
    print(f"✅ DR-22: Node Mapping → node_mapping.csv ({len(node_mapping_df):,} node)")
    print(f"✅ DR-23: Feature Matrix → T={T:,}, N={N:,}, F={F}")
    print(f"✅ DR-24: Feature Stats → Ortalama hız, jam_factor, confidence, functional_class")
    print(f"\n📁 Tüm dosyalar: outputs/graph_data/")


if __name__ == "__main__":
    main()
