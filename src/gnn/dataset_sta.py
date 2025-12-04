#!/usr/bin/env python3
"""
dataset_sta.py
--------------
PyTorch Dataset sınıfı - Neo4j'den STA-GCN için veri yükler

STA-GCN (Spatio-Temporal Graph Convolutional Network) için özel olarak tasarlanmış
dataset sınıfı. Neo4j'deki Segment ve Measure verilerini PyTorch tensörlerine çevirir.

Özellikler:
- Temporal window sliding (zaman penceresi kaydırma)
- Node feature normalizasyon
- Edge feature extraction (CONNECTS_TO distance)
- Batch processing desteği

Kullanım:
    from src.gnn.dataset_sta import TrafficDataset
    
    dataset = TrafficDataset(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="123456789",
        window_size=12,
        prediction_horizon=3
    )
    
    # PyTorch DataLoader ile kullan
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from neo4j import GraphDatabase
from dotenv import load_dotenv

# .env yükle
ENV_PATH = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)


class TrafficDataset(Dataset):
    """
    STA-GCN için trafik veri seti
    
    Args:
        neo4j_uri: Neo4j bağlantı URI'si
        neo4j_user: Neo4j kullanıcı adı
        neo4j_pass: Neo4j şifresi
        neo4j_database: Neo4j veritabanı adı
        window_size: Geçmiş zaman adımı sayısı (örn: 12 = 3 saat, 15dk interval)
        prediction_horizon: Tahmin edilecek gelecek adım sayısı (örn: 3 = 45dk)
        stride: Pencere kaydırma adımı (1 = overlapping, >1 = non-overlapping)
        feature_cols: Kullanılacak özellik kolonları
    """
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_pass: str = None,
        neo4j_database: str = None,
        window_size: int = 12,
        prediction_horizon: int = 3,
        stride: int = 1,
        feature_cols: List[str] = None,
        start_time: str = None,
        end_time: str = None,
        use_last_n_days: int = None
    ):
        super().__init__()
        
        # Bağlantı parametreleri
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_pass = neo4j_pass or os.getenv("NEO4J_PASS", "123456789")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Dataset parametreleri
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        # Timestamp filtreleme
        self.start_time = start_time
        self.end_time = end_time
        self.use_last_n_days = use_last_n_days
        
        # Özellik kolonları (05_export_pyg.py ile uyumlu)
        self.feature_cols = feature_cols or [
            "speed_norm", "jf_norm", "conf",
            "sin_hour", "cos_hour", "sin_dow", "cos_dow", "is_weekend"
        ]
        
        print("=" * 70)
        print("🔄 TrafficDataset Initialization")
        print("=" * 70)
        print(f"Window size: {window_size} steps")
        print(f"Prediction horizon: {prediction_horizon} steps")
        print(f"Stride: {stride}")
        print(f"Features: {len(self.feature_cols)} → {self.feature_cols}")
        
        # Veriyi yükle
        self._load_data()
        self._create_windows()
        
        print(f"✅ Dataset hazır: {len(self)} samples")
        print()
    
    def _load_data(self):
        """Neo4j'den veriyi çek ve tensörlere dönüştür"""
        print("\n📡 Veriler Neo4j'den yükleniyor...")
        
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_pass)
        )
        
        try:
            with driver.session(database=self.neo4j_database) as session:
                # 1. Segment ID'leri al (sıralı)
                result = session.run("""
                    MATCH (s:Segment)
                    RETURN s.segmentId AS sid
                    ORDER BY sid
                """)
                self.segment_ids = [r["sid"] for r in result]
                self.sid_to_idx = {sid: i for i, sid in enumerate(self.segment_ids)}
                self.num_nodes = len(self.segment_ids)
                
                print(f"  ✓ {self.num_nodes} segment yüklendi")
                
                # 2. CONNECTS_TO edge'leri al
                result = session.run("""
                    MATCH (a:Segment)-[r:CONNECTS_TO]-(b:Segment)
                    RETURN a.segmentId AS u, b.segmentId AS v, r.distance AS dist
                """)
                edges = [(r["u"], r["v"], r.get("dist", 0.0)) for r in result]
                
                # Edge index ve edge attributes
                edge_u = []
                edge_v = []
                edge_dist = []
                
                for u_sid, v_sid, dist in edges:
                    if u_sid in self.sid_to_idx and v_sid in self.sid_to_idx:
                        edge_u.append(self.sid_to_idx[u_sid])
                        edge_v.append(self.sid_to_idx[v_sid])
                        edge_dist.append(dist)
                
                self.edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
                self.edge_attr = torch.tensor(edge_dist, dtype=torch.float32).unsqueeze(1)
                
                print(f"  ✓ {self.edge_index.shape[1]} edge yüklendi")
                
                # 3. Measure verilerini al (timestamp sıralı)
                # Timestamp filtresi oluştur
                where_clause = ""
                params = {}
                
                if self.use_last_n_days:
                    # Son N gün
                    where_clause = "WHERE datetime(m.timestamp) >= datetime() - duration({days: $n_days})"
                    params["n_days"] = self.use_last_n_days
                elif self.start_time and self.end_time:
                    # Belirli tarih aralığı
                    where_clause = "WHERE m.timestamp >= $start_time AND m.timestamp <= $end_time"
                    params["start_time"] = self.start_time
                    params["end_time"] = self.end_time
                elif self.start_time:
                    # Sadece başlangıç
                    where_clause = "WHERE m.timestamp >= $start_time"
                    params["start_time"] = self.start_time
                elif self.end_time:
                    # Sadece bitiş
                    where_clause = "WHERE m.timestamp <= $end_time"
                    params["end_time"] = self.end_time
                
                query = f"""
                    MATCH (s:Segment)-[:AT_TIME]->(m:Measure)
                    {where_clause}
                    RETURN s.segmentId AS sid,
                           m.timestamp AS ts,
                           m.speed AS sp,
                           m.freeFlow AS ff,
                           m.jamFactor AS jf,
                           m.confidence AS conf
                    ORDER BY m.timestamp, s.segmentId
                """
                
                result = session.run(query, **params)
                
                measures = []
                for r in result:
                    measures.append({
                        "sid": r["sid"],
                        "ts": r["ts"],
                        "sp": r["sp"],
                        "ff": r["ff"],
                        "jf": r["jf"],
                        "conf": r["conf"]
                    })
                
                if not measures:
                    raise ValueError("Neo4j'de Measure verisi bulunamadı! Önce veri yükleyin.")
                
                df = pd.DataFrame(measures)
                
                # Tarih aralığını göster
                if len(df) > 0:
                    min_ts = df['ts'].min()
                    max_ts = df['ts'].max()
                    print(f"  ✓ {len(df)} measure kaydı yüklendi")
                    print(f"  📅 Tarih aralığı: {min_ts} → {max_ts}")
                else:
                    print(f"  ⚠️  Filtre sonucu 0 kayıt!")
                
        finally:
            driver.close()
        
        # 4. Feature engineering (04_generate_features.py ile aynı)
        print("\n🔄 Feature engineering...")
        
        # Timestamp'i datetime'a çevir
        df["dt"] = pd.to_datetime(df["ts"])
        
        # speed_norm = min(speed / freeFlow, 2.0)
        df["ff_eps"] = df["ff"].fillna(1e-6).replace(0.0, 1e-6)
        df["speed_norm"] = (df["sp"] / df["ff_eps"]).clip(0.0, 2.0)
        
        # jf_norm = jamFactor / 10
        df["jf_norm"] = (df["jf"].fillna(0.0) / 10.0).clip(0.0, 1.0)
        
        # confidence [0, 1]
        df["conf"] = df["conf"].fillna(0.0).clip(0.0, 1.0)
        
        # Zaman özellikleri (sinüs/kosinüs - döngüsel)
        hours = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
        df["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
        
        dow = df["dt"].dt.dayofweek
        df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
        
        df["is_weekend"] = (df["dt"].dt.dayofweek >= 5).astype(float)
        
        print(f"  ✓ {len(self.feature_cols)} feature hazırlandı")
        
        # 5. Tensor oluştur: (T, N, F)
        self.timestamps = sorted(df["ts"].unique())
        self.time_to_idx = {t: i for i, t in enumerate(self.timestamps)}
        
        T = len(self.timestamps)
        N = self.num_nodes
        F = len(self.feature_cols)
        
        print(f"\n📊 Tensor boyutları: T={T}, N={N}, F={F}")
        
        # Sıfır matrisi
        x_np = np.zeros((T, N, F), dtype=np.float32)
        
        # DataFrame'den doldur
        for _, row in df.iterrows():
            t_idx = self.time_to_idx.get(row["ts"])
            n_idx = self.sid_to_idx.get(row["sid"])
            
            if t_idx is None or n_idx is None:
                continue
            
            x_np[t_idx, n_idx, :] = [row[col] for col in self.feature_cols]
        
        self.x = torch.tensor(x_np, dtype=torch.float32)
        
        # NaN kontrolü
        nan_count = torch.isnan(self.x).sum().item()
        if nan_count > 0:
            print(f"  ⚠️  {nan_count} NaN değer tespit edildi, 0 ile dolduruldu")
            self.x = torch.nan_to_num(self.x, nan=0.0)
        
        print(f"  ✓ Feature tensor hazır: {self.x.shape} (RAW - normalizasyon train.py'de yapılacak)")
        
        # num_features property için
        self.num_features = len(self.feature_cols)
    
    def _create_windows(self):
        """Temporal window'ları oluştur (sliding window)"""
        print("\n🪟 Sliding windows oluşturuluyor...")
        
        T = self.x.shape[0]
        total_len = self.window_size + self.prediction_horizon
        
        if T < total_len:
            raise ValueError(
                f"Yetersiz zaman adımı! T={T}, "
                f"gerekli={total_len} (window={self.window_size} + horizon={self.prediction_horizon})"
            )
        
        # Window başlangıç indeksleri
        self.window_starts = list(range(0, T - total_len + 1, self.stride))
        
        print(f"  ✓ {len(self.window_starts)} window oluşturuldu (stride={self.stride})")
    
    def __len__(self) -> int:
        """Dataset boyutu"""
        return len(self.window_starts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Bir sample döndür
        
        Returns:
            {
                'x': (window_size, N, F) - Geçmiş features
                'y': (prediction_horizon, N, F) - Gelecek features (hedef)
                'edge_index': (2, E) - Graf yapısı
                'edge_attr': (E, 1) - Edge features
                'timestamp': str - İlk timestamp
            }
        """
        start_idx = self.window_starts[idx]
        
        # Geçmiş window
        x_window = self.x[start_idx : start_idx + self.window_size]
        
        # Gelecek target
        y_window = self.x[
            start_idx + self.window_size : start_idx + self.window_size + self.prediction_horizon
        ]
        
        return {
            'x': x_window,  # (T_in, N, F)
            'y': y_window,  # (T_out, N, F)
            'edge_index': self.edge_index,  # (2, E)
            'edge_attr': self.edge_attr,    # (E, 1)
            'timestamp': self.timestamps[start_idx]
        }
    
    def get_node_info(self, node_idx: int) -> Dict:
        """Bir node'un bilgisini döndür (debug için)"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            raise IndexError(f"Node index {node_idx} out of range [0, {self.num_nodes})")
        
        return {
            'node_idx': node_idx,
            'segment_id': self.segment_ids[node_idx]
        }
    
    def normalize_features(self, method: str = 'zscore'):
        """
        Feature normalizasyonu (opsiyonel)
        
        Args:
            method: 'zscore' veya 'minmax'
        """
        print(f"\n📏 Feature normalization ({method})...")
        
        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = self.x.mean(dim=(0, 1), keepdim=True)
            std = self.x.std(dim=(0, 1), keepdim=True)
            self.x = (self.x - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            # Min-Max normalization: (x - min) / (max - min)
            min_val = self.x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
            max_val = self.x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            self.x = (self.x - min_val) / (max_val - min_val + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        print(f"  ✓ Features normalized ({method})")


# Test fonksiyonu
def test_dataset():
    """Dataset test"""
    print("\n" + "="*70)
    print("🧪 TrafficDataset Test")
    print("="*70 + "\n")
    
    # Dataset oluştur
    dataset = TrafficDataset(
        window_size=12,
        prediction_horizon=3,
        stride=1
    )
    
    print(f"\n📊 Dataset İstatistikleri:")
    print(f"  - Toplam sample: {len(dataset)}")
    print(f"  - Node sayısı: {dataset.num_nodes}")
    print(f"  - Edge sayısı: {dataset.edge_index.shape[1]}")
    print(f"  - Zaman adımı: {len(dataset.timestamps)}")
    print(f"  - Feature dim: {len(dataset.feature_cols)}")
    
    # İlk sample'ı al
    sample = dataset[0]
    print(f"\n🔍 İlk Sample:")
    print(f"  - x shape: {sample['x'].shape}")
    print(f"  - y shape: {sample['y'].shape}")
    print(f"  - edge_index shape: {sample['edge_index'].shape}")
    print(f"  - timestamp: {sample['timestamp']}")
    
    # Son sample
    sample_last = dataset[-1]
    print(f"\n🔍 Son Sample:")
    print(f"  - timestamp: {sample_last['timestamp']}")
    
    print("\n✅ Test tamamlandı!\n")


if __name__ == "__main__":
    test_dataset()
