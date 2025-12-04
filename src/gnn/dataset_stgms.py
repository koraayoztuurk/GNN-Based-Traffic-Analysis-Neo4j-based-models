#!/usr/bin/env python3
"""
dataset_stgms.py
----------------
STGMS (Spatio-Temporal Graph Neural Network with Multi-timeScale) için özelleştirilmiş Dataset

Bu modül, STGMS makalesindeki "Multi-timescale Feature Decomposition" (Eq. 2 & 3) işlemini
uygulayarak trafik verilerini farklı periyotlara (haftalık, günlük, saatlik) ayrıştırır.

Temel Özellikler:
- Online Decomposition: Causal padding ile gelecekten bilgi sızıntısını önler
- Multi-scale Periods: Trend, günlük döngü ve anlık değişimleri ayrıştırır
- Feature Augmentation: (T, N, F) -> (T, N, F * (m+1)) boyut artırımı

Kullanım:
    from src.gnn.dataset_stgms import STGMSDataset
    
    dataset = STGMSDataset(
        window_size=12,
        periods=[96, 16, 4]  # Günlük, 4-saatlik, 1-saatlik
    )

Referans:
    "Spatio-Temporal Graph Neural Network with Multi-timeScale"
    - Eq. 2: Online-decomposing algorithm
    - Eq. 3: Multi-scale feature concatenation
"""

import torch
import torch.nn.functional as func_F  # Alias to avoid conflict with variable F
from typing import List, Dict

# TrafficDataset'i import et
from src.gnn.dataset_sta import TrafficDataset


class STGMSDataset(TrafficDataset):
    """
    STGMS Modeli için özelleştirilmiş Dataset.
    
    TrafficDataset'ten miras alır ve multi-timescale decomposition ekler.
    Makaledeki 'Multi-timescale Feature Decomposition' (Eq. 2 & 3) işlemini uygular.
    
    Args:
        periods (list): Ayrıştırılacak periyotlar (büyükten küçüğe sıralı).
                       15dk interval verisi için örnek:
                       - 96 (1 gün = 24 saat * 4)
                       - 16 (4 saat = 4 * 4)
                       - 4 (1 saat = 1 * 4)
        **kwargs: TrafficDataset argümanları (window_size, prediction_horizon, vb.)
    
    Attributes:
        periods (list): Kullanılan periyotlar
        decomposed_x (torch.Tensor): Ayrıştırılmış özellikler (T, N, F * (m+1))
        num_features_decomposed (int): Yeni feature boyutu
    """
    
    def __init__(self, periods: List[int] = None, **kwargs):
        """
        Args:
            periods: Ayrıştırılacak periyotlar. None ise [96, 16, 4] kullanılır.
            **kwargs: TrafficDataset'e aktarılacak tüm parametreler
        """
        # Varsayılan periyotlar (15dk interval için)
        if periods is None:
            periods = [96, 16, 4]  # 1 gün, 4 saat, 1 saat
        
        # Parent class'ı başlat (veriyi yükler)
        super().__init__(**kwargs)
        
        # Periyotları büyükten küçüğe sırala (P1 > P2 > ... > Pm)
        self.periods = sorted(periods, reverse=True)
        
        # Veriyi önceden ayrıştır (Pre-compute decomposition)
        # Bu işlem CPU'da yapılır, eğitim sırasında hız kazandırır.
        print(f"\n🧩 STGMS Multi-timescale Decomposition başlatılıyor...")
        print(f"   Periyotlar: {self.periods}")
        
        self.decomposed_x = self._decompose_data(self.x, self.periods)
        
        # Yeni feature boyutu: Original_F * (Num_Periods + 1)
        # +1: Son residual bileşen için
        self.num_features_decomposed = self.x.shape[2] * (len(self.periods) + 1)
        
        print(f"   ✓ Decomposition tamamlandı")
        print(f"   - Orijinal Feature boyutu: {self.x.shape[2]}")
        print(f"   - Yeni Feature boyutu: {self.num_features_decomposed}")
        print(f"   - Decomposed X shape: {self.decomposed_x.shape}")
        print()
    
    def _decompose_data(self, x_tensor: torch.Tensor, periods: List[int]) -> torch.Tensor:
        """
        Multi-timescale Feature Decomposition (Makale Eq. 2 & 3)
        
        Algoritma:
        1. Her periyot P için:
           - Moving average hesapla (causal padding ile)
           - X^i = MovingAvg_P(S^{i-1})
           - Residual güncelle: S^i = S^{i-1} - X^i
        2. Tüm bileşenleri concatenate et: [X^1, X^2, ..., X^m, S^m]
        
        Causal Padding:
        - Gelecekten bilgi sızıntısını önlemek için padding=(P-1, 0)
        - Bu sayede t anındaki değer sadece [t-P+1, t] aralığını görür
        
        Args:
            x_tensor: Orijinal feature tensörü (T, N, F)
            periods: Periyot listesi [P1, P2, ..., Pm] (büyükten küçüğe)
        
        Returns:
            decomposed: Ayrıştırılmış tensör (T, N, F * (m+1))
        """
        T, N, F = x_tensor.shape
        
        # İşlem kolaylığı için (N, F, T) formatına çevir
        # PyTorch Conv1d/AvgPool1d (Batch, Channel, Length) bekler.
        # Burada: Batch=N, Channel=F, Length=T
        signal = x_tensor.permute(1, 2, 0)  # (N, F, T)
        
        components = []
        current_signal = signal.clone()
        
        print(f"   Ayrıştırma başlıyor...")
        for i, P in enumerate(periods, 1):
            # Moving Average (Causal Padding ile)
            # Kernel size = P
            # Left padding = P-1 (geçmiş), Right padding = 0 (gelecek yok)
            # Bu sayede t anındaki değer [t-P+1, t] aralığının ortalaması olur
            
            # Padding: (Left, Right) -> (P-1, 0)
            # mode='replicate': Sınırlardaki değerleri tekrarla
            padded_signal = func_F.pad(current_signal, (P - 1, 0), mode='replicate')
            
            # Average Pooling (Moving Average)
            # X^i = MovingAvg(S^{i-1})
            # avg_pool1d: kernel_size=P, stride=1 (her adımda 1 kaydır)
            component = func_F.avg_pool1d(padded_signal, kernel_size=P, stride=1)
            
            components.append(component)
            
            # Residual hesapla (Eq. 2): S^i = S^{i-1} - X^i
            current_signal = current_signal - component
            
            print(f"     Periyot {i} (P={P}): Component shape={component.shape}")
        
        # Son kalan sinyal Residual (S^m) - Anlık değişimler/gürültü
        components.append(current_signal)
        print(f"     Residual (Anlık değişimler): shape={current_signal.shape}")
        
        # Eq. 3: Concatenation along feature dimension
        # components listesi: [X^1, X^2, ..., X^m, S^m]
        # Her biri: (N, F, T)
        # Concatenate: (N, F * (m+1), T)
        decomposed = torch.cat(components, dim=1)
        
        # Orijinal formata geri dön: (T, N, F_new)
        # (N, F_new, T) -> (T, N, F_new)
        decomposed = decomposed.permute(2, 0, 1)
        
        return decomposed
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Bir sample döndür (Decomposed features ile)
        
        Override: Parent class'ın __getitem__ metodunu değiştirir.
        X için decomposed features kullanır, Y için orijinal değerleri kullanır.
        
        Args:
            idx: Sample indeksi
        
        Returns:
            {
                'x': (window_size, N, F_decomposed) - Geçmiş (decomposed)
                'y': (prediction_horizon, N, F_original) - Gelecek (raw target)
                'edge_index': (2, E) - Graf yapısı
                'edge_attr': (E, 1) - Edge features
                'timestamp': str - İlk timestamp
            }
        
        Not:
            Target (y) genellikle raw değer olarak kalır.
            Makale, sadece input'u decompose ettiğini belirtmiş.
        """
        start_idx = self.window_starts[idx]
        
        # Decomposed X (Geçmiş)
        # decomposed_x shape: (T, N, F_decomposed)
        x_window = self.decomposed_x[start_idx : start_idx + self.window_size]
        
        # Target Y (Gelecek - Raw değer olarak kullanılır)
        # Makale, target'ı decompose etmediğini ima ediyor
        # Model, raw değeri tahmin etmeyi öğrenir
        y_window = self.x[
            start_idx + self.window_size : start_idx + self.window_size + self.prediction_horizon
        ]
        
        return {
            'x': x_window,          # (Window, N, F_decomposed)
            'y': y_window,          # (Horizon, N, F_original)
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'timestamp': self.timestamps[start_idx]
        }


# Test fonksiyonu
def test_stgms_dataset():
    """STGMSDataset test"""
    print("\n" + "="*70)
    print("🧪 STGMSDataset Test")
    print("="*70 + "\n")
    
    # Dataset oluştur
    dataset = STGMSDataset(
        window_size=12,
        prediction_horizon=3,
        periods=[96, 16, 4],  # 1 gün, 4 saat, 1 saat
        stride=1
    )
    
    print(f"\n📊 Dataset İstatistikleri:")
    print(f"  - Toplam sample: {len(dataset)}")
    print(f"  - Node sayısı: {dataset.num_nodes}")
    print(f"  - Orijinal Feature dim: {dataset.num_features}")
    print(f"  - Decomposed Feature dim: {dataset.num_features_decomposed}")
    print(f"  - Feature artış oranı: {dataset.num_features_decomposed / dataset.num_features:.1f}x")
    
    # İlk sample'ı al
    sample = dataset[0]
    print(f"\n🔍 İlk Sample:")
    print(f"  - x shape (decomposed): {sample['x'].shape}")
    print(f"  - y shape (original): {sample['y'].shape}")
    print(f"  - timestamp: {sample['timestamp']}")
    
    # Feature değer dağılımı
    print(f"\n📈 Feature Statistics (decomposed X):")
    print(f"  - Mean: {sample['x'].mean():.4f}")
    print(f"  - Std: {sample['x'].std():.4f}")
    print(f"  - Min: {sample['x'].min():.4f}")
    print(f"  - Max: {sample['x'].max():.4f}")
    
    print("\n✅ Test tamamlandı!\n")


if __name__ == "__main__":
    test_stgms_dataset()
