#!/usr/bin/env python3
"""
stgms.py
--------
STGMS (Spatio-Temporal Graph Neural Network with Multi-timeScale) Model Implementasyonu

Bu modül, STGMS makalesindeki ana mimariyi PyTorch ile uygular:
- Temporal Attention: Zaman adımları arası bağımlılıkları öğrenir
- Spatial Attention: Segment'ler arası bağımlılıkları öğrenir
- Chebyshev Graph Convolution (ChebNet): Graf topolojisini kullanır
- Spatiotemporal Encoding Blocks: Yukarıdaki bileşenleri birleştirir

Mimari Akış (Fig. 1):
    Input (T, N, F_decomposed)
    -> ST Block 1 (Temporal + Spatial + GCN)
    -> ST Block 2 (Temporal + Spatial + GCN)
    -> Output Layer (Fully Connected)
    -> Prediction (H, N, F)

Kullanım:
    from src.gnn.models.stgms import STGMS
    
    model = STGMS(
        num_nodes=100,
        in_channels=32,  # F_decomposed from STGMSDataset
        out_channels=8,  # F_original (hedef feature sayısı)
        window_size=12,
        horizon=3
    )

Referans:
    "Spatio-Temporal Graph Neural Network with Multi-timeScale"
    - Section 3.2: Temporal Attention
    - Section 3.3: Spatial Attention
    - Section 3.4: Chebyshev Graph Convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from typing import List


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism (Makale Section 3.2)
    
    Amacı: Farklı zaman adımları arasındaki bağımlılıkları öğrenmek.
    Çıktı: Attention matrix E (T, T) - Her zaman adımının diğerleriyle ilişkisi
    
    Formül (Basitleştirilmiş):
        E = softmax(Ve * sigmoid(score(X)) + be)
    
    Args:
        num_nodes: Node sayısı (N)
        in_channels: Feature boyutu (F)
        num_timesteps: Zaman adımı sayısı (T)
    """
    
    def __init__(self, num_nodes: int, in_channels: int, num_timesteps: int):
        super(TemporalAttention, self).__init__()
        
        # Learnable parameters (Sadece kullanılanlar)
        # Not: U1, U2, U3 forward'da kullanılmıyor, bu yüzden kaldırıldı
        self.be = nn.Parameter(torch.FloatTensor(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_timesteps, num_timesteps))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (Batch, N, F, T)
        
        Returns:
            E: Temporal attention matrix (Batch, T, T)
        """
        # x shape: (Batch, N, F, T)
        batch_size, num_nodes, num_features, num_timesteps = x.shape
        
        # Basitleştirilmiş Temporal Attention (ASTGCN benzeri)
        # Dot-Product Attention: Q = K = V = x
        
        # (B, N, F, T) reshape -> (B, N*F, T)
        x_reshaped = x.reshape(batch_size, -1, num_timesteps)
        
        # Self-attention score: (B, T, T)
        # (B, T, N*F) @ (B, N*F, T) -> (B, T, T)
        score = torch.matmul(x_reshaped.transpose(1, 2), x_reshaped)
        score = torch.sigmoid(score)
        
        # Learnable transformation ile birleştir
        # E = Ve @ score + be
        E = torch.matmul(self.Ve, score) + self.be  # (B, T, T)
        
        # Softmax normalization
        E = F.softmax(E, dim=-1)
        
        return E


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism (Makale Section 3.3)
    
    Amacı: Farklı node'lar (segment'ler) arasındaki bağımlılıkları öğrenmek.
    Çıktı: Attention matrix S (N, N) - Her node'un diğerleriyle ilişkisi
    
    Formül (Basitleştirilmiş):
        S = softmax(Vs * sigmoid(score(X)) + bs)
    
    Args:
        num_nodes: Node sayısı (N)
        in_channels: Feature boyutu (F)
        num_timesteps: Zaman adımı sayısı (T)
    """
    
    def __init__(self, num_nodes: int, in_channels: int, num_timesteps: int):
        super(SpatialAttention, self).__init__()
        
        # Learnable parameters (Sadece kullanılanlar)
        # Not: W1, W2, W3 forward'da kullanılmıyor, bu yüzden kaldırıldı
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (Batch, N, F, T)
        
        Returns:
            S: Spatial attention matrix (Batch, N, N)
        """
        # x shape: (Batch, N, F, T)
        batch_size, num_nodes, num_features, num_timesteps = x.shape
        
        # Basitleştirilmiş Spatial Attention
        # (B, N, F*T) - Flatten time and feature dimensions
        x_reshaped = x.reshape(batch_size, num_nodes, -1)
        
        # Correlation matrix: (B, N, N)
        # (B, N, F*T) @ (B, F*T, N) -> (B, N, N)
        score = torch.matmul(x_reshaped, x_reshaped.transpose(1, 2))
        score = torch.sigmoid(score)
        
        # Learnable transformation
        # S = Vs @ score + bs
        S = torch.matmul(self.Vs, score) + self.bs  # (B, N, N)
        
        # Softmax normalization
        S = F.softmax(S, dim=-1)
        
        return S


class STBlock(nn.Module):
    """
    Spatiotemporal Encoding Block (Makale Fig. 1c)
    
    Akış:
        1. Temporal Attention -> X_TA
        2. Spatial Attention -> X_SA
        3. Chebyshev Graph Convolution -> X_GCN
        4. Residual Connection + ReLU
    
    Bu blok, hem zaman hem de mekân boyutlarında bilgi akışını sağlar.
    
    Args:
        num_nodes: Node sayısı (N)
        in_channels: Giriş feature boyutu
        out_channels: Çıkış feature boyutu
        num_timesteps: Zaman adımı sayısı (T)
        k_order: Chebyshev polynomial sırası (default: 3)
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        num_timesteps: int,
        k_order: int = 3
    ):
        super(STBlock, self).__init__()
        
        self.temporal_att = TemporalAttention(num_nodes, in_channels, num_timesteps)
        self.spatial_att = SpatialAttention(num_nodes, in_channels, num_timesteps)
        
        # Chebyshev Graph Convolution (PyG implementation)
        # ChebConv: (N, Fin) -> (N, Fout)
        self.cheb_conv = ChebConv(in_channels, out_channels, K=k_order)
        
        # Residual connection için boyut eşitleme (gerekirse)
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1)  # 1x1 convolution
        ) if in_channels != out_channels else nn.Identity()
        
        # Layer normalization (stabilite için)
        self.ln = nn.LayerNorm(out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (Batch, N, F_in, T)
            edge_index: Graf edge'leri (2, E)
            edge_weight: Edge ağırlıkları (E,) [Opsiyonel]
        
        Returns:
            out: Output tensor (Batch, N, F_out, T)
        """
        batch_size, num_nodes, in_channels, num_timesteps = x.shape
        
        # 1. Temporal Attention
        # E: (B, T, T) - Zaman adımları arası ilişki matrisi
        T_att = self.temporal_att(x)
        
        # x'i T_att ile çarp
        # (B, N, F, T) @ (B, T, T) -> (B, N, F, T)
        # Reshape: (B, N*F, T) @ (B, T, T) -> (B, N*F, T)
        x_flat = x.reshape(batch_size, -1, num_timesteps)
        x_TA = torch.matmul(x_flat, T_att)  # (B, N*F, T)
        x_TA = x_TA.reshape(batch_size, num_nodes, in_channels, num_timesteps)
        
        # 2. Spatial Attention
        # S: (B, N, N) - Node'lar arası ilişki matrisi
        S_att = self.spatial_att(x_TA)
        
        # x_TA'yı S_att ile çarp
        # (B, N, N) @ (B, N, F*T) -> (B, N, F*T)
        x_TA_flat = x_TA.reshape(batch_size, num_nodes, -1)
        x_SA = torch.matmul(S_att, x_TA_flat)  # (B, N, F*T)
        x_SA = x_SA.reshape(batch_size, num_nodes, in_channels, num_timesteps)
        
        # 3. Chebyshev Graph Convolution
        # ChebConv her zaman adımı için ayrı ayrı uygulanır
        # (B, N, F, T) -> (B*T, N, F) -> ChebConv -> (B*T, N, F_out) -> (B, N, F_out, T)
        
        # Permute: (B, N, F, T) -> (B, T, N, F)
        x_graph = x_SA.permute(0, 3, 1, 2)
        
        # Reshape: (B, T, N, F) -> (B*T, N, F)
        x_graph_flat = x_graph.reshape(-1, num_nodes, in_channels)
        
        # GCN Operation (batched)
        # Not: edge_index tüm batch için aynı (static graph)
        out_list = []
        for t_step in range(batch_size * num_timesteps):
            out_t = self.cheb_conv(x_graph_flat[t_step], edge_index, edge_weight)
            out_list.append(out_t)
        
        x_gcn = torch.stack(out_list)  # (B*T, N, F_out)
        
        # Reshape back: (B*T, N, F_out) -> (B, T, N, F_out) -> (B, N, F_out, T)
        out_channels = x_gcn.shape[-1]
        x_gcn = x_gcn.reshape(batch_size, num_timesteps, num_nodes, out_channels)
        x_gcn = x_gcn.permute(0, 2, 3, 1)  # (B, N, F_out, T)
        
        # 4. Residual Connection
        # (B, N, F_in, T) -> (B, F_in, N, T) -> Conv2d -> (B, F_out, N, T) -> (B, N, F_out, T)
        x_res = self.residual_conv(x.permute(0, 2, 1, 3))  # (B, F_out, N, T)
        x_res = x_res.permute(0, 2, 1, 3)  # (B, N, F_out, T)
        
        # Element-wise addition + ReLU
        out = F.relu(x_gcn + x_res)
        
        return out


class STGMS(nn.Module):
    """
    STGMS Ana Model
    
    Multi-scale decomposed features'ları alıp spatio-temporal encoding yapar.
    
    Mimari:
        Input (B, T_in, N, F_decomposed)
        -> Permute (B, N, F_decomposed, T_in)
        -> ST Block 1 (64 channels)
        -> Dropout
        -> ST Block 2 (64 channels)
        -> Dropout
        -> Conv2d (Time dimension'ı 1'e indir)
        -> FC (Horizon prediction)
        -> Output (B, T_out, N, F_out)
    
    Args:
        num_nodes: Node sayısı (N)
        in_channels: Giriş feature boyutu (F_decomposed)
        out_channels: Çıkış feature boyutu (F_original)
        window_size: Geçmiş pencere boyutu (T_in)
        horizon: Tahmin horizon'u (T_out)
        k_order: Chebyshev polynomial sırası
        dropout: Dropout oranı
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        window_size: int,
        horizon: int,
        k_order: int = 3,
        dropout: float = 0.5
    ):
        super(STGMS, self).__init__()
        
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.horizon = horizon
        self.dropout = dropout
        
        # Encoder: Stack of STBlocks
        # Makalede genellikle 2-3 layer kullanılır
        self.block1 = STBlock(num_nodes, in_channels, 64, window_size, k_order)
        self.block2 = STBlock(num_nodes, 64, 64, window_size, k_order)
        
        # Decoder: Temporal Convolution + FC
        # Conv2d ile zaman boyutunu 1'e indir
        # (B, 64, N, T) -> (B, 128, N, 1)
        self.end_conv = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, window_size)  # (Height, Width) = (1, T)
        )
        
        # Final FC: Her node için horizon adım tahmin et
        # (B, N, 128) -> (B, N, Horizon * F_out)
        self.fc = nn.Linear(128, horizon * out_channels)
        
        self.out_channels = out_channels
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (Batch, T_in, N, F_decomposed)
               Dataset'ten gelen format
            edge_index: Graf edge'leri (2, E)
            edge_weight: Edge ağırlıkları (E,)
        
        Returns:
            out: Prediction tensor (Batch, T_out, N, F_out)
        """
        # Model (Batch, N, F, T) formatını seviyor
        # (B, T_in, N, F) -> (B, N, F, T_in)
        x = x.permute(0, 2, 3, 1)
        
        # ST Block 1
        x = self.block1(x, edge_index, edge_weight)  # (B, N, 64, T)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # ST Block 2
        x = self.block2(x, edge_index, edge_weight)  # (B, N, 64, T)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Output Layer
        # (B, N, 64, T) -> (B, 64, N, T) [Conv2d için permute]
        x = x.permute(0, 2, 1, 3)
        
        # Conv2d: Zaman boyutunu 1'e indir
        # (B, 64, N, T) -> (B, 128, N, 1)
        x = self.end_conv(x)
        
        # Squeeze ve permute: (B, 128, N, 1) -> (B, N, 128)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        # Final Prediction: (B, N, 128) -> (B, N, Horizon * F_out)
        out = self.fc(x)
        
        # Reshape: (B, N, Horizon * F_out) -> (B, N, Horizon, F_out)
        out = out.reshape(-1, self.num_nodes, self.horizon, self.out_channels)
        
        # Permute: (B, N, Horizon, F_out) -> (B, Horizon, N, F_out)
        out = out.permute(0, 2, 1, 3)
        
        return out


def count_parameters(model: nn.Module) -> int:
    """Model parametre sayısını hesapla"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test fonksiyonu
def test_stgms_model():
    """STGMS model testi"""
    print("\n" + "="*70)
    print("🧪 STGMS Model Test")
    print("="*70 + "\n")
    
    # Parametreler
    batch_size = 4
    num_nodes = 50
    in_channels = 32  # Decomposed features (örn: 8 * 4)
    out_channels = 8  # Original features
    window_size = 12
    horizon = 3
    num_edges = 200
    
    # Dummy data
    x = torch.randn(batch_size, window_size, num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges)
    
    # Model
    model = STGMS(
        num_nodes=num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        window_size=window_size,
        horizon=horizon,
        k_order=3,
        dropout=0.5
    )
    
    print(f"📊 Model İstatistikleri:")
    print(f"  - Parametre sayısı: {count_parameters(model):,}")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Edge count: {num_edges}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_weight)
    
    print(f"\n🔍 Forward Pass:")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected: ({batch_size}, {horizon}, {num_nodes}, {out_channels})")
    
    # Shape kontrolü
    assert output.shape == (batch_size, horizon, num_nodes, out_channels), \
        f"Output shape mismatch! Got {output.shape}"
    
    print(f"\n✅ Test başarılı!")
    print(f"  - Output min: {output.min():.4f}")
    print(f"  - Output max: {output.max():.4f}")
    print(f"  - Output mean: {output.mean():.4f}\n")


if __name__ == "__main__":
    test_stgms_model()
