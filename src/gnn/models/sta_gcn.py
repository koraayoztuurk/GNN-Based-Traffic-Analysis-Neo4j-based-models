#!/usr/bin/env python3
"""
model_sta_gcn.py
----------------
Spatio-Temporal Attention Graph Convolutional Network (STA-GCN)

Traffic speed prediction için STA-GCN modeli:
- Spatial Graph Convolution (spektral filtre + Chebyshev approx)
- Temporal Gated CNN (1D konvolüsyon + GLU)
- Attention mekanizması (spatial + temporal)

Mimari:
    Input (T, N, F) → [ST-Conv Block × L] → Output Layer → (T', N, F')
    
    ST-Conv Block:
        1. Temporal Conv (gated)
        2. Spatial Graph Conv (Chebyshev)
        3. Temporal Conv (gated)
        4. Layer Norm + Residual

Referans:
    - STA-GCN: https://arxiv.org/abs/1709.04875
    - Graph WaveNet'ten attention: https://arxiv.org/abs/1906.00121
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class Chomp1d(nn.Module):
    """
    Causal Convolution için padding kırpma (Future information leak'i önler)
    
    Temporal Conv1d'de geleceği görmemek için:
    1. padding=(kernel_size - 1) ile left padding yap
    2. Chomp1d ile sağdan fazla padding'i kes
    
    Bu sayede t anındaki prediction sadece t ve öncesini görür.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T-chomp_size) tensor
        """
        return x[:, :, :-self.chomp_size].contiguous()


class ChebConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolution
    
    Spektral filtre: g_θ * x = Σ θ_k T_k(L̃) x
    
    Args:
        in_channels: Input feature dim
        out_channels: Output feature dim
        k_order: Chebyshev polinom derecesi
    """
    
    def __init__(self, in_channels: int, out_channels: int, k_order: int):
        super().__init__()
        self.k_order = k_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # θ parametreleri (K × in_channels × out_channels)
        self.weight = nn.Parameter(
            torch.FloatTensor(k_order + 1, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Parametreleri initialize et (Glorot uniform)"""
        stdv = 1.0 / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, cheb_basis: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass (FIXED: Doğru sparse matmul ile batch handling)
        
        Args:
            x: (B, N, F_in) input features
            cheb_basis: List of K+1 sparse tensors [T_0, T_1, ..., T_k]
        
        Returns:
            (B, N, F_out) output features
        """
        batch_size, num_nodes, in_features = x.shape
        device = x.device
        
        # Her Chebyshev polynomial için hesapla
        out = None
        for k in range(self.k_order + 1):
            T_k = cheb_basis[k].to(device)
            
            # FIX: Doğru sparse @ dense matmul
            # T_k: (N, N) sparse, x: (B, N, F_in) dense
            # Her batch için ayrı ayrı hesapla
            batch_results = []
            for b in range(batch_size):
                # T_k @ x[b]: (N, N) @ (N, F_in) → (N, F_in)
                x_b = x[b]  # (N, F_in)
                support_b = torch.sparse.mm(T_k, x_b)  # (N, F_in)
                batch_results.append(support_b)
            
            support = torch.stack(batch_results, dim=0)  # (B, N, F_in)
            
            # θ_k weight'i uygula: (B, N, F_in) @ (F_in, F_out) → (B, N, F_out)
            support = torch.einsum('bnf,fo->bno', support, self.weight[k])
            
            if out is None:
                out = support
            else:
                out = out + support
        
        # Bias ekle
        out = out + self.bias
        
        return out


class TemporalConvLayer(nn.Module):
    """
    Temporal Gated Convolutional Layer
    
    1D Conv + GLU (Gated Linear Unit) aktivasyonu:
        GLU(x) = x_a ⊙ σ(x_b)
    
    Args:
        in_channels: Input feature dim
        out_channels: Output feature dim
        kernel_size: Temporal kernel size
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # FIXED: Causal padding ile future information leak önlendi
        # Left padding (kernel_size-1), sonra Chomp1d ile sağ tarafı kırp
        self.padding = kernel_size - 1
        
        # Conv1d: (B, F_in, T) → (B, 2×F_out, T)
        # 2× çünkü GLU için split yapacağız
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            padding=self.padding  # Full left padding
        )
        
        # Causal padding için sağ tarafı kırp
        self.chomp = Chomp1d(self.padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F_in)
        
        Returns:
            (B, T, N, F_out)
        """
        batch_size, time_steps, num_nodes, in_features = x.shape
        
        # Reshape: (B, T, N, F) → (B×N, F, T)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, in_features, time_steps)
        
        # Causal Conv1d: pad left, then chomp right
        x = self.conv(x)  # (B×N, 2×F_out, T+padding)
        x = self.chomp(x)  # (B×N, 2×F_out, T) - removes future information
        
        # Split for GLU: (2×F_out) → (F_out, F_out)
        x_a, x_b = torch.chunk(x, 2, dim=1)
        
        # GLU activation: x_a ⊙ σ(x_b)
        x = x_a * torch.sigmoid(x_b)
        
        # Reshape back: (B×N, F_out, T) → (B, T, N, F_out)
        out_features = x.shape[1]
        x = x.reshape(batch_size, num_nodes, out_features, time_steps).permute(0, 3, 1, 2)
        
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism
    
    Her node'un komşularına farklı ağırlık verir.
    """
    
    def __init__(self, in_channels: int, num_nodes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        
        # Learnable attention parameters
        self.W_s = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        self.b_s = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_s)
        nn.init.constant_(self.b_s, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F)
        
        Returns:
            Attention weights (B, T, N, N)
        """
        # Spatial attention: softmax(x W_s x^T + b_s)
        # (B, T, N, F) @ (F, F) → (B, T, N, F)
        x_transformed = torch.matmul(x, self.W_s)
        
        # (B, T, N, F) @ (B, T, F, N) → (B, T, N, N)
        attention = torch.matmul(x_transformed, x.transpose(-1, -2))
        
        # Bias ekle ve normalize et
        attention = attention + self.b_s
        attention = F.softmax(attention, dim=-1)
        
        return attention


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block
    
    Temporal Conv → Spatial Graph Conv → Temporal Conv → LayerNorm + Residual
    """
    
    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        out_channels: int,
        num_nodes: int,
        k_order: int = 3,
        kernel_size: int = 3
    ):
        super().__init__()
        
        # Temporal layers
        self.temporal1 = TemporalConvLayer(in_channels, spatial_channels, kernel_size)
        self.temporal2 = TemporalConvLayer(spatial_channels, out_channels, kernel_size)
        
        # Spatial layer (Chebyshev Graph Conv)
        self.spatial = ChebConv(spatial_channels, spatial_channels, k_order)
        
        # Attention (optional, disabled by default)
        self.use_attention = False
        if self.use_attention:
            self.attention = SpatialAttention(spatial_channels, num_nodes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Residual connection (if dimensions match)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
    
    def forward(
        self,
        x: torch.Tensor,
        cheb_basis: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F_in)
            cheb_basis: Chebyshev polynomial basis
        
        Returns:
            (B, T, N, F_out)
        """
        residual = x
        
        # 1. Temporal Conv (gated)
        x = self.temporal1(x)  # (B, T, N, F_spatial)
        
        # 2. Spatial Graph Conv
        batch_size, time_steps, num_nodes, channels = x.shape
        
        # Apply spatial conv to each time step
        x_spatial = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # (B, N, F_spatial)
            x_t = self.spatial(x_t, cheb_basis)  # (B, N, F_spatial)
            x_spatial.append(x_t)
        
        x = torch.stack(x_spatial, dim=1)  # (B, T, N, F_spatial)
        
        # 3. Temporal Conv (gated)
        x = self.temporal2(x)  # (B, T, N, F_out)
        
        # 4. Layer Norm
        x = self.layer_norm(x)
        
        # 5. Residual connection
        if self.residual is not None:
            # (B, T, N, F_in) → (B, F_in, T, N) → (B, F_out, T, N) → (B, T, N, F_out)
            residual = residual.permute(0, 3, 1, 2)
            residual = self.residual(residual)
            residual = residual.permute(0, 2, 3, 1)
        
        x = x + residual
        
        return F.relu(x)


class STAGCN(nn.Module):
    """
    Spatio-Temporal Attention Graph Convolutional Network
    
    Multi-layer ST-Conv blocks ile traffic prediction.
    
    Args:
        num_nodes: Node sayısı (N)
        in_channels: Input feature dim (F)
        hidden_channels: Hidden layer dims (list)
        out_channels: Output feature dim
        k_order: Chebyshev polinom derecesi
        kernel_size: Temporal kernel size
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        k_order: int = 3,
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # ST-Conv blocks
        self.st_blocks = nn.ModuleList()
        
        channels = [in_channels] + hidden_channels
        for i in range(len(channels) - 1):
            self.st_blocks.append(
                STConvBlock(
                    in_channels=channels[i],
                    spatial_channels=hidden_channels[i],
                    out_channels=channels[i + 1],
                    num_nodes=num_nodes,
                    k_order=k_order,
                    kernel_size=kernel_size
                )
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_channels[-1], out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        cheb_basis: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, T_in, N, F_in) input sequence
            cheb_basis: Chebyshev basis matrices
        
        Returns:
            (B, T_in, N, F_out) predictions
        """
        # ST-Conv blocks
        for block in self.st_blocks:
            x = block(x, cheb_basis)
        
        # Output projection: (B, T, N, F_hidden) → (B, T, N, F_out)
        x = self.output_layer(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test fonksiyonu
def test_model():
    """Model test"""
    print("\n" + "="*70)
    print("🧪 STA-GCN Model Test")
    print("="*70 + "\n")
    
    # Test parametreleri
    batch_size = 8
    time_steps = 12
    num_nodes = 100
    in_channels = 8
    out_channels = 8
    k_order = 3
    
    print(f"📊 Test Parametreleri:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time steps: {time_steps}")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Input features: {in_channels}")
    print(f"  - Output features: {out_channels}")
    
    # Model oluştur
    model = STAGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=[64, 64, 32],
        out_channels=out_channels,
        k_order=k_order,
        kernel_size=3
    )
    
    print(f"\n🏗️  Model Mimari:")
    print(f"  - ST-Conv blocks: {len(model.st_blocks)}")
    print(f"  - Hidden channels: [64, 64, 32]")
    print(f"  - Total parameters: {count_parameters(model):,}")
    
    # Dummy Chebyshev basis oluştur
    print(f"\n🔄 Chebyshev Basis (dummy)...")
    cheb_basis = []
    for k in range(k_order + 1):
        indices = torch.LongTensor([[0, 1], [1, 0]])
        values = torch.FloatTensor([1.0, 1.0])
        cheb_basis.append(
            torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        )
    
    # Dummy input
    x = torch.randn(batch_size, time_steps, num_nodes, in_channels)
    print(f"  - Input: {x.shape}")
    
    # Forward pass
    print(f"\n🚀 Forward Pass...")
    with torch.no_grad():
        output = model(x, cheb_basis)
    
    print(f"  - Output: {output.shape}")
    
    # Validation
    assert output.shape == (batch_size, time_steps, num_nodes, out_channels)
    print(f"\n✅ Test başarılı! Shape doğru: {output.shape}")
    
    print(f"\n📈 Output İstatistikleri:")
    print(f"  - Mean: {output.mean().item():.4f}")
    print(f"  - Std: {output.std().item():.4f}")
    print(f"  - Min: {output.min().item():.4f}")
    print(f"  - Max: {output.max().item():.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_model()
