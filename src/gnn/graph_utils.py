#!/usr/bin/env python3
"""
graph_utils.py
--------------
Graf yardımcı fonksiyonları - STA-GCN için adjacency matrix ve Laplacian hesaplama

STA-GCN modeli için gerekli graf işlemleri:
- Adjacency matrix (komşuluk matrisi)
- Normalized Laplacian (normalize Laplacian matrisi)
- Chebyshev polynomial basis (spektral filtreleme için)

Teori:
    STA-GCN spektral graf konvolüsyonu kullanır:
    g_θ * x = Σ θ_k T_k(L̃) x
    
    T_k: Chebyshev polinomları
    L̃: Normalize Laplacian = 2L/λ_max - I
    L: Laplacian = D - A (D: derece matrisi, A: komşuluk matrisi)

Referans:
    - STA-GCN paper: https://arxiv.org/abs/1709.04875
    - Kipf & Welling GCN: https://arxiv.org/abs/1609.02907
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Optional


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None
) -> sp.csr_matrix:
    """
    Edge index'ten adjacency matrix oluştur (sparse format)
    
    Args:
        edge_index: (2, E) edge listesi
        num_nodes: Node sayısı
        edge_weight: (E,) edge ağırlıkları (None ise 1.0)
    
    Returns:
        Adjacency matrix (N, N) sparse CSR format
    """
    edge_index = edge_index.cpu().numpy()
    
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1])
    else:
        edge_weight = edge_weight.cpu().numpy().flatten()
    
    # Sparse matrix oluştur
    adj = sp.coo_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    
    # Symmetrize (undirected graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return adj.tocsr()


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """
    Adjacency matrix'i normalize et: D^(-1/2) A D^(-1/2)
    
    Bu, GCN'de kullanılan symmetric normalization'dır.
    
    Args:
        adj: (N, N) adjacency matrix
    
    Returns:
        Normalized adjacency matrix (N, N)
    """
    # Self-loop ekle (A + I)
    adj = adj + sp.eye(adj.shape[0])
    
    # Derece matrisi D
    rowsum = np.array(adj.sum(1)).flatten()
    
    # D^(-1/2)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # D^(-1/2) A D^(-1/2)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()


def calculate_laplacian(adj: sp.csr_matrix) -> sp.csr_matrix:
    """
    Laplacian matrix hesapla: L = D - A
    
    Args:
        adj: (N, N) adjacency matrix
    
    Returns:
        Laplacian matrix (N, N)
    """
    # Derece matrisi D
    degree = np.array(adj.sum(1)).flatten()
    degree_matrix = sp.diags(degree)
    
    # L = D - A
    laplacian = degree_matrix - adj
    
    return laplacian.tocsr()


def normalize_laplacian(laplacian: sp.csr_matrix) -> sp.csr_matrix:
    """
    Laplacian'ı normalize et: L̃ = 2L/λ_max - I
    
    λ_max: En büyük eigenvalue (yaklaşık olarak 2 alınır)
    
    Args:
        laplacian: (N, N) Laplacian matrix
    
    Returns:
        Normalized Laplacian (N, N)
    """
    # λ_max yaklaşık 2 (genelde yeterli)
    lambda_max = 2.0
    
    # L̃ = 2L/λ_max - I
    n = laplacian.shape[0]
    normalized_laplacian = (2.0 / lambda_max) * laplacian - sp.eye(n)
    
    return normalized_laplacian.tocsr()


def chebyshev_polynomials(
    laplacian: sp.csr_matrix,
    k_order: int
) -> list:
    """
    Chebyshev polinomları hesapla: T_0, T_1, ..., T_k
    
    Recursive formula:
        T_0(x) = 1
        T_1(x) = x
        T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    
    Args:
        laplacian: (N, N) normalized Laplacian
        k_order: Polinom derecesi (K)
    
    Returns:
        List of k+1 matrices: [T_0(L̃), T_1(L̃), ..., T_k(L̃)]
    """
    n = laplacian.shape[0]
    
    # T_0 = I (identity)
    T_k = [sp.eye(n, format='csr')]
    
    # T_1 = L̃
    T_k.append(laplacian)
    
    # T_k = 2 L̃ T_{k-1} - T_{k-2}
    for i in range(2, k_order + 1):
        T_new = 2 * laplacian.dot(T_k[-1]) - T_k[-2]
        T_k.append(T_new)
    
    return T_k


def sparse_to_torch(sparse_mx: sp.csr_matrix) -> torch.Tensor:
    """
    Scipy sparse matrix → PyTorch sparse tensor
    
    Args:
        sparse_mx: Scipy sparse matrix
    
    Returns:
        PyTorch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo()
    
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)


def precompute_cheb_basis(
    edge_index: torch.Tensor,
    num_nodes: int,
    k_order: int = 3,
    edge_weight: Optional[torch.Tensor] = None
) -> list:
    """
    Chebyshev basis'i önceden hesapla (model training'den önce)
    
    Bu fonksiyon STA-GCN modeli için gerekli tüm matrisleri hazırlar.
    
    Args:
        edge_index: (2, E) edge listesi
        num_nodes: Node sayısı
        k_order: Chebyshev polinom derecesi (default: 3)
        edge_weight: (E,) edge ağırlıkları
    
    Returns:
        List of PyTorch sparse tensors: [T_0, T_1, ..., T_k]
    """
    print(f"🔄 Chebyshev basis hesaplanıyor (K={k_order})...")
    
    # 1. Adjacency matrix
    adj = edge_index_to_adjacency(edge_index, num_nodes, edge_weight)
    print(f"  ✓ Adjacency matrix: {adj.shape}")
    
    # 2. Laplacian
    laplacian = calculate_laplacian(adj)
    print(f"  ✓ Laplacian: {laplacian.shape}")
    
    # 3. Normalize Laplacian
    norm_laplacian = normalize_laplacian(laplacian)
    print(f"  ✓ Normalized Laplacian: {norm_laplacian.shape}")
    
    # 4. Chebyshev polynomials
    cheb_polynomials = chebyshev_polynomials(norm_laplacian, k_order)
    print(f"  ✓ {len(cheb_polynomials)} Chebyshev polinomları hesaplandı")
    
    # 5. Convert to PyTorch
    cheb_basis = [sparse_to_torch(T_k) for T_k in cheb_polynomials]
    
    print(f"✅ Chebyshev basis hazır!")
    
    return cheb_basis


def build_adjacency_from_distance(
    edge_index: torch.Tensor,
    edge_distance: torch.Tensor,
    sigma: float = 10.0
) -> torch.Tensor:
    """
    Mesafe bazlı adjacency weight hesapla (Gaussian kernel)
    
    w_ij = exp(-d_ij^2 / σ^2)
    
    Yakın segment'ler yüksek ağırlık alır.
    
    Args:
        edge_index: (2, E) edge listesi
        edge_distance: (E,) mesafeler (metre)
        sigma: Gaussian kernel genişliği (default: 10m)
    
    Returns:
        Edge weights (E,)
    """
    # Gaussian kernel: exp(-d^2 / σ^2)
    weights = torch.exp(-edge_distance.pow(2) / (sigma ** 2))
    
    return weights


def get_neighbor_counts(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Her node'un komşu sayısını hesapla (derece)
    
    Args:
        edge_index: (2, E)
        num_nodes: Node sayısı
    
    Returns:
        Degree tensor (N,)
    """
    degree = torch.zeros(num_nodes, dtype=torch.long)
    
    for i in range(num_nodes):
        degree[i] = (edge_index[0] == i).sum() + (edge_index[1] == i).sum()
    
    return degree


def compute_graph_statistics(
    edge_index: torch.Tensor,
    num_nodes: int
) -> dict:
    """
    Graf istatistikleri hesapla (debug/analysis için)
    
    Returns:
        {
            'num_nodes': int,
            'num_edges': int,
            'avg_degree': float,
            'max_degree': int,
            'min_degree': int,
            'density': float
        }
    """
    num_edges = edge_index.shape[1]
    degree = get_neighbor_counts(edge_index, num_nodes)
    
    # İstatistikler
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': degree.float().mean().item(),
        'max_degree': degree.max().item(),
        'min_degree': degree.min().item(),
        'density': num_edges / (num_nodes * (num_nodes - 1))  # yönlü graf için
    }
    
    return stats


# Test fonksiyonu
def test_graph_utils():
    """Graph utils test"""
    print("\n" + "="*70)
    print("🧪 Graph Utils Test")
    print("="*70 + "\n")
    
    # Basit graf oluştur
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    num_nodes = 4
    
    print(f"📊 Test Graf:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Edges: {edge_index.shape[1]}")
    
    # İstatistikler
    stats = compute_graph_statistics(edge_index, num_nodes)
    print(f"\n📈 Graf İstatistikleri:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Chebyshev basis
    print(f"\n🔄 Chebyshev Basis Test:")
    cheb_basis = precompute_cheb_basis(edge_index, num_nodes, k_order=3)
    
    for i, T_k in enumerate(cheb_basis):
        print(f"  - T_{i}: {T_k.shape} (sparse)")
    
    print("\n✅ Test tamamlandı!\n")


if __name__ == "__main__":
    test_graph_utils()
