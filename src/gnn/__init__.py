"""
src/gnn/__init__.py
-------------------
GNN modülü - STA-GCN, STGMS ve yardımcı fonksiyonlar

Exports:
    - TrafficDataset: PyTorch dataset for STA-GCN
    - STGMSDataset: PyTorch dataset for STGMS (with multi-timescale decomposition)
    - STAGCN: STA-GCN model
    - STGMS: STGMS model
    - ChebConv: Chebyshev graph convolution layer
    - precompute_cheb_basis: Chebyshev basis hesaplama
"""

# Datasets
from .dataset_sta import TrafficDataset
from .dataset_stgms import STGMSDataset

# Models
from .models.sta_gcn import (
    STAGCN,
    ChebConv,
    TemporalConvLayer,
    STConvBlock,
    count_parameters
)
from .models.stgms import (
    STGMS,
    TemporalAttention,
    SpatialAttention,
    STBlock
)

# Graph utilities
from .graph_utils import (
    precompute_cheb_basis,
    edge_index_to_adjacency,
    normalize_adjacency,
    calculate_laplacian,
    normalize_laplacian,
    chebyshev_polynomials,
    sparse_to_torch,
    build_adjacency_from_distance,
    get_neighbor_counts,
    compute_graph_statistics
)

__all__ = [
    # Datasets
    'TrafficDataset',
    'STGMSDataset',
    
    # Models
    'STAGCN',
    'STGMS',
    'ChebConv',
    'TemporalConvLayer',
    'STConvBlock',
    'TemporalAttention',
    'SpatialAttention',
    'STBlock',
    'count_parameters',
    
    # Graph utilities
    'precompute_cheb_basis',
    'edge_index_to_adjacency',
    'normalize_adjacency',
    'calculate_laplacian',
    'normalize_laplacian',
    'chebyshev_polynomials',
    'sparse_to_torch',
    'build_adjacency_from_distance',
    'get_neighbor_counts',
    'compute_graph_statistics',
]
