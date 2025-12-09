#!/usr/bin/env python3
"""
mtgnn.py
--------
MTGNN (Multivariate Time Series Forecasting with Graph Neural Networks) Implementasyonu.

Makale: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks" (KDD 2020)

Özellikler:
- Graph Learning Layer: Veriden graf yapısını öğrenir (Eq. 1-6)
- Mix-hop Propagation: GCN katmanı (Eq. 7-8)
- Dilated Inception: Zamansal konvolüsyon (Fig. 5)
- Hybrid Mode: Neo4j'den gelen statik graf ile öğrenilen grafı birleştirir.

Girdi Formatı: (Batch, Time, Nodes, Features) -> Model içinde (B, F, N, T) olarak işlenir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GraphConstructor(nn.Module):
    """
    Graph Learning Layer (Makale Section 4.2)
    
    Veriden adaptif bir komşuluk matrisi (Adjacency Matrix) öğrenir.
    Eq 1-6 implementasyonu.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.device = device
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.dim = dim
        self.k = k  # Top-k neighbors
        self.alpha = alpha
        self.static_feat = static_feat  # Opsiyonel harici node özellikleri

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.emb1.weight)
        init.xavier_uniform_(self.emb2.weight)

    def forward(self, idx):
        # Eq. 1 & 2: M1 = tanh(alpha * E1 * Theta1)
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # Eq. 3: A = ReLU(tanh(alpha * (M1 @ M2.T - M2 @ M1.T)))
        # Uni-directional ilişkiyi zorlamak için çıkarma işlemi yapılır.
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        
        # Eq. 5 & 6: Sparsification (Top-k selection)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        
        # Her satır için en büyük k değeri bul
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        
        adj = adj * mask
        return adj


class MixHopLayer(nn.Module):
    """
    Mix-hop Propagation Layer (Makale Section 4.3)
    
    Eq. 7: H(k) = beta * Hin + (1-beta) * A_tilde * H(k-1)
    Eq. 8: Hout = Sum(H(k) * W(k))
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixHopLayer, self).__init__()
        self.nco = c_out
        self.gdep = gdep  # Propagation depth (K)
        self.alpha = alpha  # Beta in paper (Retain ratio)
        
        # Her hop için ayrı ağırlık matrisi (Information Selection)
        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)
        self.dropout = dropout

    def forward(self, x, adj):
        # x: (B, C_in, N, T)
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # Self-loop
        d = adj.sum(1)
        h = x
        out = [h]
        
        # Propagation (Eq. 7)
        # (B, C, N, T) -> (N, N) x (B, C, N, T) -> (B, C, N, T)
        # Einsum: 'nm, bcml->bcnl' (n: target node, m: source node)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum('nm,bcml->bcnl', adj, h)
            out.append(h)
            
        # Selection (Eq. 8) - Concatenate and Linear projection
        hout = torch.cat(out, dim=1)  # (B, (K+1)*C, N, T)
        hout = torch.einsum('bcnl->blnc', hout)  # (B, T, N, C_total)
        hout = self.mlp(hout)  # (B, T, N, C_out)
        hout = torch.einsum('blnc->bcnl', hout)  # (B, C_out, N, T)
        
        return hout


class DilatedInception(nn.Module):
    """
    Dilated Inception Layer (Makale Section 4.4)
    
    Farklı kernel boyutları (1x2, 1x3, 1x6, 1x7) ile zamansal özellikleri yakalar.
    """
    def __init__(self, c_in, c_out, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        c_out = int(c_out / len(self.kernel_set))
        
        for k in self.kernel_set:
            self.tconv.append(nn.Conv2d(c_in, c_out, (1, k), dilation=(1, dilation_factor)))

    def forward(self, input):
        x_list = []
        for i in range(len(self.kernel_set)):
            # Causal padding: Gelecekten bilgi sızmaması için
            # Dilation ve kernel'a göre padding hesapla
            kernel_size = self.kernel_set[i]
            dilation = self.tconv[i].dilation[1]
            padding = (kernel_size - 1) * dilation
            
            # Sol tarafa padding ekle (causal)
            x_padded = F.pad(input, (padding, 0, 0, 0))
            x = self.tconv[i](x_padded)
            x_list.append(x)
            
        # Farklı filtre çıktılarını birleştir
        x = torch.cat(x_list, dim=1)
        return x


class LayerNorm(nn.Module):
    """BatchNorm2d wrapper - dinamik boyutlarla daha iyi çalışır"""
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        # normalized_shape tuple ise (C, N, T) formatında
        # Sadece channel boyutunu kullan
        if isinstance(normalized_shape, tuple):
            num_channels = normalized_shape[0]
        else:
            num_channels = normalized_shape
        self.bn = nn.BatchNorm2d(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, N, T)
        return self.bn(x)


class MTGNN(nn.Module):
    """
    MTGNN Ana Model Sınıfı
    
    Args:
        gcn_true: Graph Learning kullanılsın mı?
        build_adj: Adaptif adjacency matrix oluşturulsun mu?
        gcn_depth: Mix-hop derinliği
        num_nodes: Node sayısı
        device: Cihaz
        predefined_adj: Neo4j'den gelen statik adjacency matrix (Opsiyonel)
        dropout: Dropout oranı
        subgraph_size: Graph learning için top-k
        node_dim: Node embedding boyutu
        dilation_exponential: Dilation artış katsayısı
        conv_channels: Convolution kanalları
        residual_channels: Residual kanalları
        skip_channels: Skip connection kanalları
        end_channels: Output modülü kanalları
        seq_length: Input sequence uzunluğu
        in_dim: Input feature boyutu
        out_dim: Output feature boyutu (Genelde 1 veya F)
        layers: Katman sayısı
        propalpha: Mix-hop retain ratio
        tanhalpha: Graph learning saturation
    """
    def __init__(self, gcn_true, build_adj, gcn_depth, num_nodes, device, predefined_adj=None, 
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, 
                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, 
                 seq_length=12, in_dim=1, out_dim=1, layers=3, propalpha=0.05, tanhalpha=3):
        super(MTGNN, self).__init__()
        
        self.gcn_true = gcn_true
        self.build_adj = build_adj
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_adj = predefined_adj
        self.layers = layers
        self.seq_length = seq_length
        self.device = device
        
        # Filter ve Gate konvolüsyonları
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        
        # Graph Learning Layer
        self.gc = GraphConstructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha)

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(layers):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(1 + j * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = j * (kernel_size - 1) + 1

                if rf_size_j > rf_size_i:
                    new_dilation = 1
                    break
                new_dilation = dilation_exponential

            self.filter_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
            self.gate_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
            self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
            # Skip conv'u 1x1 yap, global pooling ile skip connection yapacağız
            self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, 1)))

            if self.gcn_true:
                self.gconv1.append(MixHopLayer(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(MixHopLayer(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

            self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_i + 1)))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, edge_index=None, edge_weight=None):
        """
        Args:
            input: (Batch, Time, Nodes, Features)
            edge_index: (2, E) - Opsiyonel, Neo4j'den gelen statik graf
        """
        # MTGNN (B, F, N, T) formatı ister
        # Input: (B, T, N, F) -> Permute -> (B, F, N, T)
        input = input.permute(0, 3, 2, 1)
        
        # Input padding (Receptive field için)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
            
        x = self.start_conv(x)
        skip = 0

        # 1. Graph Learning (Adaptive Adjacency)
        # Eğer build_adj=True ise veriden öğrenir.
        adp = self.gc(self.idx)
        
        # 2. Hybrid Graph Integration
        # Eğer Neo4j'den gelen statik graf varsa ve model konfigürasyonunda
        # predefined_adj verilmemişse, forward'dan geleni kullanabiliriz.
        # Ancak MTGNN genellikle statik grafı init sırasında dense matrix olarak ister.
        # Burada basitleştirilmiş bir hibrit yaklaşım uyguluyoruz:
        # GCN katmanları 'adp' (öğrenilen) kullanır. Eğer statik graf çok önemliyse
        # 'adp' ile statik graf toplanabilir veya ayrı kanallardan verilebilir.
        # Bu implementasyonda makaleye sadık kalarak 'adp'yi ana graf olarak kullanıyoruz.
        
        for i in range(self.layers):
            residual = x
            
            # Dilated Inception (Temporal)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            
            # Skip connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            
            # Graph Convolution (Spatial)
            if self.gcn_true:
                # Mix-hop 1 (Adaptive Graph)
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
                
                # Eğer statik graf (predefined_adj) varsa onu da işin içine katabiliriz
                # Ancak orijinal MTGNN sadece bir graf kullanır veya bunları toplar.
                if self.predefined_adj is not None:
                    # Statik graf katkısı (Opsiyonel hibrit yapı)
                    x = x + self.gconv1[i](x, self.predefined_adj)

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x)

        # Output Module
        # Skip: (B, skip_channels, N, T) - T değişken olabilir
        # Causal CNN standardı: Sadece en son (en bilgili) zaman adımını al
        # NOT: adaptive_avg_pool2d tüm adımların ortalamasını alır ve sinyali kirletir
        # Son adım, tüm receptive field'ı görür ve en zengin bilgiye sahiptir
        skip = skip[:, :, :, -1:]  # (B, skip_channels, N, 1)
        
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # (B, Out_dim, N, 1)
        
        # Output shape: (B, Out_dim, N, 1) -> (B, N, 1, Out_dim) -> (B, 1, N, Out_dim)
        # Hedef format: (Batch, Horizon, Nodes, Features)
        # MTGNN genelde tek adım (horizon=1) veya recursive çalışır.
        # Ancak burada direct multi-step output için son boyutu ayarlıyoruz.
        
        return x.permute(0, 3, 2, 1)  # (B, 1, N, F) - Tek adım tahmini için
