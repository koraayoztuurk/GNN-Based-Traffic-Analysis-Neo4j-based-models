# GNN Modülleri

Bu klasör, trafik tahmini için Graph Neural Network (GNN) modellerini içerir.

## 📁 Dosya Yapısı

```
gnn/
├── __init__.py                     # Modül export'ları
├── README.md                       # Bu dosya
├── README_MODELS.md                # Model detayları
├── README_STGMS.md                 # STGMS özellikleri
│
├── dataset_sta.py                  # STA-GCN için PyTorch Dataset
├── dataset_stgms.py                # STGMS için Dataset (multi-timescale)
├── graph_utils.py                  # Graf işleme fonksiyonları
├── evaluate.py                     # Model değerlendirme scripti
│
├── models/                         # Model implementasyonları
│   ├── __init__.py
│   ├── sta_gcn.py                 # STA-GCN modeli
│   └── stgms.py                   # STGMS modeli
│
├── trainers/                       # Training scriptleri
│   ├── __init__.py
│   ├── train_sta_gcn.py           # STA-GCN eğitimi
│   ├── train_stgms.py             # STGMS eğitimi
│   └── incremental_train_sta_gcn.py  # Artımlı eğitim
│
└── evaluators/                     # Evaluation scriptleri
    ├── __init__.py
    └── evaluate_sta_gcn.py        # STA-GCN değerlendirmesi
```

## 🎯 Modeller

### 1. STA-GCN (Spatio-Temporal Attention GCN)
- **Dataset**: `dataset_sta.py` → `TrafficDataset`
- **Model**: `models/sta_gcn.py` → `STAGCN`
- **Trainer**: `trainers/train_sta_gcn.py`
- **Özellikler**:
  - Spatial Graph Convolution (Chebyshev)
  - Temporal Gated CNN
  - Attention mekanizması

### 2. STGMS (Multi-timeScale Graph Neural Network)
- **Dataset**: `dataset_stgms.py` → `STGMSDataset`
- **Model**: `models/stgms.py` → `STGMS`
- **Trainer**: `trainers/train_stgms.py`
- **Özellikler**:
  - Multi-timescale feature decomposition
  - Temporal ve Spatial Attention
  - Online decomposition (causal padding)

## 🚀 Kullanım

### STA-GCN Eğitimi
```bash
cd src/gnn
python trainers/train_sta_gcn.py --epochs 100 --batch_size 32
```

### STGMS Eğitimi
```bash
cd src/gnn
python trainers/train_stgms.py --epochs 100 --periods 96 16 4
```

### Model Değerlendirme
```bash
python evaluate.py --model_path outputs/models/best_model.pt
```

## 📦 Import Kullanımı

```python
# Ana modülden import
from src.gnn import TrafficDataset, STGMSDataset
from src.gnn import STAGCN, STGMS
from src.gnn import precompute_cheb_basis

# Doğrudan import
from src.gnn.dataset_sta import TrafficDataset
from src.gnn.dataset_stgms import STGMSDataset
from src.gnn.models.sta_gcn import STAGCN
from src.gnn.models.stgms import STGMS
```

## 🔧 Graf Utilities

`graph_utils.py` şunları sağlar:
- Chebyshev polynomial basis hesaplama
- Adjacency matrix işlemleri
- Laplacian normalizasyonu
- Graf istatistikleri

## 📊 Veri Akışı

1. **Neo4j** → Segment ve Measure verileri
2. **Dataset** → PyTorch tensörlerine çevirme
3. **Model** → Tahmin üretme
4. **Evaluation** → Metrik hesaplama

## 🏗️ Mimari Notlar

- Her iki model de **Neo4j'den veri çeker**
- **Chebyshev graph convolution** kullanır
- **Temporal ve spatial** bağımlılıkları öğrenir
- **Early stopping** ve checkpoint desteği vardır

## 📝 Önemli Dosyalar

- `dataset_sta.py`: STA-GCN için standart zaman serisi yüklemesi
- `dataset_stgms.py`: STGMS için multi-timescale ayrıştırma
- `graph_utils.py`: Tüm modeller için ortak graf işlemleri
- `evaluate.py`: Test metrikleri ve görselleştirme

---

**Son Güncelleme**: 2025-12-04  
**Düzenleme**: Profesyonel dosyalama ve organizasyon
