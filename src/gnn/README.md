# GNN Modülleri - Tam Entegre Sistem

Bu klasör, trafik tahmini için Graph Neural Network (GNN) modellerini içerir.
**Her model için tam training ve evaluation pipeline'ı mevcuttur.**

## 📁 Dosya Yapısı

```
gnn/
├── __init__.py                     # Modül export'ları
├── README.md                       # Bu dosya (kapsamlı dokümantasyon)
│
├── dataset_sta.py                  # STA-GCN için PyTorch Dataset
├── dataset_stgms.py                # STGMS için Dataset (multi-timescale)
├── graph_utils.py                  # Graf işleme fonksiyonları
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
    ├── evaluate_sta_gcn.py        # STA-GCN değerlendirmesi
    └── evaluate_stgms.py          # STGMS değerlendirmesi
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
- **Evaluator**: `evaluators/evaluate_stgms.py`
- **Özellikler**:
  - Multi-timescale feature decomposition (trend, orta dönem, residual)
  - Temporal ve Spatial Attention
  - Online decomposition (causal padding)
  - Feature boyutu: F × (m+1) - örn: 8 → 32 (3 periyot + 1 residual)
  - Chebyshev Graph Convolution (K=3)
  - Gaussian kernel mesafe ağırlıklandırması

## 🚀 Kullanım - Tam Workflow

### STA-GCN Modeli

#### 1. Eğitim (Training)
```bash
# Basit eğitim
python src/gnn/trainers/train_sta_gcn.py --epochs 100 --batch_size 32

# Özelleştirilmiş
python src/gnn/trainers/train_sta_gcn.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --k_order 3 \
    --device cuda

# Checkpoint'ten devam et
python src/gnn/trainers/train_sta_gcn.py \
    --resume outputs/models/checkpoint_epoch_50.pt \
    --epochs 150
```

**Eğitim Sonucu:**
- ✅ Training metrikleri (her epoch)
- ✅ Validation metrikleri
- ✅ **Test set evaluation** (otomatik)
- ✅ Best model checkpoint
- ✅ Training history (JSON)
- ✅ Test metrics (JSON)

#### 2. Detaylı Evaluation
```bash
# Tam analiz
python src/gnn/evaluators/evaluate_sta_gcn.py \
    --checkpoint outputs/models/best_model.pt \
    --compare-baselines

# Visualization ile
python src/gnn/evaluators/evaluate_sta_gcn.py \
    --checkpoint outputs/models/best_model.pt \
    --visualize
```

**Evaluation Çıktıları:**
- 📊 Test metrikleri (MSE, MAE, RMSE, MAPE, R²)
- 📈 Prediction vs Actual grafikleri
- 🔍 Segment-level analiz
- 📉 Baseline model karşılaştırması

---

### STGMS Modeli

#### 1. Eğitim (Training)
```bash
# Basit eğitim
python src/gnn/trainers/train_stgms.py --epochs 100 --batch_size 32

# Özelleştirilmiş periyotlar
python src/gnn/trainers/train_stgms.py \
    --epochs 100 \
    --batch_size 32 \
    --periods 96 16 4 \
    --lr 0.001

# GPU ile
python src/gnn/trainers/train_stgms.py \
    --epochs 100 \
    --batch_size 64 \
    --device cuda

# Fine-tuning
python src/gnn/trainers/train_stgms.py \
    --resume outputs/models/stgms/best_model_stgms.pt \
    --fine_tune \
    --epochs 50 \
    --lr 0.0001
```

**Eğitim Sonucu:**
- ✅ Training metrikleri (her epoch)
- ✅ Validation metrikleri  
- ✅ **Test set evaluation** (otomatik)
- ✅ Multi-timescale decomposition
- ✅ Best model checkpoint
- ✅ Training history (JSON)
- ✅ Test metrics (JSON)

#### 2. Detaylı Evaluation
```bash
# Temel evaluation
python src/gnn/evaluators/evaluate_stgms.py \
    --checkpoint outputs/models/stgms/best_model_stgms.pt

# Detaylı analiz
python src/gnn/evaluators/evaluate_stgms.py \
    --checkpoint outputs/models/stgms/best_model_stgms.pt \
    --detailed

# Özelleştirilmiş
python src/gnn/evaluators/evaluate_stgms.py \
    --checkpoint outputs/models/stgms/best_model_stgms.pt \
    --periods 96 16 4 \
    --batch_size 16 \
    --detailed
```

**Evaluation Çıktıları:**
- 📊 Test metrikleri (MSE, MAE, RMSE, MAPE, R²)
- 📈 Prediction vs Actual grafikleri
- 🔍 Segment-level analiz
- 🕒 Horizon-level analiz
- 🧩 Feature-level analiz (decomposed)

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

## 🔄 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

1. Data Loading
   Neo4j → Dataset → Train/Val/Test Split
   
2. Training Loop
   ├─ Train Epoch (backprop + optimizer step)
   ├─ Validation Epoch (metrics)
   ├─ Early Stopping Check
   └─ Checkpoint Saving (best model)
   
3. Automatic Test Evaluation ✨
   ├─ Load best model
   ├─ Compute test metrics
   └─ Save test_metrics.json
   
4. Training Complete
   Output: checkpoints/ + test_metrics.json

┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION WORKFLOW                          │
└─────────────────────────────────────────────────────────────┘

1. Load Checkpoint
   best_model.pt → Model State
   
2. Basic Evaluation
   ├─ MSE, MAE, RMSE, MAPE, R²
   └─ Save test_metrics.json
   
3. Detailed Analysis (--detailed) ✨
   ├─ Segment-level errors
   ├─ Horizon-level errors
   ├─ Feature-level errors
   └─ Visualization (plots)
   
4. Evaluation Complete
   Output: evaluation/ folder
```

## ⚙️ Configuration Options

### Dataset Parameters
- `--window_size`: Input time window (default: 12)
- `--prediction_horizon`: Prediction steps (default: 3)
- `--stride`: Window stride (default: 1)
- `--use_last_n_days`: Use only recent data (optional)

### Model Parameters (STA-GCN)
- `--k_order`: Chebyshev order (default: 3)
- `--num_blocks`: ST blocks (default: 2)
- `--dropout`: Dropout rate (default: 0.5)

### Model Parameters (STGMS)
- `--periods`: Decomposition periods (default: [96, 16, 4])
- `--k_order`: Chebyshev order (default: 3)
- `--dropout`: Dropout rate (default: 0.5)

### Training Parameters
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--device`: cuda/cpu (default: cuda)

### ⚙️ STGMS Periyot Ayarları

Periyotlar veri sıklığınıza göre ayarlanmalıdır:

**15 dakikalık veri (4 sample/saat):**
```bash
--periods 96 16 4
# 96 = 1 gün (24 × 4)
# 16 = 4 saat (4 × 4)
# 4 = 1 saat (1 × 4)
```

**5 dakikalık veri (12 sample/saat):**
```bash
--periods 288 48 12
# 288 = 1 gün (24 × 12)
# 48 = 4 saat (4 × 12)
# 12 = 1 saat (1 × 12)
```

## 📝 Önemli Dosyalar

- `dataset_sta.py`: STA-GCN için standart zaman serisi yüklemesi
- `dataset_stgms.py`: STGMS için multi-timescale ayrıştırma
- `graph_utils.py`: Tüm modeller için ortak graf işlemleri
- `evaluators/evaluate_sta_gcn.py`: STA-GCN test metrikleri ve görselleştirme
- `evaluators/evaluate_stgms.py`: STGMS detaylı analiz ve görselleştirme

## ⚠️ Önemli Notlar

### Training Sırasında Otomatik Test Evaluation
✨ **YENİ**: Artık training scriptleri otomatik olarak test evaluation yapar:
- Training bitince en iyi model yüklenir
- Test seti üzerinde metrics hesaplanır
- `test_metrics.json` dosyası oluşturulur
- Manuel evaluation opsiyoneldir (detaylı analiz için)

### Data Leakage Prevention
- ✅ Normalizasyon **sadece train set** üzerinden hesaplanır
- ✅ Test seti hiçbir zaman training'e katılmaz
- ✅ STGMS decomposition causal padding kullanır

### Checkpoint Sistemi
Her training şunları kaydeder:
- `best_model_*.pt` - En iyi validation loss'lu model
- `checkpoint_epoch_*.pt` - Her 10 epoch
- `training_history_*.json` - Loss/MAE grafiği için
- `test_metrics.json` - ✨ Test sonuçları (otomatik)

### Output Klasörleri
```
outputs/
├── models/
│   ├── best_model.pt              # STA-GCN best
│   ├── test_metrics.json          # ✨ STA-GCN test
│   ├── training_history.json      # STA-GCN history
│   └── stgms/
│       ├── best_model_stgms.pt    # STGMS best
│       ├── test_metrics.json      # ✨ STGMS test
│       └── training_history_stgms.json
└── evaluation/
    ├── sta_gcn/
    │   ├── test_metrics.json      # Detaylı eval
    │   └── predictions_plot.png
    └── stgms/
        ├── test_metrics.json      # Detaylı eval
        ├── detailed_analysis.json # ✨ Segment/horizon
        └── predictions_plot.png
```

## 🎯 Hızlı Başlangıç

```bash
# 1. STA-GCN - Tam süreç
python src/gnn/trainers/train_sta_gcn.py --epochs 100
# ↳ Otomatik test evaluation yapılır ✅

# 2. STGMS - Tam süreç  
python src/gnn/trainers/train_stgms.py --epochs 100
# ↳ Otomatik test evaluation yapılır ✅

# 3. (Opsiyonel) Detaylı analiz
python src/gnn/evaluators/evaluate_sta_gcn.py --checkpoint outputs/models/best_model.pt
python src/gnn/evaluators/evaluate_stgms.py --checkpoint outputs/models/stgms/best_model_stgms.pt --detailed
```

## 📞 Model Karşılaştırma

### Hızlı Karşılaştırma
```bash
# Test metrics'leri kontrol et
python check_model.py

# Veya JSON dosyalarını direkt oku (PowerShell)
Get-Content outputs/models/test_metrics.json | ConvertFrom-Json
Get-Content outputs/models/stgms/test_metrics.json | ConvertFrom-Json
```

### 🆚 STA-GCN vs STGMS

| Özellik | STA-GCN | STGMS |
|---------|---------|-------|
| **Feature Boyutu** | F (örn: 8) | F × (m+1) (örn: 32) |
| **Periyodik Modelleme** | ❌ Yok | ✅ Multi-timescale decomposition |
| **Temporal Attention** | ❌ Yok | ✅ Var |
| **Spatial Attention** | ❌ Yok | ✅ Var |
| **Parametre Sayısı** | ~50K | ~133K |
| **Eğitim Süresi** | Daha hızlı | Daha yavaş |
| **Memory Kullanımı** | Daha az | Daha fazla |
| **Tahmin Doğruluğu** | İyi | Daha iyi (özellikle periyodik veriler) |
| **Anlık Olay Hassasiyeti** | Orta | Yüksek (residual component) |

### 🔍 STGMS Teknik Detaylar

**Feature Decomposition:**
- Original features: F (örn: 8)
- Periods: m (örn: 3 periyot)
- Decomposed features: F × (m + 1) = 8 × 4 = 32
- Model `in_channels=dataset.num_features_decomposed` kullanır

**Normalizasyon:**
- Sadece train set üzerinden istatistikler hesaplanır
- Data leakage önlenir
- Decomposed features normalize edilir

**Target:**
- Target (y) orijinal feature boyutunda (F_original)
- Decomposition sadece input'a uygulanır
- Model raw değerleri tahmin etmeyi öğrenir

## 🐛 Sorun Giderme

### "Out of memory" hatası
```bash
# Batch size'ı azalt
--batch_size 16

# Dropout artır (memory footprint azalır)
--dropout 0.6
```

### "NaN loss" sorunu
```bash
# Learning rate'i azalt
--lr 0.0001

# Gradient clipping zaten aktif (max_norm=5.0)
```

### Yavaş eğitim
```bash
# DataLoader worker'ları artır
--num_workers 4

# GPU kullan (otomatik tespit edilir)
# CUDA mevcut değilse CPU kullanılır
```

### Neo4j bağlantı hatası
```bash
# Neo4j'nin çalıştığından emin ol
# bolt://localhost:7687 adresinde erişilebilir olmalı
# Measure ve CONNECTS_TO verileri olmalı
```

---

**Son Güncelleme**: 2025-12-04  
**Özellikler**: 
- ✅ Otomatik test evaluation
- ✅ Tam entegre training/evaluation pipeline
- ✅ Data leakage prevention
- ✅ Detaylı analiz araçları
- ✅ İki model (STA-GCN, STGMS)
- ✅ Kapsamlı dokümantasyon
