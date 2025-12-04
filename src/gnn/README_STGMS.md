# STGMS Model Integration

## 📚 Genel Bakış

Bu klasörde **STGMS (Spatio-Temporal Graph Neural Network with Multi-timeScale)** modeli için gerekli tüm dosyalar bulunmaktadır. STGMS, trafik tahmininde standart GNN'lere göre daha başarılı sonuçlar veren, multi-timescale decomposition kullanan gelişmiş bir modeldir.

## 🎯 Model Özellikleri

### 1. Multi-timescale Decomposition
- **Trend bileşeni**: Uzun dönem periyodik değişimler (haftalık, günlük)
- **Orta dönem bileşenler**: Saatlik döngüler
- **Residual bileşen**: Anlık değişimler ve olaylar (kazalar, hava durumu)

### 2. Attention Mechanisms
- **Temporal Attention**: Zaman adımları arası bağımlılıkları öğrenir
- **Spatial Attention**: Segment'ler arası bağımlılıkları öğrenir

### 3. Graph Convolution
- **Chebyshev Graph Convolution**: Topoloji bilgisini kullanır
- **Gaussian Kernel**: Mesafe bazlı ağırlıklandırma

## 📁 Dosya Yapısı

```
src/gnn/
├── dataset_stgms.py       # STGMSDataset - Multi-timescale decomposition
├── model_stgms.py         # STGMS model mimarisi
└── trainers/
    └── train_stgms.py     # Training script
```

## 🚀 Kullanım

### 1. Temel Kullanım

```bash
# Basit eğitim (varsayılan parametreler)
python src/gnn/trainers/train_stgms.py

# Özel parametrelerle eğitim
python src/gnn/trainers/train_stgms.py \
    --epochs 100 \
    --batch_size 32 \
    --periods 96 16 4 \
    --lr 0.001
```

### 2. Periyot Seçimi

Periyotlar, veri sıklığınıza göre ayarlanmalıdır:

**15 dakikalık veri için (4 sample/saat):**
```bash
--periods 96 16 4
# 96 = 1 gün (24 * 4)
# 16 = 4 saat (4 * 4)
# 4 = 1 saat (1 * 4)
```

**5 dakikalık veri için (12 sample/saat):**
```bash
--periods 288 48 12
# 288 = 1 gün (24 * 12)
# 48 = 4 saat (4 * 12)
# 12 = 1 saat (1 * 12)
```

### 3. Zaman Filtreleme

```bash
# Son 7 günün verisi
python src/gnn/trainers/train_stgms.py --use_last_n_days 7

# Belirli tarih aralığı
python src/gnn/trainers/train_stgms.py \
    --start_time "2024-11-01T00:00:00Z" \
    --end_time "2024-11-30T23:59:59Z"
```

### 4. Checkpoint Kullanımı

```bash
# Training'e devam et
python src/gnn/trainers/train_stgms.py \
    --resume outputs/models/stgms/checkpoint_stgms_epoch_50.pt

# Fine-tuning (optimizer sıfırla)
python src/gnn/trainers/train_stgms.py \
    --resume outputs/models/stgms/best_model_stgms.pt \
    --fine_tune \
    --lr 0.0001
```

## 🔧 Önemli Parametreler

### Dataset Parametreleri
```bash
--window_size 12              # Geçmiş pencere boyutu (varsayılan: 12)
--prediction_horizon 3        # Tahmin horizon'u (varsayılan: 3)
--periods 96 16 4             # Decomposition periyotları
--stride 1                    # Window kaydırma adımı
```

### Model Parametreleri
```bash
--k_order 3                   # Chebyshev polynomial sırası
--dropout 0.5                 # Dropout oranı
--sigma 50.0                  # Gaussian kernel sigma (metre)
```

### Training Parametreleri
```bash
--epochs 100                  # Epoch sayısı
--batch_size 32               # Batch boyutu
--lr 0.001                    # Learning rate
--patience 10                 # Early stopping patience
--train_ratio 0.7             # Train set oranı
--val_ratio 0.15              # Validation set oranı
```

## 📊 Çıktılar

Training sonrası oluşturulacak dosyalar:

```
outputs/models/stgms/
├── best_model_stgms.pt                    # En iyi model
├── checkpoint_stgms_epoch_10.pt           # Checkpoint'ler
├── checkpoint_stgms_epoch_20.pt
└── training_history_stgms.json            # Eğitim geçmişi
```

## 🧪 Test ve Doğrulama

### Dataset Testi
```bash
python src/gnn/dataset_stgms.py
```

### Model Testi
```bash
python src/gnn/model_stgms.py
```

## 📈 Beklenen Performans

STGMS modelinin standart GNN'lere göre avantajları:

1. **Daha iyi periyodik kalıp öğrenme**: Trend ve döngüsel değişimleri ayrı ayrı modeller
2. **Anlık olay hassasiyeti**: Residual bileşen sayesinde ani değişimleri yakalar
3. **Daha az overfitting**: Multi-scale ayrıştırma regularization etkisi yapar

## 🔍 Kritik Notlar

### 1. Feature Boyutu
```python
# Original features: F (örn: 8)
# Periods: m (örn: 3)
# Decomposed features: F * (m + 1) = 8 * 4 = 32
```

Model `in_channels=dataset.num_features_decomposed` kullanır!

### 2. Normalizasyon
- Sadece **train set** üzerinden istatistikler hesaplanır
- **Data leakage** önlenir
- Decomposed features normalize edilir

### 3. Target
- Target (y) orijinal feature boyutunda (F_original)
- Decomposition sadece input'a uygulanır
- Model raw değerleri tahmin etmeyi öğrenir

## 🆚 STA-GCN vs STGMS

| Özellik | STA-GCN | STGMS |
|---------|---------|-------|
| Feature boyutu | F | F * (m+1) |
| Periyodik modelleme | ❌ | ✅ |
| Temporal attention | ❌ | ✅ |
| Spatial attention | ❌ | ✅ |
| Parametre sayısı | Daha az | Daha fazla |
| Eğitim süresi | Daha hızlı | Daha yavaş |
| Tahmin doğruluğu | İyi | Daha iyi |

## 📚 Referans

**Makale**: "Spatio-Temporal Graph Neural Network with Multi-timeScale"
- Section 3.1: Multi-timescale Feature Decomposition (Eq. 2 & 3)
- Section 3.2: Temporal Attention Mechanism
- Section 3.3: Spatial Attention Mechanism
- Section 3.4: Chebyshev Graph Convolution

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
--num_workers 8

# CUDA kullan (otomatik tespit edilir)
--device cuda
```

## ✅ Checklist

Eğitime başlamadan önce kontrol edin:

- [ ] Neo4j'de Measure verileri var mı? (`MATCH (m:Measure) RETURN count(m)`)
- [ ] Graf topolojisi hazır mı? (`MATCH ()-[r:CONNECTS_TO]->() RETURN count(r)`)
- [ ] Periyotlar veri sıklığına uygun mu?
- [ ] Yeterli RAM/GPU memory var mı?
- [ ] Python dependencies yüklü mü? (torch, torch_geometric, neo4j)

## 📞 Yardım

Sorun yaşarsanız:
1. Test scriptlerini çalıştırın (`dataset_stgms.py`, `model_stgms.py`)
2. Log dosyalarını kontrol edin
3. `--batch_size` ve `--num_workers` parametrelerini ayarlayın
