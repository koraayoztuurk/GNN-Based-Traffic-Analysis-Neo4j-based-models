# STA-GCN Training Stratejileri

## 📊 Veri Kaynağı

Model Neo4j'deki verileri kullanır:
- **Segment**: Yol segmentleri (node'lar)
- **CONNECTS_TO**: Segment bağlantıları (edge'ler)  
- **Measure**: Trafik ölçümleri (timestamp'li)

```cypher
MATCH (s:Segment)-[:AT_TIME]->(m:Measure)
WHERE m.timestamp >= '2024-11-01T00:00:00Z'
RETURN s.segmentId, m.timestamp, m.speed, m.jamFactor, ...
```

---

## 🎯 Training Senaryoları

### **Senaryo 1: İlk Tam Eğitim** (Full Training)

Tüm geçmiş veriyle modeli sıfırdan eğit.

```bash
# Tüm Neo4j verisi ile eğit
python src/gnn/train.py --epochs 100 --batch_size 32

# Çıktı: outputs/models/best_model.pt (120,000 parametre)
```

**Ne zaman kullanılır:**
- ✅ İlk model oluşturulurken
- ✅ Model mimarisini değiştirince
- ✅ Ayda 1 kez tam retrain

**Süre:** ~2-4 saat (GPU), ~10-20 saat (CPU)

---

### **Senaryo 2: Son N Gün Verisi ile Eğitim**

Sadece son birkaç gün/hafta verisi kullan (hızlı test için).

```bash
# Son 7 gün verisi ile eğit
python src/gnn/train.py \
    --use_last_n_days 7 \
    --epochs 50 \
    --batch_size 32

# Son 30 gün (rolling window)
python src/gnn/train.py --use_last_n_days 30 --epochs 100
```

**Ne zaman kullanılır:**
- ✅ Hızlı prototip test
- ✅ Son trendleri yakalamak için
- ✅ Eski veri kalitesi düşükse

**Süre:** ~30 dakika - 1 saat

---

### **Senaryo 3: Belirli Tarih Aralığı**

Spesifik bir zaman dilimindeki veriyle eğit.

```bash
# Kasım ayı verisi
python src/gnn/train.py \
    --start_time "2024-11-01T00:00:00Z" \
    --end_time "2024-11-30T23:59:59Z" \
    --epochs 80

# Sadece Kasım 15'ten sonrası
python src/gnn/train.py --start_time "2024-11-15T00:00:00Z"
```

**Ne zaman kullanılır:**
- ✅ Belirli bir event analizi (tatil, etkinlik)
- ✅ Veri kalitesi problemi olan dönemleri çıkarmak
- ✅ Sezonsal model eğitimi

---

### **Senaryo 4: Incremental Training** ⭐ (ÖNERİLEN)

Mevcut modeli yeni veriyle güncelle (hızlı, verimli).

```bash
# 1. İlk tam eğitim (bir kez)
python src/gnn/train.py --epochs 100

# 2. Her gün yeni veri çek
python run_pipeline.py

# 3. Sadece son 1 gün ile model güncelle (fine-tune)
python src/gnn/incremental_train.py --last_n_days 1
```

**Otomatik pipeline:**
```bash
# Her gün çalıştır (Task Scheduler / cron)
python run_pipeline.py && python src/gnn/incremental_train.py --last_n_days 1
```

**Ne zaman kullanılır:**
- ✅ **Günlük veri güncellemeleri** (HERE API'den yeni veri geldi)
- ✅ Modeli sıfırdan eğitmeden güncelleme
- ✅ Hızlı deployment (20 epoch yeterli)

**Süre:** ~10-15 dakika

**Avantajlar:**
- 🚀 Çok hızlı (100 epoch yerine 20 epoch)
- 💾 Eski öğrendikleri kaybetmez
- 🔄 Sürekli öğrenme (continual learning)

---

### **Senaryo 5: Checkpoint'ten Devam Etme**

Eğitim yarıda kaldıysa devam et.

```bash
# Training yarıda kesildi (Ctrl+C veya crash)
python src/gnn/train.py \
    --resume outputs/models/best_model.pt \
    --epochs 100

# Optimizer state'i de yükler, kaldığı yerden devam eder
```

**Ne zaman kullanılır:**
- ✅ Power outage / sistem crash
- ✅ Daha fazla epoch eklemek
- ✅ Learning rate değiştirip devam etmek

---

## 🔄 Önerilen İş Akışı

### **Haftalık Döngü**

```
Pazartesi 00:00:
├─ python run_pipeline.py  (yeni veri çek)
└─ python src/gnn/incremental_train.py --last_n_days 1

Salı 00:00:
├─ python run_pipeline.py
└─ python src/gnn/incremental_train.py --last_n_days 1

...

Pazar 00:00:
├─ python run_pipeline.py
└─ python src/gnn/train.py --use_last_n_days 30 --epochs 100
   (haftalık tam retrain)
```

### **Aylık Döngü**

```
Her gün:
  python run_pipeline.py
  python src/gnn/incremental_train.py --last_n_days 1

Her Ay 1.:
  python src/gnn/train.py --epochs 100
  (tüm veriyle tam retrain)
```

---

## 📝 Parametreler

### Dataset Filtreleme

| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `--use_last_n_days` | Son N gün verisi | `--use_last_n_days 7` |
| `--start_time` | Başlangıç zamanı | `--start_time "2024-11-01T00:00:00Z"` |
| `--end_time` | Bitiş zamanı | `--end_time "2024-11-30T23:59:59Z"` |

### Checkpoint

| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `--resume` | Checkpoint'ten devam et | `--resume outputs/models/best_model.pt` |
| `--fine_tune` | Sadece weights yükle, optimizer reset | `--fine_tune` |

### Training

| Parametre | Açıklama | Default | Fine-tune Önerisi |
|-----------|----------|---------|-------------------|
| `--epochs` | Epoch sayısı | 100 | 20 |
| `--lr` | Learning rate | 0.001 | 0.0001 |
| `--batch_size` | Batch size | 32 | 32 |
| `--patience` | Early stopping | 10 | 5 |

---

## 🧪 Test ve Debugging

### Hızlı Test (Küçük Veri)

```bash
# Son 1 gün, 10 epoch (3 dakika)
python src/gnn/train.py --use_last_n_days 1 --epochs 10
```

### Veri Miktarı Kontrolü

```python
from src.gnn.dataset import TrafficDataset

# Tüm veri
dataset_all = TrafficDataset()
print(f"Toplam sample: {len(dataset_all)}")

# Son 7 gün
dataset_week = TrafficDataset(use_last_n_days=7)
print(f"Son 7 gün: {len(dataset_week)}")
```

### Checkpoint İnceleme

```python
import torch

checkpoint = torch.load('outputs/models/best_model.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['best_val_loss']}")
print(f"History: {checkpoint['history']}")
```

---

## ⚠️ Önemli Notlar

### Veri Kalitesi

- ❌ **Neo4j boşsa:** `ValueError: Neo4j'de Measure verisi bulunamadı!`
- ✅ **Önce veri yükle:** `python run_pipeline.py`

### Timestamp Formatı

- ✅ ISO 8601: `2024-11-27T14:30:00Z`
- ❌ Yanlış: `2024-11-27 14:30:00` (Z eksik)

### GPU Memory

- 100 node, batch_size=32 → ~2GB VRAM
- 500 node, batch_size=32 → ~8GB VRAM
- Out of memory? → `--batch_size 16` veya `--batch_size 8`

### Incremental Training Sınırları

- Her 50-100 incremental update'ten sonra **tam retrain** yapın
- Model drift önlemek için aylık full retrain önerilir

---

## 📊 Örnek Çıktı

```
======================================================================
STA-GCN Training
======================================================================

📦 Loading dataset...
  🕒 Sadece son 7 gün verisi kullanılacak
  ✓ 100 segment yüklendi
  ✓ 450 edge yüklendi
  ✓ 16800 measure kaydı yüklendi
  📅 Tarih aralığı: 2024-11-21T00:00:00Z → 2024-11-27T23:45:00Z
  ✓ 8 feature hazırlandı
  ✓ Feature tensor hazır: (672, 100, 8)
  ✓ 657 window oluşturuldu
✅ Dataset hazır: 657 samples

📊 Splitting dataset...
  - Train: 459 samples (70%)
  - Val: 98 samples (15%)
  - Test: 100 samples (15%)

🏗️  Building model...
  - Model: STA-GCN
  - Parameters: 119,432
  - Device: cuda

🚀 Training başlıyor...

Epoch 1/50 (12.3s)
  Train - Loss: 0.0234, MAE: 0.1123
  Val   - Loss: 0.0198, MAE: 0.0987
  LR: 0.001000
  ✅ New best model! Val loss: 0.0198

...

✅ Training tamamlandı!
  - Total time: 8.5 dakika
  - Best val loss: 0.0145
```

---

## 🚀 Hızlı Başlangıç

```bash
# 1. İlk tam eğitim (bir kez)
python src/gnn/train.py --epochs 100

# 2. Günlük pipeline (otomate et)
python run_pipeline.py
python src/gnn/incremental_train.py --last_n_days 1

# 3. Haftalık tam eğitim (otomate et)
python src/gnn/train.py --use_last_n_days 30 --epochs 100
```

**Başarılar!** 🎉
