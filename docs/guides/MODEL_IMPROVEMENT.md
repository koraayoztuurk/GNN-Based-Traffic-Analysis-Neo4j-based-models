# STA-GCN Model İyileştirme Önerileri

## 🔴 Mevcut Durum

**Test Sonuçları:**
- STA-GCN MAE: **0.2022** (3. sıra)
- Linear Regression MAE: **0.0449** (1. sıra, 81% daha iyi!)
- Last Value MAE: **0.0727** (2. sıra, 69% daha iyi!)

**Problem:** Model overfitting yapıyor ve basit baseline'lardan kötü performans gösteriyor.

---

## ✅ İyileştirme Önerileri

### 1. **Daha Fazla Veri Topla** (En Önemli!)

```bash
# Her 15 dakikada bir veri çek (7 gün)
# 7 gün × 96 sample/gün = 672 sample
# Sliding window (12+3=15) → ~650 sample

# Günlük pipeline
python run_pipeline.py  # Her 15 dakikada çalıştır
```

**Hedef:** 500+ sample (şu an 27)

---

### 2. **Model Basitleştir**

Şu anki model: 136,136 parametre (çok fazla!)

**Önerilen değişiklik:**

```python
# train.py'de
model = STAGCN(
    num_nodes=1342,
    in_channels=8,
    hidden_channels=[32, 16],  # [64, 64, 32] → [32, 16] (daha basit)
    out_channels=8,
    k_order=2,  # 3 → 2 (daha basit Chebyshev)
    kernel_size=3
)
```

**Yeni parametre sayısı:** ~40,000 (3× daha az)

---

### 3. **Longer Prediction Horizon**

Şu an: 12 → 3 (çok kısa, basit persistence yeterli)

```bash
# Daha zor bir task dene
python src/gnn/train.py \
    --window_size 24 \
    --prediction_horizon 12 \
    --epochs 50
```

**Hipotez:** Uzun vadeli tahmin'de GNN avantajlı olacak.

---

### 4. **Regularization Ekle**

```python
# train.py Trainer class'ında
self.optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4  # 1e-5 → 1e-4 (daha güçlü)
)

# Dropout ekle (model_sta_gcn.py'de)
self.dropout = nn.Dropout(0.2)
```

---

### 5. **Data Augmentation**

```python
# dataset.py'de
def __getitem__(self, idx):
    ...
    # Gaussian noise ekle (training'de)
    if self.training and np.random.rand() < 0.5:
        noise = torch.randn_like(x_window) * 0.01
        x_window = x_window + noise
    
    return {'x': x_window, 'y': y_window, ...}
```

---

### 6. **Ensemble Model**

```python
# En iyi strategi: Hybrid model
# - Short-term (1-3 step): Last Value
# - Mid-term (4-6 step): Linear Regression
# - Long-term (7+ step): STA-GCN

def ensemble_predict(x, horizon):
    if horizon <= 3:
        return last_value_predict(x)
    elif horizon <= 6:
        return linear_regression_predict(x)
    else:
        return sta_gcn_predict(x)
```

---

## 🎯 Önerilen Eylem Planı

### Kısa Vadede (1 hafta):

1. **7 gün veri topla** (her 15 dakika)
   ```bash
   # Windows Task Scheduler ile otomatikleştir
   python run_pipeline.py
   ```

2. **Basit model test et**
   ```bash
   python src/gnn/train.py \
       --hidden_channels 32 16 \
       --k_order 2 \
       --epochs 100 \
       --patience 20
   ```

3. **Tekrar değerlendir**
   ```bash
   python src/gnn/evaluate.py --compare_baselines
   ```

### Orta Vadede (1 ay):

4. **Longer horizon dene**
   ```bash
   python src/gnn/train.py \
       --window_size 24 \
       --prediction_horizon 12
   ```

5. **Ensemble model oluştur**

6. **Spatial attention ekle** (model_sta_gcn.py'de zaten var, aktifleştir)

---

## 📈 Beklenen Gelişme

| Senaryo | MAE (şu an: 0.2022) | Improvement |
|---------|---------------------|-------------|
| **Daha fazla veri** (500+ sample) | 0.08 - 0.12 | 40-60% ⬆️ |
| **Model basitleştirme** | 0.15 - 0.18 | 10-25% ⬆️ |
| **Longer horizon** | 0.05 - 0.10 | 50-75% ⬆️ (GNN avantajı!) |
| **Ensemble** | 0.04 - 0.06 | **80-85% ⬆️** 🏆 |

---

## 🧠 Önemli Not

**Şu an Linear Regression kazanıyor çünkü:**
- ✅ Task çok basit (3 step)
- ✅ Az veri var (27 sample)
- ✅ Spatial bilgi yeterince önemli değil (short-term)

**GNN ne zaman kazanır:**
- ✅ Uzun vadeli tahmin (12+ step)
- ✅ Çok veri (500+ sample)
- ✅ Kompleks spatial dependencies (trafik propagation)

---

## 📚 Kaynaklar

**Papers:**
- T-GCN: "Temporal Graph Convolutional Network for Urban Traffic Flow Prediction" (2019)
- DCRNN: "Diffusion Convolutional Recurrent Neural Network" (2018)
- Graph WaveNet: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" (2019)

**Insight:** Tüm papers 6+ aylık veri kullanıyor (10,000+ sample)

---

## ✅ Sonuç

**Model çalışıyor!** Ama daha fazla veri gerekiyor. 

**Next steps:**
1. 7 gün veri topla
2. Model basitleştir
3. Tekrar test et
4. Longer horizon dene

27 sample ile 0.20 MAE almak aslında fena değil - daha fazla veriyle 0.05-0.08 MAE mümkün! 🚀
