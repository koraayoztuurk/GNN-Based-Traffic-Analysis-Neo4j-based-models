# 🚀 Database Benchmark Sistemi

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Benchmark Tipleri](#benchmark-tipleri)
3. [Hızlı Başlangıç](#hızlı-başlangıç)
4. [Dokümanlar](#dokümanlar)
5. [Sonuçlar](#sonuçlar)

---

## 🎯 Genel Bakış

Bu proje **2 seviye** benchmark sistemi sunar:

### 1️⃣ Basit Benchmark (`benchmark_databases.py`)
- ✅ **7 temel test**
- ✅ **Hızlı sonuç** (~30 saniye)
- ✅ **Basit metrikler** (ortalama, toplam)
- 👉 İlk karşılaştırma için ideal

### 2️⃣ Kapsamlı Benchmark (`benchmark_comprehensive.py`)
- ✅ **8 kategori, 15+ metrik**
- ✅ **İstatistiksel analiz** (Mean, Median, P50, P90, P95, P99, StdDev)
- ✅ **Concurrent test** (5-500 eşzamanlı kullanıcı)
- ✅ **Stress test** (10-300 saniye sürekli yük)
- ✅ **Write performance** (CREATE, UPDATE, DELETE)
- ✅ **4 profil** (quick, standard, production, stress)
- 👉 Production kararı için ideal

---

## 📊 Benchmark Tipleri

### Basit Benchmark
```powershell
# Hızlı test (3 basit metrik)
python benchmark_databases.py --quick

# Full test (7 detaylı metrik)
python benchmark_databases.py --full

# Tek database
python benchmark_databases.py --db arangodb
```

**Çıktılar**:
- `benchmark_results.json` - JSON sonuçlar
- `BENCHMARK_REPORT.md` - Markdown rapor
- Konsol tablosu

### Kapsamlı Benchmark

#### Quick Profile (~30 saniye)
```powershell
python benchmark_comprehensive.py --profile quick --db arangodb
```
- 3 iterasyon
- 1 warmup run
- 5 concurrent user
- 10 saniye stress test

#### Standard Profile (~2 dakika)
```powershell
python benchmark_comprehensive.py --profile standard
```
- 10 iterasyon
- 3 warmup run
- 20 concurrent user
- 30 saniye stress test

#### Production Profile (~10 dakika)
```powershell
python benchmark_comprehensive.py --profile production --db neo4j,arangodb
```
- 50 iterasyon
- 5 warmup run
- 100 concurrent user
- 60 saniye stress test

#### Stress Profile (~30 dakika)
```powershell
python benchmark_comprehensive.py --profile stress --db tigergraph
```
- 100 iterasyon
- 10 warmup run
- 500 concurrent user
- 300 saniye stress test

**Çıktılar**:
- `comprehensive_benchmark_results.json` - Detaylı JSON
- Konsol tabloları (istatistiklerle)

#### HTML Dashboard
```powershell
# Dashboard oluştur
python generate_dashboard.py

# Custom input/output
python generate_dashboard.py --input results.json --output dashboard.html
```

**Dashboard Özellikleri**:
- 📊 Interaktif grafikler (Chart.js)
- 🏆 Winner badges
- 📈 Karşılaştırma chartları
- 📋 Detaylı tablolar
- 🎨 Modern responsive tasarım

---

## 🚀 Hızlı Başlangıç

### 1. İlk Deneme (30 saniye)
```powershell
# Basit benchmark ile başla
python benchmark_databases.py --quick --db arangodb,tigergraph
```

✅ Hangi database daha hızlı?
✅ Problem var mı?

### 2. Detaylı Analiz (2 dakika)
```powershell
# Kapsamlı benchmark - standard profile
python benchmark_comprehensive.py --profile standard --db arangodb
```

✅ İstatistiksel güvenilir sonuçlar
✅ P95/P99 metrikleri
✅ Concurrent + Stress test

### 3. Dashboard Görselleştirme
```powershell
# HTML dashboard oluştur
python generate_dashboard.py

# Browser'da aç
start benchmark_dashboard.html
```

✅ Grafiklerle karşılaştırma
✅ Winner badges
✅ Export/print ready

---

## 📚 Dokümanlar

### Ana Dokümanlar

1. **[BENCHMARK_USAGE.md](BENCHMARK_USAGE.md)**
   - Basit benchmark kullanımı
   - Test kategorileri açıklaması
   - Troubleshooting

2. **[COMPREHENSIVE_BENCHMARK_GUIDE.md](COMPREHENSIVE_BENCHMARK_GUIDE.md)**
   - Kapsamlı benchmark rehberi
   - İstatistik metrikleri (P95, P99, etc.)
   - Gerçek dünya örnekleri
   - Best practices

3. **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**
   - Son test sonuçları
   - Database karşılaştırması
   - Öneriler

### Test Sonuçları

```
comprehensive_benchmark_results.json  # Kapsamlı test sonuçları (JSON)
benchmark_results.json                 # Basit test sonuçları (JSON)
benchmark_dashboard.html               # Interaktif dashboard
BENCHMARK_REPORT.md                    # Markdown rapor
```

---

## 📊 Sonuçlar

### Test Kategorileri

#### Basit Benchmark (7 test)
1. ✅ Connection Speed
2. ✅ Read Segments
3. ✅ Read Measures
4. ✅ Spatial Query
5. ✅ Temporal Query
6. ✅ Graph Traversal
7. ✅ Aggregation

#### Kapsamlı Benchmark (8 kategori, 15+ metrik)
1. ✅ **Connection Speed** - Bağlantı latency
2. ✅ **Read Performance** - Segments ve Measures okuma
3. ✅ **Graph Traversal** - 1-hop, 2-hop, 3-hop
4. ✅ **Shortest Path** - En kısa yol bulma
5. ✅ **Aggregation** - AVG, MIN, MAX, SUM
6. ✅ **Write Performance** - CREATE, UPDATE, DELETE
7. ✅ **Concurrent Reads** - Çoklu kullanıcı (5-500 user)
8. ✅ **Stress Test** - Sürekli yük (10-300 saniye)

### Metrikler

#### Basit Benchmark
- Ortalama süre
- Toplam kayıt sayısı
- Memory kullanımı

#### Kapsamlı Benchmark
- **Mean**: Ortalama
- **Median**: Ortanca
- **P50**: 50th percentile
- **P90**: 90th percentile
- **P95**: 95th percentile ⭐
- **P99**: 99th percentile ⭐⭐
- **Min/Max**: En düşük/yüksek
- **StdDev**: Standart sapma
- **Variance**: Varyans

### Örnek Sonuç (ArangoDB - Quick Profile)

```
[TEST] Connection Speed
  Mean:   3.00 ms [WINNER]
  P95:    3.55 ms
  P99:    3.55 ms

[TEST] Read Performance
  segments:
    Mean:   4.86 ms [WINNER]
    P95:    6.37 ms
  measures:
    Mean:   4.84 ms [WINNER]
    P95:    5.82 ms

[TEST] Aggregation
  avg:
    Mean:   6.31 ms [WINNER]
    P95:    8.20 ms

[TEST] Write Performance
  create:
    Mean:   7.30 ms [WINNER]
    P99:    10.88 ms
  update:
    Mean:   5.43 ms [WINNER]
  delete:
    Mean:   4.64 ms [WINNER]

[TEST] Concurrent Reads (5 users)
  throughput:
    580.12 QPS [WINNER]
  times:
    P99:    11.99 ms

[TEST] Stress Test (10 seconds)
  total_queries:
    3187
  queries_per_second:
    318.69 QPS
  times:
    P99:    7.97 ms
  errors:
    0

GENEL SKOR: 18/18 metrik kazandı (100.0%)
```

---

## 🎯 Karar Matrisi

### Development Database Seçimi
```
Kullan: benchmark_comprehensive.py --profile standard
Öncelik: Read hızı, Graph traversal
Kriterler:
  - Read Performance Mean <10ms
  - Graph Traversal 2-hop P95 <50ms
  - Write Performance Mean <15ms
```

### Production Database Seçimi
```
Kullan: benchmark_comprehensive.py --profile production
Öncelik: P99, Concurrent, Stress
Kriterler:
  - Concurrent Reads P99 <20ms
  - Stress Test QPS >200
  - Stress Test Errors = 0
  - All Tests P99 < 2x Mean
```

### Scalability Analizi
```
Kullan: benchmark_comprehensive.py --profile stress
Öncelik: Throughput, Stability
Kriterler:
  - Stress Test QPS >500
  - P99/P95 ratio <2
  - Max latency <100ms
  - Zero errors
```

---

## 🔧 Kurulum

### Gerekli Paketler
```powershell
pip install -r config/requirements.txt
```

**Paketler**:
- `neo4j>=5.0` - Neo4j driver
- `python-arango>=7.1` - ArangoDB client
- `pyTigerGraph>=1.0` - TigerGraph client
- `psutil>=5.9` - Resource monitoring
- `python-dotenv>=1.0` - .env dosyası

### Konfigürasyon

`config/.env` dosyasını düzenle:
```env
# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=123456789

# ArangoDB
ARANGO_HOST=http://127.0.0.1:8529
ARANGO_USER=root
ARANGO_PASS=1234
ARANGO_DATABASE=traffic_db

# TigerGraph
TIGER_HOST=http://127.0.0.1
TIGER_USERNAME=tigergraph
TIGER_PASSWORD=tigergraph
TIGER_GRAPHNAME=TrafficGraph
```

---

## 📈 Workflow Önerisi

### Adım 1: İlk Test (30 saniye)
```powershell
python benchmark_databases.py --quick --db arangodb,tigergraph
```
→ Genel bakış, hangi DB daha hızlı?

### Adım 2: Detaylı Analiz (2 dakika)
```powershell
python benchmark_comprehensive.py --profile standard --db arangodb
```
→ İstatistiksel güvenilir sonuçlar

### Adım 3: Dashboard (10 saniye)
```powershell
python generate_dashboard.py
start benchmark_dashboard.html
```
→ Görsel karşılaştırma

### Adım 4: Final Karar (10 dakika)
```powershell
python benchmark_comprehensive.py --profile production --db arangodb
```
→ Production'a geçmeden önce son test

### Adım 5: Limit Testi (30 dakika - opsiyonel)
```powershell
python benchmark_comprehensive.py --profile stress --db arangodb
```
→ Maksimum kapasite analizi

---

## 🏆 Sonuç Yorumlama

### Target Değerler

| Metrik | Target | Good | Excellent |
|--------|--------|------|-----------|
| Connection Speed | <50ms | <10ms | <5ms |
| Read Performance | <20ms | <10ms | <5ms |
| Graph Traversal (1-hop) | <50ms | <30ms | <20ms |
| Graph Traversal (2-hop) | <100ms | <50ms | <30ms |
| Shortest Path | <100ms | <50ms | <30ms |
| Aggregation | <20ms | <10ms | <5ms |
| Write (Create) | <20ms | <10ms | <5ms |
| Concurrent (P99) | <50ms | <20ms | <10ms |
| Stress (QPS) | >100 | >300 | >500 |

### Önemli Metrikler

#### Development
- ✅ Mean (ortalama performans)
- ✅ Read Performance
- ✅ Write Performance

#### Production
- ✅ **P99** (worst-case latency)
- ✅ **Concurrent Reads** (throughput)
- ✅ **Stress Test** (stability)
- ✅ **Errors** (zero olmalı!)

---

## 🆘 Troubleshooting

### Problem: Connection failed
```
[ERROR] Unable to connect to database
```

**Çözüm**:
1. Database servisini başlat
2. `config/.env` dosyasını kontrol et
3. Firewall ayarlarını kontrol et

### Problem: Graph tests fail
```
[ERROR]: [HTTP 404][ERR 1924] graph 'traffic_flow_graph' not found
```

**Çözüm**: ArangoDB'de graph oluştur
```python
from arango import ArangoClient

client = ArangoClient(hosts='http://localhost:8529')
db = client.db('traffic_db', username='root', password='1234')

if not db.has_graph('traffic_flow_graph'):
    graph = db.create_graph('traffic_flow_graph')
    graph.create_edge_definition(
        edge_collection='CONNECTS_TO',
        from_vertex_collections=['Segment'],
        to_vertex_collections=['Segment']
    )
```

### Problem: Low throughput
```
Concurrent Reads:
  throughput: 50 QPS  (Çok düşük!)
```

**Çözüm**:
1. Index'leri kontrol et
2. Connection pool size'ı artır
3. Profile değiştir (quick → standard)
4. Warmup run sayısını artır

---

## 📞 İletişim

Sorular için: emiralibulutt@gmail.com

Proje: Traffic Flow Analysis & GNN

---

## 📄 Lisans

Bu benchmark sistemi Traffic Flow Analysis & GNN projesi kapsamında geliştirilmiştir.

---

## 🙏 Teşekkürler

- Neo4j, ArangoDB, TigerGraph ekiplerine
- Chart.js ekibine
- Python community

---

**Son Güncelleme**: 23 Kasım 2025

**Versiyon**: 2.0 (Kapsamlı Benchmark Sistemi)
