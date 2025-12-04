# Database Benchmark Kullanım Kılavuzu

## Hızlı Başlangıç

```powershell
# Quick test (3 temel metrik - ~1 saniye)
python benchmark_databases.py --quick

# Full test (7 detaylı metrik - ~5 saniye)
python benchmark_databases.py --full

# Tek database test et
python benchmark_databases.py --db arangodb
python benchmark_databases.py --db tigergraph
python benchmark_databases.py --db neo4j
```

## Test Kategorileri

### 1. Connection Speed (Bağlantı Hızı)
- **Ne test eder**: İlk bağlantı kurma süresi
- **Neden önemli**: Uygulama başlatma performansı
- **Beklenen**: <50ms ideal

### 2. Read Segments (Segment Okuma)
- **Ne test eder**: `Segment` tablosundaki tüm kayıtları sayma
- **Neden önemli**: Basit okuma operasyonlarının hızı
- **Beklenen**: <10ms ideal

### 3. Read Measures (Ölçüm Okuma)
- **Ne test eder**: `Measure` tablosundaki tüm kayıtları sayma
- **Neden önemli**: Zaman serisi verisi okuma hızı
- **Beklenen**: <10ms ideal

### 4. Spatial Query (Coğrafi Sorgu)
- **Ne test eder**: `CONNECTS_TO` edge sayısı
- **Neden önemli**: Graf ilişkilerini sorgulama hızı
- **Beklenen**: <10ms ideal

### 5. Temporal Query (Zamansal Sorgu)
- **Ne test eder**: Son 10 ölçümü timestamp'e göre sıralama
- **Neden önemli**: Zaman bazlı filtreleme ve sıralama
- **Beklenen**: <20ms ideal

### 6. Graph Traversal (Graf Gezinme)
- **Ne test eder**: 1-2 hop komşuları bulma (A8001_113599020)
- **Neden önemli**: GNN/ML modellerinde kritik
- **Beklenen**: <50ms ideal

### 7. Aggregation (Toplama)
- **Ne test eder**: Ortalama hız hesaplama (AVG speed_kmh)
- **Neden önemli**: Analitik sorgular için
- **Beklenen**: <20ms ideal

## Sonuçlar

### Çıktı Formatları

1. **Konsol Tablosu**: Terminal'de anında görme
2. **benchmark_results.json**: Makine-okunabilir detaylı sonuçlar
3. **BENCHMARK_REPORT.md**: İnsan-okunabilir analiz ve öneriler

### Sonuç Yorumlama

```
🏆 = Bu metrikte kazanan database
Time: Daha düşük = Daha iyi
Memory: Daha düşük = Daha iyi
Count/Neighbors: Doğruluk kontrolü (tüm DB'lerde aynı olmalı)
```

## Mevcut Test Sonuçları (2025-11-23)

### ArangoDB ⭐ (11/14 metrik kazandı)

**Güçlü Yönleri:**
- ✅ Read operations (2-3ms)
- ✅ Graph traversal (26ms)
- ✅ Aggregation (3ms)
- ✅ Memory efficiency (0.09MB)

**Zayıf Yönleri:**
- ❌ Connection speed (28ms vs TigerGraph 6ms)

### TigerGraph (3/14 metrik kazandı)

**Güçlü Yönleri:**
- ✅ Connection speed (6ms)
- ✅ Temporal query (6ms)

**Zayıf Yönleri:**
- ❌ Graph traversal (92ms vs ArangoDB 26ms)
- ❌ Memory usage (1.82MB vs ArangoDB 0.09MB)

### Neo4j (Test edilemedi - servis çalışmıyor)

**Test etmek için:**
```powershell
# 1. Neo4j Desktop'ı başlat
# 2. Database'i start et
# 3. Benchmark'ı tekrar çalıştır
python benchmark_databases.py --db neo4j
```

## Öneriler

### Geliştirme (Development)
**→ ArangoDB kullan**
- Hızlı read/write
- Az memory kullanımı
- AQL sorguları kolay yazılır

### Production (Üretim)
**→ İkisi de uygun, kullanım senaryosuna göre:**

#### ArangoDB tercih et eğer:
- Çok sayıda graph traversal yapacaksın (GNN/ML)
- Memory/maliyet önemliyse
- Aggregation/analitik sorgular çoksa

#### TigerGraph tercih et eğer:
- Çok sayıda concurrent connection varsa
- Çok büyük ölçeklere çıkacaksın (>1M node)
- GSQL ile complex query'ler yazacaksın

## Troubleshooting

### Neo4j bağlantı hatası
```
ServiceUnavailable: Unable to retrieve routing information
```
**Çözüm**: Neo4j Desktop'tan database'i start et

### TigerGraph timeout
```
ReadTimeout: HTTPSConnectionPool
```
**Çözüm**: TigerGraph Cloud'un ayakta olduğunu kontrol et

### ArangoDB authentication error
```
ServerConnectionError: [401][ERR 11] not authorized
```
**Çözüm**: .env dosyasında ARANGO_PASSWORD doğru olduğunu kontrol et

## İleri Seviye Kullanım

### Custom Testler Eklemek

```python
# benchmark_databases.py içinde yeni test ekle
def test_custom_query(self):
    """Custom test açıklaması."""
    start = time.time()
    
    # Sorgunuzu buraya yazın
    result = self.db.aql.execute("YOUR QUERY")
    
    elapsed = (time.time() - start) * 1000
    return elapsed, result.count()
```

### Benchmark'ı Otomatize Etmek

```powershell
# Windows Task Scheduler ile günlük benchmark
# setup_windows_task.ps1 benzeri bir script oluştur
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
$action = New-ScheduledTaskAction -Execute "python" -Argument "benchmark_databases.py --full"
Register-ScheduledTask -TaskName "DailyBenchmark" -Trigger $trigger -Action $action
```

## Notlar

- Her test 3 kez çalıştırılır, ortalama alınır (daha güvenilir sonuçlar)
- Memory ölçümü test öncesi/sonrası delta'dır
- Tüm testler aynı veri seti üzerinde çalışır (1,563 segment, 3,452 edge)
- JSON sonuçları timestamp içerir, geçmiş sonuçları karşılaştırabilirsin

## İletişim

Sorular için: emiralibulutt@gmail.com
Proje: Traffic Flow Analysis & GNN
