# HERE Traffic Flow Pipeline# HERE Traffic Flow Pipeline# HERE Traffic Flow Pipeline# HERE Traffic Flow → Neo4j → GNN Pipeline#  HERE Traffic Flow - GNN Analysis Pipeline# HERE Flow v7 → Neo4j GNN Pipeline



Real-time traffic monitoring system integrating HERE Traffic API with Neo4j graph database. Optimized for Graph Neural Network (GNN) applications with automated topology management.



## Quick StartReal-time traffic monitoring system integrating HERE Traffic API with Neo4j graph database. Optimized for Graph Neural Network (GNN) applications with automated topology management.



### Prerequisites



- Neo4j Desktop 2.0+ (database named `ict`)## Quick StartReal-time traffic monitoring system integrating HERE Traffic API with Neo4j graph database. Optimized for Graph Neural Network (GNN) applications with automated topology management.

- Python 3.10+

- HERE API Key ([get free key](https://platform.here.com/))



### Installation### Prerequisites



```powershell

# Install dependencies

pip install -r requirements.txt- Neo4j Desktop 2.0+ (database named `ict`)## Quick Startİstanbul trafik verilerini HERE API'den çekip Neo4j'ye yükleyen ve GNN/GCN için hazırlayan otomatik pipeline.



# Configure environment- Python 3.10+

# Edit config/.env with your credentials

```- HERE API Key ([get free key](https://platform.here.com/))



### Run Pipeline



**Single Execution:**### Installation### Prerequisites

```powershell

python run_pipeline.py

```

```powershell

**Continuous Monitoring:**

```powershell# Install dependencies

python run_loop.py

```pip install -r requirements.txt- Neo4j Desktop 2.0+ (database named `ict`)## 🚀 Hızlı BaşlangıçProfesyonel trafik akış analizi ve Graph Neural Network (GNN) pipeline'ı.> ** YENİ:** [GNN/GCN-Hazır Veri Hattı →](mvp/QUICKSTART.md) Graph Neural Network modelleri için 5 dakikada hazırlık!



**Clean Database:**

```powershell

python clean_all.py# Configure environment- Python 3.10+

```

# Edit config/.env with your credentials

## What It Does

```- HERE API Key ([get free key](https://platform.here.com/))

1. **Fetches** real-time traffic data from HERE API

2. **Processes** GeoJSON and creates timestamped archives

3. **Aggregates** time series data for analysis

4. **Loads** segments and measurements to Neo4j### Run Pipeline

5. **Builds** spatial topology (CONNECTS_TO relationships)



### First Run

Duration: 3-5 minutes (includes topology creation)**Single Execution:**### Installation### 1. Tek Seferlik Çalıştır



### Subsequent Runs```powershell

Duration: 20-30 seconds (topology skipped automatically)

python run_pipeline.py

## Project Structure

```

```

HERE V6/```powershell

├── run_pipeline.py              # Single execution

├── run_loop.py                  # Automated loop**Continuous Monitoring:**

├── clean_all.py                 # Database cleanup

├── config/```powershell# Install dependencies

│   └── .env                     # Configuration

├── src/python run_loop.py

│   ├── pipeline/                # Data acquisition and processing

│   ├── neo4j/                   # Database loading```pip install -r requirements.txt```bash##  Proje YapısıBu proje, HERE Traffic Flow API v7 verilerini Neo4j graf veritabanına yükleyerek Graph Neural Network (GNN) analizleri için hazırlar.

│   ├── gnn/                     # Topology management

│   └── visualization/           # Map rendering

├── data/

│   ├── timeseries.parquet       # Aggregated time series**Clean Database:**

│   └── edges_static.geojson     # Segment geometries

├── archive/```powershell

│   └── flow_*.geojson           # Timestamped snapshots

└── docs/guides/                 # Detailed documentationpython clean_all.py# Configure environmentpython run_pipeline.py

```

```

## Configuration

# Edit config/.env with your credentials

Edit `config/.env`:

## What It Does

```properties

# HERE API``````

HERE_API_KEY=your_api_key_here

BBOX=30.4000,39.7000,30.7500,39.86001. **Fetches** real-time traffic data from HERE API



# Neo4j Connection2. **Processes** GeoJSON and creates timestamped archives

NEO4J_URI=neo4j://127.0.0.1:7687

NEO4J_USER=neo4j3. **Aggregates** time series data for analysis

NEO4J_PASS=your_password

NEO4J_DATABASE=ict4. **Loads** segments and measurements to Neo4j### Run Pipeline



# Pipeline Settings5. **Builds** spatial topology (CONNECTS_TO relationships)

PIPELINE_INTERVAL_MIN=15

TIMEZONE=Europe/Istanbul

MAX_ARCHIVES=500

```### First Run



## Database SchemaDuration: 3-5 minutes (includes topology creation)**Single Execution:****Ne yapar?**```---



### Nodes



**Segment** - Road segments with geometry### Subsequent Runs```powershell

- Properties: `segment_id`, `lat`, `lon`, `road_name`, `direction`, `length_m`

- Index: `segment_id` (unique)Duration: 20-30 seconds (topology skipped automatically)



**Measure** - Traffic measurementspython run_pipeline.py- HERE API'den trafik verisi çeker

- Properties: `timestamp`, `speed_kmh`, `confidence`, `jam_factor`, `free_flow_kmh`

- Index: `timestamp`## Project Structure



### Relationships```



**HAS_MEASURE** - Links segments to measurements```

- Pattern: `(Segment)-[:HAS_MEASURE]->(Measure)`

HERE V6/- Neo4j'ye yüklerHERE V6/

**CONNECTS_TO** - Spatial topology for GNN

- Pattern: `(Segment)-[:CONNECTS_TO {distance_m}]->(Segment)`├── run_pipeline.py              # Single execution

- Threshold: 12 meters proximity

- Created automatically on first run├── run_loop.py                  # Automated loop**Continuous Monitoring:**



## Verification├── clean_all.py                 # Database cleanup



### Neo4j Browser (http://localhost:7474)├── config/```powershell- Koordinatları çıkarır



```cypher│   └── .env                     # Configuration

// Count segments

MATCH (s:Segment) RETURN count(s)├── src/python run_loop.py



// Count measurements│   ├── pipeline/                # Data acquisition and processing

MATCH (m:Measure) RETURN count(m)

│   ├── neo4j/                   # Database loading```- CONNECTS_TO bağlantılarını oluşturur├──  src/                         # Kaynak kodları##  Mevcut Veri Durumu

// Check topology

MATCH ()-[r:CONNECTS_TO]->() RETURN count(r)│   ├── gnn/                     # Topology management



// View recent traffic│   └── visualization/           # Map rendering

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)

RETURN s.road_name, m.speed_kmh, m.timestamp├── data/

ORDER BY m.timestamp DESC

LIMIT 20│   ├── timeseries.parquet       # Aggregated time series**Clean Database:**

```

│   └── edges_static.geojson     # Segment geometries

### Expected Results

├── archive/```powershell

- Segments: 1,500-2,000 (depends on BBOX)

- Measures: Growing with each iteration│   └── flow_*.geojson           # Timestamped snapshots

- CONNECTS_TO: 2,500-3,000 (created once)

└── docs/guides/                 # Detailed documentationpython clean_all.py---│   ├── pipeline/                   # Pipeline scriptleri

## Visualization

```

### Static Map

```

Open `map.html` in browser - color-coded traffic flow with segment metadata.

## Configuration

### Live Dashboard



```powershell

python src/visualization/12_simple_web_server.pyEdit `config/.env`:

```

## What It Does

Access: http://localhost:5000

```properties

Features:

- Auto-refresh with latest Neo4j data# HERE API### 2. Otomatik Döngü (1 Dakikada Bir)│   │   ├── 01_fetch_here_flow.py   # HERE API veri çekme **Neo4j Veritabanında:**

- Interactive segment selection

- Timestamp navigationHERE_API_KEY=your_api_key_here

- REST API endpoints

BBOX=30.4000,39.7000,30.7500,39.86001. **Fetches** real-time traffic data from HERE API

## Performance



| Operation | Duration | Notes |

|-----------|----------|-------|# Neo4j Connection2. **Processes** GeoJSON and creates timestamped archives

| First pipeline run | 3-5 min | Includes topology creation |

| Subsequent runs | 20-30 sec | Topology skipped |NEO4J_URI=neo4j://127.0.0.1:7687

| Loop iteration (15-min interval) | 20-30 sec | After initial setup |

| Topology verification | 2-3 sec | Automatic check |NEO4J_USER=neo4j3. **Aggregates** time series data for analysis



### OptimizationNEO4J_PASS=your_password



Smart topology management:NEO4J_DATABASE=ict4. **Loads** segments and measurements to Neo4j```bash│   │   ├── 02_render_flow_map.py   # Harita render- **2,366** yol segmenti (Segment nodes)

- Creates CONNECTS_TO once (first run)

- Skips on subsequent runs (saves 5-30 minutes)

- Verifies automatically via `check_topology.py`

- Manual rebuild available if needed# Pipeline Settings5. **Builds** spatial topology (CONNECTS_TO relationships)



## Common WorkflowsPIPELINE_INTERVAL_MIN=15



### Daily MonitoringTIMEZONE=Europe/Istanbulpython run_loop.py



```powershellMAX_ARCHIVES=500

# Morning: Start automated collection

python run_loop.py```### First Run



# Evening: Stop with Ctrl+C

```

## Database SchemaDuration: 3-5 minutes (includes topology creation)```│   │   ├── 04_run_loop.py          # Loop çalıştırıcı- **6,811** trafik ölçümü (Measure nodes)

### Weekly Maintenance



```powershell

# Check topology health### Nodes

python src/gnn/check_topology.py



# Verify GNN readiness

python src/gnn/test_gnn_readiness.py**Segment** - Road segments with geometry### Subsequent Runs

```

- Properties: `segment_id`, `lat`, `lon`, `road_name`, `direction`, `length_m`

### Change Coverage Area

- Index: `segment_id` (unique)Duration: 20-30 seconds (topology skipped automatically)

```powershell

# 1. Update BBOX in config/.env

# 2. Clean database

python clean_all.py**Measure** - Traffic measurements**Ne yapar?**│   │   ├── 05_build_timeseries.py  # Timeseries oluşturma- **85,350** topoloji ilişkisi (CONNECTS_TO relationships, 12m threshold)



# 3. Restart pipeline- Properties: `timestamp`, `speed_kmh`, `confidence`, `jam_factor`, `free_flow_kmh`

python run_pipeline.py

```- Index: `timestamp`## Project Structure



## Troubleshooting



### Authentication Failed### Relationships- `.env` dosyasındaki `PIPELINE_INTERVAL_MIN` ayarına göre sürekli çalışır



**Cause**: Wrong password or database name



**Fix**:**HAS_MEASURE** - Links segments to measurements```

1. Verify database name in Neo4j Desktop matches `NEO4J_DATABASE` in config/.env

2. Check password matches `NEO4J_PASS`- Pattern: `(Segment)-[:HAS_MEASURE]->(Measure)`

3. Restart Neo4j database

HERE V6/- Her iterasyonda yukarıdaki tüm adımları tekrarlar│   │   └── 08_auto_pipeline.py     # Otomatik pipeline- **4** zaman dilimi (TS15 time buckets)

### No Traffic Data

**CONNECTS_TO** - Spatial topology for GNN

**Cause**: Invalid API key or BBOX

- Pattern: `(Segment)-[:CONNECTS_TO {distance_m}]->(Segment)`├── run_pipeline.py              # Single execution

**Fix**:

1. Test API: `python test_api.py`- Threshold: 12 meters proximity

2. Verify BBOX format: `lon_min,lat_min,lon_max,lat_max`

3. Check area has traffic coverage (urban areas better)- Created automatically on first run├── run_loop.py                  # Automated loop- Yeni segmentler için bağlantıları günceller



### Topology Not Created



**Cause**: Normal behavior after first run (skipped for performance)## Verification├── clean_all.py                 # Database cleanup



**Verify**:

```powershell

python src/gnn/check_topology.py### Neo4j Browser (http://localhost:7474)├── config/│   ├── neo4j/                      # Neo4j yönetimi

```



**Force Rebuild** (if needed):

```powershell```cypher│   └── .env                     # Configuration

python src/gnn/run_step1_enhance_schema.py

python src/gnn/run_step2_build_connects_to.py// Count segments

```

MATCH (s:Segment) RETURN count(s)├── src/**Durdurmak için:** `Ctrl + C`

### Slow Performance



**Cause**: Large BBOX or short interval

// Count measurements│   ├── pipeline/                # Data acquisition and processing

**Fix**:

1. Reduce BBOX size for testingMATCH (m:Measure) RETURN count(m)

2. Increase `PIPELINE_INTERVAL_MIN` (production: 15-30 min)

3. Verify Neo4j has adequate memory (check neo4j.conf)│   ├── neo4j/                   # Database loading│   │   ├── neo4j_loader.py         # Neo4j loader modülü---



## Documentation// Check topology



Comprehensive guides in `docs/guides/`:MATCH ()-[r:CONNECTS_TO]->() RETURN count(r)│   ├── gnn/                     # Topology management



- **PIPELINE_README.md** - Complete system architecture, database schema, performance tuning

- **QUICKSTART.md** - Installation, first run, verification steps

- **TOPOLOGY_MANAGEMENT.md** - Spatial relationships, optimization, troubleshooting// View recent traffic│   └── visualization/           # Map rendering---

- **SMART_PIPELINE_SUMMARY.md** - Optimization techniques, benchmarks, best practices

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)

## GNN Integration

RETURN s.road_name, m.speed_kmh, m.timestamp├── data/

System produces GNN-ready graph data:

ORDER BY m.timestamp DESC

### Validation

LIMIT 20│   ├── timeseries.parquet       # Aggregated time series│   │   ├── 06_auto_load_to_neo4j.py

```powershell

python src/gnn/test_gnn_readiness.py```

```

│   └── edges_static.geojson     # Segment geometries

Checks:

1. Node count (>100 segments)### Expected Results

2. Coordinate coverage (100%)

3. Topology connectivity (CONNECTS_TO)├── archive/## ⚙️ Ayarlar

4. Feature richness (measurements)

5. Temporal depth (multiple timestamps)- Segments: 1,500-2,000 (depends on BBOX)



Target: 100% readiness score- Measures: Growing with each iteration│   └── flow_*.geojson           # Timestamped snapshots



### Query Examples- CONNECTS_TO: 2,500-3,000 (created once)



**Traffic hotspots:**└── docs/guides/                 # Detailed documentation│   │   └── 07_silent_load_to_neo4j.py## 🚀 Kurulum ve Başlangıç

```cypher

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)## Visualization

WHERE m.jam_factor > 5

WITH s, count(m) AS congestion_count```

WHERE congestion_count > 10

RETURN s.road_name, congestion_count### Static Map

ORDER BY congestion_count DESC

````config/.env` dosyasını düzenleyin:



**Traffic propagation:**Open `map.html` in browser - color-coded traffic flow with segment metadata.

```cypher

MATCH path = (s1:Segment)-[:CONNECTS_TO*1..3]->(s2:Segment)## Configuration

WHERE s1.segment_id = 'start_segment_id'

WITH s2, length(path) AS hops### Live Dashboard

MATCH (s2)-[:HAS_MEASURE]->(m:Measure)

WHERE m.timestamp > datetime() - duration({minutes: 30})│   ├── gnn/                        # GNN hazırlık

RETURN s2.road_name, hops, avg(m.jam_factor) AS avg_jam

ORDER BY hops, avg_jam DESC```powershell

```

python src/visualization/12_simple_web_server.pyEdit `config/.env`:

## Technology Stack

```

- **Data Source**: HERE Traffic Flow API v7

- **Database**: Neo4j Community Edition```env

- **Language**: Python 3.10+

- **Key Libraries**: neo4j, pandas, shapely, flaskAccess: http://localhost:5000

- **Visualization**: Leaflet.js, Neo4j Browser

```properties

## API Usage

Features:

With 15-minute interval:

- Calls per day: 96- Auto-refresh with latest Neo4j data# HERE API# HERE API│   │   ├── test_gnn_readiness.py   # GNN hazırlık testi### 1️ Virtual Environment'ı Aktive Edin

- Calls per month: ~2,880

- Well within HERE free tier (250,000/month)- Interactive segment selection



## Support- Timestamp navigationHERE_API_KEY=your_api_key_here



- Issues: Check `logs/` directory for error details- REST API endpoints

- Testing: `test_api.py`, `test_neo4j_connection.py`

- Inspection: Neo4j Browser (http://localhost:7474)BBOX=30.4000,39.7000,30.7500,39.8600HERE_API_KEY=your_api_key_here

- Documentation: `docs/guides/` for detailed references

## Performance

## License



Project for traffic monitoring and GNN research. Ensure HERE API terms of service compliance.

| Operation | Duration | Notes |

## Summary

|-----------|----------|-------|# Neo4j ConnectionBBOX=30.4000,39.7000,30.7500,39.8600│   │   ├── run_step1_enhance_schema.py

Two-command system for real-time traffic monitoring:

- `python run_pipeline.py` - Single execution| First pipeline run | 3-5 min | Includes topology creation |

- `python run_loop.py` - Continuous monitoring

| Subsequent runs | 20-30 sec | Topology skipped |NEO4J_URI=neo4j://127.0.0.1:7687

Optimized for:

- Fast iterations (20-30 seconds after initial setup)| Loop iteration (15-min interval) | 20-30 sec | After initial setup |

- GNN-ready data (spatial topology + temporal features)

- Production deployment (automated cleanup, error handling)| Topology verification | 2-3 sec | Automatic check |NEO4J_USER=neo4j

- Scalability (efficient batching, smart skip logic)



Start collecting traffic data in under 10 minutes.

### OptimizationNEO4J_PASS=your_password



Smart topology management:NEO4J_DATABASE=ict# Neo4j│   │   ├── run_step2_build_connects_to.py```powershell

- Creates CONNECTS_TO once (first run)

- Skips on subsequent runs (saves 5-30 minutes)

- Verifies automatically via `check_topology.py`

- Manual rebuild available if needed# Pipeline SettingsNEO4J_URI=neo4j://127.0.0.1:7687



## Common WorkflowsPIPELINE_INTERVAL_MIN=15



### Daily MonitoringTIMEZONE=Europe/IstanbulNEO4J_USER=neo4j│   │   ├── check_topology.py       # Topoloji kontrolü# PowerShell execution policy'yi ayarlayın (ilk kez gerekli)



```powershellMAX_ARCHIVES=500

# Morning: Start automated collection

python run_loop.py```NEO4J_PASS=your_password



# Evening: Stop with Ctrl+C

```

## Database SchemaNEO4J_DATABASE=ict│   │   ├── ensure_topology.py      # Akıllı topoloji yönetimiSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

### Weekly Maintenance



```powershell

# Check topology health### Nodes

python src/gnn/check_topology.py



# Verify GNN readiness

python src/gnn/test_gnn_readiness.py**Segment** - Road segments with geometry# Pipeline│   │   ├── 04_generate_features.py # Feature engineering

```

- Properties: `segment_id`, `lat`, `lon`, `road_name`, `direction`, `length_m`

### Change Coverage Area

- Index: `segment_id` (unique)PIPELINE_INTERVAL_MIN=1    # Döngü için interval (dakika)

```powershell

# 1. Update BBOX in config/.env

# 2. Clean database

python clean_all.py**Measure** - Traffic measurementsCONNECT_THRESHOLD=12       # CONNECTS_TO mesafe eşiği (metre)│   │   └── 05_export_pyg.py        # PyTorch Geometric export# Virtual environment'ı aktive edin



# 3. Restart pipeline- Properties: `timestamp`, `speed_kmh`, `confidence`, `jam_factor`, `free_flow_kmh`

python run_pipeline.py

```- Index: `timestamp````



## Troubleshooting



### Authentication Failed### Relationships│   └── visualization/              # Görselleştirme.\.venv\Scripts\Activate.ps1



**Cause**: Wrong password or database name



**Fix**:**HAS_MEASURE** - Links segments to measurements---

1. Verify database name in Neo4j Desktop matches `NEO4J_DATABASE` in config/.env

2. Check password matches `NEO4J_PASS`- Pattern: `(Segment)-[:HAS_MEASURE]->(Measure)`

3. Restart Neo4j database

│       ├── 09_neo4j_map_viewer.py```

### No Traffic Data

**CONNECTS_TO** - Spatial topology for GNN

**Cause**: Invalid API key or BBOX

- Pattern: `(Segment)-[:CONNECTS_TO {distance_m}]->(Segment)`## 📊 Veritabanı Durumu Kontrolü

**Fix**:

1. Test API: `python test_api.py`- Threshold: 12 meters proximity

2. Verify BBOX format: `lon_min,lat_min,lon_max,lat_max`

3. Check area has traffic coverage (urban areas better)- Created automatically on first run│       ├── 10_neo4j_interactive_viewer.py



### Topology Not Created



**Cause**: Normal behavior after first run (skipped for performance)## Verification```bash



**Verify**:

```powershell

python src/gnn/check_topology.py### Neo4j Browser (http://localhost:7474)python src/gnn/test_gnn_readiness.py│       ├── 11_web_server.py**Aktif olduğunda** terminal başında `(.venv)` görünecektir.

```



**Force Rebuild** (if needed):

```powershell```cypher```

python src/gnn/run_step1_enhance_schema.py

python src/gnn/run_step2_build_connects_to.py// Count segments

```

MATCH (s:Segment) RETURN count(s)│       └── 12_simple_web_server.py

### Slow Performance



**Cause**: Large BBOX or short interval

// Count measurements**Sonuç:**

**Fix**:

1. Reduce BBOX size for testingMATCH (m:Measure) RETURN count(m)

2. Increase `PIPELINE_INTERVAL_MIN` (production: 15-30 min)

3. Verify Neo4j has adequate memory (check neo4j.conf)- Segment sayısı├── 📂 data/                        # Veri dosyaları### 2️⃣ Gerekli Paketlerin Yüklü Olduğunu Kontrol Edin



## Documentation// Check topology



Comprehensive guides in `docs/guides/`:MATCH ()-[r:CONNECTS_TO]->() RETURN count(r)- Measure sayısı  



- **PIPELINE_README.md** - Complete system architecture, database schema, performance tuning

- **QUICKSTART.md** - Installation, first run, verification steps

- **TOPOLOGY_MANAGEMENT.md** - Spatial relationships, optimization, troubleshooting// View recent traffic- CONNECTS_TO bağlantı sayısı│   ├── edges_static.geojson        # Statik segment verileri

- **SMART_PIPELINE_SUMMARY.md** - Optimization techniques, benchmarks, best practices

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)

## GNN Integration

RETURN s.road_name, m.speed_kmh, m.timestamp- GNN Hazırlık Skoru (0-100%)

System produces GNN-ready graph data:

ORDER BY m.timestamp DESC

### Validation

LIMIT 20│   ├── timeseries.parquet/csv      # Zaman serisi verileri```powershell

```powershell

python src/gnn/test_gnn_readiness.py```

```

---

Checks:

1. Node count (>100 segments)### Expected Results

2. Coordinate coverage (100%)

3. Topology connectivity (CONNECTS_TO)│   ├── features_window.csv         # Normalize özelliklerpip install neo4j shapely pyproj scikit-learn python-dateutil

4. Feature richness (measurements)

5. Temporal depth (multiple timestamps)- Segments: 1,500-2,000 (depends on BBOX)



Target: 100% readiness score- Measures: Growing with each iteration## 🧹 Veritabanını Temizle



### Query Examples- CONNECTS_TO: 2,500-3,000 (created once)



**Traffic hotspots:**│   └── pyg_graph.npz               # PyTorch Geometric tensörler```

```cypher

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)## Visualization

WHERE m.jam_factor > 5

WITH s, count(m) AS congestion_count```bash

WHERE congestion_count > 10

RETURN s.road_name, congestion_count### Static Map

ORDER BY congestion_count DESC

```python clean_all.py├── 📂 archive/                     # GeoJSON arşivi



**Traffic propagation:**Open `map.html` in browser - color-coded traffic flow with segment metadata.

```cypher

MATCH path = (s1:Segment)-[:CONNECTS_TO*1..3]->(s2:Segment)```

WHERE s1.segment_id = 'start_segment_id'

WITH s2, length(path) AS hops### Live Dashboard

MATCH (s2)-[:HAS_MEASURE]->(m:Measure)

WHERE m.timestamp > datetime() - duration({minutes: 30})│   └── flow_YYYYMMDD_HHMM.geojson  # Zaman damgalı flow verileri### 3️⃣ Neo4j Veritabanının Çalıştığından Emin Olun

RETURN s2.road_name, hops, avg(m.jam_factor) AS avg_jam

ORDER BY hops, avg_jam DESC```powershell

```

python src/visualization/12_simple_web_server.py**Uyarı:** Tüm Neo4j verileri ve arşiv dosyaları silinir!

## Technology Stack

```

- **Data Source**: HERE Traffic Flow API v7

- **Database**: Neo4j Community Edition├── 📂 logs/                        # Log dosyaları

- **Language**: Python 3.10+

- **Key Libraries**: neo4j, pandas, shapely, flaskAccess: http://localhost:5000

- **Visualization**: Leaflet.js, Neo4j Browser

---

## API Usage

Features:

With 15-minute interval:

- Calls per day: 96- Auto-refresh with latest Neo4j data│   └── pipeline_YYYYMMDD.log- Neo4j Desktop'ı açın ve veritabanınızı başlatın

- Calls per month: ~2,880

- Well within HERE free tier (250,000/month)- Interactive segment selection



## Support- Timestamp navigation## 📁 Proje Yapısı



- Issues: Check `logs/` directory for error details- REST API endpoints

- Testing: `test_api.py`, `test_neo4j_connection.py`

- Inspection: Neo4j Browser (http://localhost:7474)├── 📂 config/                      # Konfigürasyon- Varsayılan bağlantı: `bolt://localhost:7687`

- Documentation: `docs/guides/` for detailed references

## Performance

## License

```

Project for traffic monitoring and GNN research. Ensure HERE API terms of service compliance.

| Operation | Duration | Notes |

## Summary

|-----------|----------|-------|├── run_pipeline.py          # ⭐ Tek seferlik pipeline│   ├── .env                        # Ana konfig (Neo4j, HERE API)- Kullanıcı: `neo4j`

Two-command system for real-time traffic monitoring:

- `python run_pipeline.py` - Single execution| First pipeline run | 3-5 min | Includes topology creation |

- `python run_loop.py` - Continuous monitoring

| Subsequent runs | 20-30 sec | Topology skipped |├── run_loop.py              # ⭐ Otomatik döngü

Optimized for:

- Fast iterations (20-30 seconds after initial setup)| Loop iteration (15-min interval) | 20-30 sec | After initial setup |

- GNN-ready data (spatial topology + temporal features)

- Production deployment (automated cleanup, error handling)| Topology verification | 2-3 sec | Automatic check |├── clean_all.py             # Veritabanı temizleme│   ├── .env.example                # Örnek konfig- Şifre: `.env` dosyasında tanımlı

- Scalability (efficient batching, smart skip logic)



Start collecting traffic data in under 10 minutes.

### Optimization│



Smart topology management:├── config/│   ├── requirements.txt            # Python bağımlılıkları

- Creates CONNECTS_TO once (first run)

- Skips on subsequent runs (saves 5-30 minutes)│   └── .env                 # Ayarlar

- Verifies automatically via `check_topology.py`

- Manual rebuild available if needed││   ├── setup_windows_task.ps1      # Windows Task Scheduler---



## Common Workflows├── src/



### Daily Monitoring│   ├── pipeline/            # Veri çekme & işleme│   └── cypher/                     # Cypher sorguları (ileride)



```powershell│   ├── neo4j/               # Neo4j yükleme

# Morning: Start automated collection

python run_loop.py│   ├── gnn/                 # GNN hazırlık├── 📂 tests/                       # Test scriptleri## 📝 Temel Komutlar



# Evening: Stop with Ctrl+C│   └── visualization/       # Harita görselleştirme

```

││   ├── test_api.py                 # HERE API testi

### Weekly Maintenance

├── data/

```powershell

# Check topology health│   ├── timeseries.parquet   # Zaman serisi│   └── test_neo4j_connection.py    # Neo4j bağlantı testi### Neo4j Schema'yı Oluştur (İlk Kez)

python src/gnn/check_topology.py

│   └── edges_static.geojson # Statik segment verileri

# Verify GNN readiness

python src/gnn/test_gnn_readiness.py│├── 📂 docs/                        # Dokümantasyon

```

└── archive/                 # GeoJSON arşiv

### Change Coverage Area

```│   └── guides/                     # Kılavuzlar```powershell

```powershell

# 1. Update BBOX in config/.env

# 2. Clean database

python clean_all.py---│       ├── QUICKSTART.mdpython neo4j_gnn_ingest.py --init-schema



# 3. Restart pipeline

python run_pipeline.py

```## 🎯 Örnek Kullanım│       ├── PIPELINE_README.md```



## Troubleshooting



### Authentication Failed### Senaryo 1: İlk Kurulum│       ├── TOPOLOGY_MANAGEMENT.md



**Cause**: Wrong password or database name```bash



**Fix**:# 1. Ayarları düzenle│       └── SMART_PIPELINE_SUMMARY.mdBu komut:

1. Verify database name in Neo4j Desktop matches `NEO4J_DATABASE` in config/.env

2. Check password matches `NEO4J_PASS`notepad config\.env

3. Restart Neo4j database

├── run_pipeline.py                 # 🚀 Ana entrypoint- `Segment` node'ları için unique constraint oluşturur

### No Traffic Data

# 2. Tek seferlik çalıştır

**Cause**: Invalid API key or BBOX

python run_pipeline.py└── README.md                       # Bu dosya- `TS15` zaman bucket'ları için unique constraint oluşturur

**Fix**:

1. Test API: `python test_api.py`

2. Verify BBOX format: `lon_min,lat_min,lon_max,lat_max`

3. Check area has traffic coverage (urban areas better)# 3. Sonucu kontrol et```- `Measure` için composite index oluşturur



### Topology Not Createdpython src/gnn/test_gnn_readiness.py



**Cause**: Normal behavior after first run (skipped for performance)```



**Verify**:

```powershell

python src/gnn/check_topology.py### Senaryo 2: Sürekli Veri Toplama---### Statik Segment Verilerini Yükle

```

```bash

**Force Rebuild** (if needed):

```powershell# Otomatik döngü başlat (1 dakikada bir)

python src/gnn/run_step1_enhance_schema.py

python src/gnn/run_step2_build_connects_to.pypython run_loop.py

```

## 🚀 Hızlı Başlangıç```powershell

### Slow Performance

# Başka terminal'de durumu izle

**Cause**: Large BBOX or short interval

python src/gnn/test_gnn_readiness.pypython neo4j_gnn_ingest.py --load-segments data/edges_static.geojson

**Fix**:

1. Reduce BBOX size for testing```

2. Increase `PIPELINE_INTERVAL_MIN` (production: 15-30 min)

3. Verify Neo4j has adequate memory (check neo4j.conf)### 1️⃣ Kurulum```



## Documentation### Senaryo 3: Temiz Başlangıç



Comprehensive guides in `docs/guides/`:```bash



- **PIPELINE_README.md** - Complete system architecture, database schema, performance tuning# Veritabanını temizle

- **QUICKSTART.md** - Installation, first run, verification steps

- **TOPOLOGY_MANAGEMENT.md** - Spatial relationships, optimization, troubleshootingpython clean_all.py```bashBu komut:

- **SMART_PIPELINE_SUMMARY.md** - Optimization techniques, benchmarks, best practices



## GNN Integration

# Yeniden başlat# Python bağımlılıklarını yükle- GeoJSON'dan yol segmentlerini okur

System produces GNN-ready graph data:

python run_loop.py

### Validation

```pip install -r config/requirements.txt- Her segment için `Segment` node'u oluşturur

```powershell

python src/gnn/test_gnn_readiness.py

```

---- Segment özellikleri: ID, HERE segment ID, OSM way ID, FRC, uzunluk, isim, geometri

Checks:

1. Node count (>100 segments)

2. Coordinate coverage (100%)

3. Topology connectivity (CONNECTS_TO)## 📈 GNN/GCN İçin Veri Formatı# Neo4j Desktop'ı indir ve başlat

4. Feature richness (measurements)

5. Temporal depth (multiple timestamps)



Target: 100% readiness scorePipeline otomatik olarak oluşturur:# https://neo4j.com/download/### Segment Yakınlık İlişkilerini Oluştur



### Query Examples- **Nodes**: Segment düğümleri (koordinatlı)



**Traffic hotspots:**- **Edges**: CONNECTS_TO ilişkileri (12m eşik)

```cypher

MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)- **Features**: Hız, jam factor, temporal encoding

WHERE m.jam_factor > 5

WITH s, count(m) AS congestion_count- **Timeseries**: Parquet format# config/.env dosyasını düzenle```powershell

WHERE congestion_count > 10

RETURN s.road_name, congestion_count

ORDER BY congestion_count DESC

```---cp config/.env.example config/.envpython neo4j_gnn_ingest.py --build-next-to --threshold-m 3



**Traffic propagation:**

```cypher

MATCH path = (s1:Segment)-[:CONNECTS_TO*1..3]->(s2:Segment)## 🆘 Sorun Giderme# HERE_API_KEY, NEO4J_PASS vs. ayarla```

WHERE s1.segment_id = 'start_segment_id'

WITH s2, length(path) AS hops

MATCH (s2)-[:HAS_MEASURE]->(m:Measure)

WHERE m.timestamp > datetime() - duration({minutes: 30})**Neo4j bağlantı hatası?**```

RETURN s2.road_name, hops, avg(m.jam_factor) AS avg_jam

ORDER BY hops, avg_jam DESC```bash

```

# Bağlantıyı test etBu komut:

## Technology Stack

python tests/test_neo4j_connection.py

- **Data Source**: HERE Traffic Flow API v7

- **Database**: Neo4j Community Edition### 2️⃣ İlk Çalıştırma- Segment başlangıç ve bitiş noktalarını analiz eder

- **Language**: Python 3.10+

- **Key Libraries**: neo4j, pandas, shapely, flask# config/.env'deki ayarları kontrol et

- **Visualization**: Leaflet.js, Neo4j Browser

```- 12 metre içinde birbirine yakın segmentler arasında `CONNECTS_TO` ilişkisi oluşturur (run_step2_build_connects_to.py)

## API Usage



With 15-minute interval:

- Calls per day: 96**Pipeline çok yavaş?**```bash- GNN için komşuluk matrisi sağlar

- Calls per month: ~2,880

- Well within HERE free tier (250,000/month)- `PIPELINE_INTERVAL_MIN` değerini artırın (örn: 5 veya 15 dakika)



## Support- `CONNECT_THRESHOLD` değerini artırın (daha az bağlantı)# Tek seferlik pipeline çalıştır



- Issues: Check `logs/` directory for error details

- Testing: `test_api.py`, `test_neo4j_connection.py`

- Inspection: Neo4j Browser (http://localhost:7474)---python run_pipeline.py### Trafik Ölçümlerini Yükle

- Documentation: `docs/guides/` for detailed references



## License

## 📝 Geliştirme

Project for traffic monitoring and GNN research. Ensure HERE API terms of service compliance.



## Summary

Detaylı dokümantasyon için `docs/` klasörüne bakın:# İlk çalıştırmada:#### Tek Dosya:

Two-command system for real-time traffic monitoring:

- `python run_pipeline.py` - Single execution- `docs/QUICKSTART.md` - Hızlı başlangıç kılavuzu

- `python run_loop.py` - Continuous monitoring

- `docs/PIPELINE_README.md` - Pipeline detayları# - HERE API'den veri çeker```powershell

Optimized for:

- Fast iterations (20-30 seconds after initial setup)- `docs/AUTOMATION_GUIDE.md` - Otomasyon ayarları

- GNN-ready data (spatial topology + temporal features)

- Production deployment (automated cleanup, error handling)# - Neo4j'ye yüklerpython neo4j_gnn_ingest.py --load-measure flow_20251003_1332.geojson --ts 2025-10-03T13:32:00Z

- Scalability (efficient batching, smart skip logic)

---

Start collecting traffic data in under 10 minutes.

# - Topoloji oluşturur (~10 dk)```

## 📊 İstatistikler

# - GNN hazırlık yapar

- **Tek iterasyon süresi:** ~10-20 saniye

- **CONNECTS_TO oluşturma:** ~5-10 saniye (segment sayısına göre)```#### Archive'deki Tüm Dosyalar:

- **API rate limit:** HERE Free tier - 250,000 transaction/ay

```powershell

---

### 3️⃣ Sürekli ÇalıştırmaGet-ChildItem archive/*.geojson | ForEach-Object { 

## 🎉 Başarı!

    python neo4j_gnn_ingest.py --load-measure $_.FullName 

Pipeline çalıştığında:

- ✅ `src/visualization/map.html` → Trafik haritası```bash}

- ✅ `data/timeseries.parquet` → Zaman serisi

- ✅ `archive/flow_*.geojson` → GeoJSON arşiv# Her 15 dakikada bir otomatik çalışsın```

- ✅ Neo4j → GNN-ready graph database

python run_pipeline.py --loop --interval 15

**GNN modelleme için hazırsınız!** 🚀

```Bu komut:

- Her segment için trafik ölçümlerini (hız, jamFactor, confidence vb.) yükler

---- `Measure` node'ları oluşturur

- `Segment -[:AT_TIME]-> Measure` ilişkisi kurar

## 📊 Ne Yapar?- `Measure -[:OF_WINDOW]-> TS15` zaman bucket ilişkisi kurar



### 🔄 Pipeline Akışı---



```## 🔧 Yapılandırma (.env Dosyası)

1. HERE API → Trafik verisi çek

2. GeoJSON   → Arşivle (archive/)`.env` dosyasında aşağıdaki ayarlar tanımlıdır:

3. Parquet   → Timeseries oluştur

4. Neo4j     → Graph database'e yükle```properties

5. GNN       → Topoloji + Features hazırla# HERE API

```HERE_API_KEY=RuTmm52lyY4vV72USiKVM38WF4wBG82TgxlLf22-kuo

BBOX=30.4000,39.7000,30.7500,39.8600

### 🧠 GNN Hazırlık

# Neo4j Bağlantısı

- **Spatial Topology**: 367,293 CONNECTS_TO ilişkisi (12m threshold)NEO4J_URI=bolt://localhost:7687

- **Node Features**: Speed, jamFactor, confidence, time featuresNEO4J_USER=neo4j

- **PyTorch Geometric**: Hazır NPZ formatında exportNEO4J_PASS=123456789

- **Akıllı Yönetim**: Topoloji bir kez oluşturulur, sürekli kullanılır

# Diğer Ayarlar

---SNAPSHOT_INTERVAL_MIN=1

TIMEZONE=Europe/Istanbul

## 🛠️ KomutlarMAX_ARCHIVES=500

```

### Pipeline

---

```bash

# Tek sefer çalıştır## 📂 Proje Yapısı

python run_pipeline.py

```

# Loop moduHERE V6/

python run_pipeline.py --loop --interval 15├── neo4j_gnn_ingest.py          # Ana script

├── 01_fetch_here_flow.py        # HERE API'den veri çekme

# HERE çekmeyi atla (sadece mevcut verileri yükle)├── 02_render_flow_map.py        # Görselleştirme

python run_pipeline.py --skip-fetch├── 04_run_loop.py               # Otomatik veri toplama döngüsü

├── 05_build_timeseries.py       # Zaman serisi oluşturma

# Detaylı log├── .env                         # Yapılandırma dosyası

python run_pipeline.py --verbose├── README.md                    # Bu dosya

```├── data/

│   ├── edges_static.geojson     # Statik yol segmentleri

### GNN Hazırlık│   ├── timeseries.csv           # Zaman serisi verileri

│   └── timeseries.jsonl         # JSON Lines formatında

```bash└── archive/

# Topoloji durumunu kontrol et    └── flow_*.geojson           # Geçmiş trafik snapshot'ları

python src/gnn/check_topology.py```



# GNN hazırlık testi---

python src/gnn/test_gnn_readiness.py

## 🔍 Neo4j'de Veri Sorgulama

# Feature engineering

python src/gnn/04_generate_features.pyNeo4j Browser'da (`http://localhost:7474`) şu sorguları çalıştırabilirsiniz:



# PyTorch Geometric export### Tüm Segmentleri Görüntüle

python src/gnn/05_export_pyg.py```cypher

```MATCH (s:Segment)

RETURN s

### GörselleştirmeLIMIT 50

```

```bash

# Interaktif harita### Bir Segmentin Trafik Ölçümlerini Görüntüle

python src/visualization/10_neo4j_interactive_viewer.py```cypher

MATCH (s:Segment)-[:AT_TIME]->(m:Measure)

# Web sunucuWHERE s.segmentId = 'your_segment_id_here'

python src/visualization/11_web_server.pyRETURN s, m

```ORDER BY m.ts

```

---

### Komşu Segmentleri Görüntüle (CONNECTS_TO İlişkileri - Topoloji)

## 📦 Çıktılar```cypher

MATCH (s1:Segment)-[:CONNECTS_TO]->(s2:Segment)

### Neo4j Graph DatabaseRETURN s1, s2, r.distance

LIMIT 100

- **18,920** Segment (yol parçaları)```

- **39,990** Measure (trafik ölçümleri)

- **367,293** CONNECTS_TO (spatial topoloji)### Zaman Serisi Analizi

- **39,990** HAS_MEASURE (zaman serisi bağlantıları)```cypher

MATCH (m:Measure)-[:OF_WINDOW]->(t:TS15)

### PyTorch GeometricWHERE t.bucket >= datetime('2025-10-03T00:00:00Z')

RETURN t.bucket, avg(m.speed) as avg_speed, avg(m.jamFactor) as avg_jam

`data/pyg_graph.npz` içeriği:ORDER BY t.bucket

- `edge_index`: (2, 367293) - Topoloji matrisi```

- `edge_attr`: (367293, 1) - Kenar özellikleri (distance)

- `x`: (T, 18920, 8) - Node features### En Yoğun 10 Segment

- `y`: (T, 18920, 1) - Target (speed prediction)```cypher

MATCH (s:Segment)-[:AT_TIME]->(m:Measure)

---RETURN s.segmentId, s.name, avg(m.jamFactor) as avg_jam

ORDER BY avg_jam DESC

## ⚙️ KonfigürasyonLIMIT 10

```

`config/.env` dosyası:

---

```ini

# HERE API## 🤖 GNN Hazırlığı

HERE_API_KEY=your_api_key_here

Neo4j'deki veriler artık GNN modelleri için hazır:

# Neo4j

NEO4J_URI=bolt://localhost:76871. **Node Features:** `Segment` özellikleri (uzunluk, FRC, koordinatlar)

NEO4J_USER=neo4j2. **Edge Features:** `CONNECTS_TO` ilişkileri (topoloji/graf yapısı, 12m threshold, distance_m özelliği)

NEO4J_PASS=your_password3. **Temporal Features:** `Measure` node'ları (zaman serisi özellikleri)

4. **Time Buckets:** `TS15` node'ları (15 dakikalık zaman dilimleri)

# GNN

CONNECT_THRESHOLD=12  # metre (spatial yakınlık)---



# Pipeline## 🛠️ Sorun Giderme

PIPELINE_INTERVAL_MIN=15  # dakika

```### PowerShell Script Çalıştırma Hatası

```powershell

---Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

```

## 📖 Dokümantasyon

### Neo4j Bağlantı Hatası

- [QUICKSTART.md](docs/guides/QUICKSTART.md) - Hızlı başlangıç kılavuzu- Neo4j Desktop'ın çalıştığını kontrol edin

- [TOPOLOGY_MANAGEMENT.md](docs/guides/TOPOLOGY_MANAGEMENT.md) - Akıllı topoloji yönetimi- `.env` dosyasındaki `NEO4J_PASS` şifresini kontrol edin

- [SMART_PIPELINE_SUMMARY.md](docs/guides/SMART_PIPELINE_SUMMARY.md) - Pipeline özet- Bağlantıyı test edin:

```powershell

---python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '123456789')); driver.verify_connectivity(); print('✅ Bağlantı başarılı!'); driver.close()"

```

## 🧪 Testler

### Eksik Paket Hatası

```bash```powershell

# HERE API testipip install neo4j shapely pyproj scikit-learn python-dateutil

python tests/test_api.py```



# Neo4j bağlantı testi---

python tests/test_neo4j_connection.py

## 📞 İletişim

# GNN hazırlık testi

python src/gnn/test_gnn_readiness.pySorularınız için: [Proje Sahibi]

```

---

---

## 📄 Lisans

## 🎯 Sırada Ne Var?

[Lisans Tipi]

- [ ] **Benchmark**: GNN vs Baseline modeller (ARIMA, HA)

- [ ] **GCN Model**: Graph Convolutional Network---

- [ ] **GAT Model**: Graph Attention Network

- [ ] **STGCN**: Spatio-Temporal GCN**Not:** Bu README, projenin mevcut durumunu ve temel kullanımını açıklar. Daha fazla detay için script dosyalarındaki docstring'lere bakabilirsiniz.

- [ ] **Dashboard**: Gerçek zamanlı monitoring

---

## 📝 Lisans

Bu proje eğitim amaçlıdır. HERE Traffic API kullanımı için kendi API key'inizi alın.

---

## 🤝 Katkı

Sorular ve öneriler için Issue açabilirsiniz.

---

**Oluşturulma:** Ekim 2025
**Python:** 3.10+
**Neo4j:** 5.x
**PyTorch Geometric:** 2.x
