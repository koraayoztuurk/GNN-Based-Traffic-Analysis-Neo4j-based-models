# Quick Start Guide#  Hızlı Başlangıç Komutları



## Prerequisites##  Temel Kurulum 



1. **Neo4j Desktop 2.0+**```powershell

   - Install from: https://neo4j.com/download/# 1. Execution Policy Ayarla

   - Create database named `ict`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

   - Start the database

   - Note connection details (default: neo4j://localhost:7687)# 2. Virtual Environment Aktive Et

.\.venv\Scripts\Activate.ps1

2. **Python 3.10+**

   - Verify: `python --version`# 3. Paketleri Yükle

   - Install if needed from: https://www.python.org/downloads/pip install neo4j shapely pyproj scikit-learn python-dateutil

```

3. **HERE API Key**

   - Sign up: https://platform.here.com/---

   - Create API key with Traffic API access

   - Free tier: 250,000 transactions/month## 📊 Veri Yükleme Sırası (Sırayla Çalıştırın)



## Installation```powershell

# Adım 1: Schema Oluştur

### Step 1: Clone or Download Projectpython neo4j_gnn_ingest.py --init-schema



```powershell# Adım 2: Segmentleri Yükle

cd "C:\Users\Yusuf\Desktop\"python neo4j_gnn_ingest.py --load-segments data/edges_static.geojson

# Project folder: HERE V6

```# Adım 3: Yakınlık İlişkilerini Oluştur

python neo4j_gnn_ingest.py --build-next-to --threshold-m 3

### Step 2: Install Python Dependencies

# Adım 4: Trafik Ölçümlerini Yükle (Archive'deki Tüm Dosyalar)

```powershellGet-ChildItem archive/*.geojson | ForEach-Object { python neo4j_gnn_ingest.py --load-measure $_.FullName }

cd "HERE V7 15.10.2025.13.02\HERE V7\HERE V6"```

pip install -r requirements.txt

```---



Required packages:##  Mevcut Durum

- neo4j>=5.0 (database driver)

- pandas (data processing)**Neo4j'de Yüklü Veri:**

- numpy (numerical operations)-  2,366 Segment

- python-dotenv (configuration)-  6,811 Measure

- shapely (geospatial calculations)-  4,086 NEXT_TO İlişkisi

- pyproj (coordinate projections)-  4 Zaman Dilimi

- flask (web visualization)

---

### Step 3: Configure Environment

##  Veri Kontrolü

Edit `config/.env`:

```powershell

```properties# Neo4j'deki veri sayısını kontrol et

# Replace with your HERE API keypython -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '123456789')); session = driver.session(); result = session.run('MATCH (s:Segment) RETURN count(s) as segments'); segments = result.single()['segments']; result = session.run('MATCH (m:Measure) RETURN count(m) as measures'); measures = result.single()['measures']; print(f'Segment: {segments}, Measure: {measures}'); session.close(); driver.close()"

HERE_API_KEY=your_actual_api_key_here```



# Define your area of interest (lon_min, lat_min, lon_max, lat_max)---

BBOX=30.4000,39.7000,30.7500,39.8600

##  Çıkış

# Neo4j connection (match your Neo4j Desktop settings)

NEO4J_URI=neo4j://127.0.0.1:7687```powershell

NEO4J_USER=neo4j# Virtual Environment'tan çık

NEO4J_PASS=your_neo4j_passworddeactivate

NEO4J_DATABASE=ict```


# Pipeline timing (minutes between data collection)
PIPELINE_INTERVAL_MIN=15
```

**Critical**: Verify NEO4J_DATABASE matches your active database in Neo4j Desktop.

### Step 4: Test Connections

**Test HERE API:**
```powershell
python test_api.py
```
Expected: JSON response with traffic data

**Test Neo4j:**
```powershell
python test_neo4j_connection.py
```
Expected: Connection successful message

## First Run

### Option 1: Single Pipeline Execution

For one-time data collection:

```powershell
python run_pipeline.py
```

What happens:
1. Fetches traffic data from HERE API (5-10 seconds)
2. Processes GeoJSON and creates archive (2-3 seconds)
3. Builds time series aggregation (1-2 seconds)
4. Loads data to Neo4j (10-15 seconds)
5. Creates spatial topology (2-5 seconds first run)

**Total time**: 3-5 minutes (first run with topology creation)

### Option 2: Automated Loop

For continuous monitoring:

```powershell
python run_loop.py
```

Behavior:
- Runs pipeline every PIPELINE_INTERVAL_MIN minutes
- Press Ctrl+C to stop
- Logs execution status
- Resumes automatically after errors

**Subsequent runs**: 20-30 seconds (topology skipped)

## Verify Data

### Neo4j Browser

1. Open: http://localhost:7474
2. Login with credentials from config/.env
3. Select database: `ict`

**Check segments:**
```cypher
MATCH (s:Segment) RETURN count(s) AS segment_count
```
Expected: 1,500-2,000 segments (depends on BBOX size)

**Check measurements:**
```cypher
MATCH (m:Measure) RETURN count(m) AS measure_count
```
Expected: Same as segment count after first run, grows with each iteration

**Check topology:**
```cypher
MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS connection_count
```
Expected: 2,500-3,000 connections (created automatically)

**View recent traffic:**
```cypher
MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)
RETURN s.road_name, m.speed_kmh, m.jam_factor, m.timestamp
ORDER BY m.timestamp DESC
LIMIT 20
```

### Visualization

**Static Map:**
1. Open `map.html` in browser
2. View color-coded traffic flow
3. Hover over segments for details

**Live Dashboard:**
```powershell
python src/visualization/12_simple_web_server.py
```
Then open: http://localhost:5000

Features:
- Auto-refresh with latest Neo4j data
- Interactive segment selection
- Timestamp navigation

## Common Workflows

### Daily Monitoring

Start automated collection in morning:
```powershell
python run_loop.py
```

Stop in evening:
- Press Ctrl+C in terminal

Data persists in Neo4j across sessions.

### Weekly Maintenance

Check topology health:
```powershell
python src/gnn/check_topology.py
```

Clean old data if needed:
```powershell
python clean_all.py
python run_pipeline.py
```

### Change Coverage Area

1. Update BBOX in `config/.env`
2. Clean database: `python clean_all.py`
3. Restart pipeline: `python run_pipeline.py`

New area will have different segment count and topology.

## Troubleshooting

### "Authentication Failed"

**Cause**: Wrong password or database name

**Fix**:
1. Open Neo4j Desktop
2. Verify database name (must be `ict` or update config/.env)
3. Check password matches NEO4J_PASS
4. Restart database

### "No Traffic Data"

**Cause**: Invalid API key or BBOX

**Fix**:
1. Run `python test_api.py` to verify API key
2. Check BBOX coordinates (must be lon,lat format)
3. Verify area has traffic coverage (urban areas better)

### "Topology Not Created"

**Cause**: Skipped on subsequent runs (expected behavior)

**Verify**:
```powershell
python src/gnn/check_topology.py
```

If shows "YOK", force rebuild:
```powershell
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
```

### Slow Performance

**Cause**: Large BBOX or many iterations

**Fix**:
1. Reduce BBOX size for testing
2. Increase PIPELINE_INTERVAL_MIN (production: 15-30 min)
3. Enable Neo4j indexes (automatic on first run)

## Next Steps

1. **Read Full Documentation**: `docs/guides/PIPELINE_README.md`
   - Detailed architecture
   - Database schema
   - Performance tuning

2. **Learn Topology Management**: `docs/guides/TOPOLOGY_MANAGEMENT.md`
   - CONNECTS_TO optimization
   - GNN readiness
   - Spatial calculations

3. **Explore Query Examples**: Check PIPELINE_README.md "Integration Examples"
   - Traffic hotspots
   - Congestion analysis
   - Network propagation

## Support

- Configuration issues: Verify `config/.env` matches your setup
- API problems: Consult HERE documentation (https://developer.here.com/)
- Neo4j questions: Use Neo4j browser or Desktop help
- Pipeline errors: Check `logs/` directory for detailed traces

## Summary

**Two commands run everything:**
- `python run_pipeline.py` - Single execution
- `python run_loop.py` - Continuous monitoring

**Data flows:**
HERE API → GeoJSON archive → Neo4j database → Visualization

**First run**: 3-5 minutes (topology creation)  
**Subsequent runs**: 20-30 seconds (topology skipped)

Start collecting traffic data in under 10 minutes.
