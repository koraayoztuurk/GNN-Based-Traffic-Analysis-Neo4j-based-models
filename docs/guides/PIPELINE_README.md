# HERE Traffic Flow Pipeline - Technical Documentation

## System Overview

Real-time traffic data collection system integrating HERE Traffic API with Neo4j graph database. Optimized for continuous monitoring with automated topology management.

## Architecture

### Pipeline Stages

1. **Data Acquisition** (`01_fetch_here_flow.py`)
   - Fetches traffic flow data from HERE API
   - Coverage: Configured bounding box (BBOX in .env)
   - Output: `here_flow_raw.json` (temporary buffer)
   - Rate: Based on PIPELINE_INTERVAL_MIN setting

2. **Geospatial Processing** (`02_render_flow_map.py`)
   - Converts raw JSON to GeoJSON format
   - Creates timestamped archive: `archive/flow_YYYYMMDD_HHMM.geojson`
   - Generates static visualization: `map.html`
   - Manages archive size (MAX_ARCHIVES limit)

3. **Time Series Aggregation** (`05_build_timeseries.py`)
   - Reads all archive GeoJSON files
   - Aggregates traffic metrics across time
   - Output: `data/timeseries.parquet` (columnar format)
   - Enables temporal analysis and forecasting

4. **Database Loading** (`07_silent_load_to_neo4j.py`)
   - Loads traffic segments and measurements to Neo4j
   - Creates/updates nodes: Segment, Measure
   - Establishes relationships: HAS_MEASURE
   - Non-interactive mode for automation

5. **Topology Management** (`ensure_topology.py`)
   - Step 1: Extracts coordinates, updates schema (always runs)
   - Step 2: Builds CONNECTS_TO relationships (conditional)
   - Smart skip: Only creates topology on first run
   - Performance: 2-3 seconds per iteration after initial setup

## Database Schema

### Nodes

**Segment**
- Properties: `segment_id`, `lat`, `lon`, `road_name`, `direction`, `length_m`
- Purpose: Represents physical road segments
- Indexes: `segment_id` (unique)

**Measure**
- Properties: `timestamp`, `speed_kmh`, `confidence`, `jam_factor`, `free_flow_kmh`
- Purpose: Traffic measurements at specific time points
- Indexes: `timestamp`

### Relationships

**HAS_MEASURE**
- Pattern: `(Segment)-[:HAS_MEASURE]->(Measure)`
- Purpose: Links traffic data to road segments

**CONNECTS_TO**
- Pattern: `(Segment)-[:CONNECTS_TO {distance_m}]->(Segment)`
- Purpose: Spatial topology for graph algorithms
- Creation: Automated on first pipeline run
- Threshold: 12 meters proximity between segment endpoints

### Constraints and Indexes

```cypher
CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE;
CREATE INDEX segment_coords IF NOT EXISTS FOR (s:Segment) ON (s.lat, s.lon);
CREATE INDEX measure_timestamp IF NOT EXISTS FOR (m:Measure) ON m.timestamp;
```

## Configuration

### Environment Variables (`config/.env`)

```properties
# HERE API
HERE_API_KEY=your_api_key_here
BBOX=30.4000,39.7000,30.7500,39.8600  # lon_min,lat_min,lon_max,lat_max

# Neo4j Connection
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
NEO4J_DATABASE=ict

# Pipeline Settings
PIPELINE_INTERVAL_MIN=15          # Collection interval (minutes)
TIMEZONE=Europe/Istanbul          # Timestamp timezone
MAX_ARCHIVES=500                  # Maximum archive files to retain
```

### Key Parameters

- **BBOX**: Geographic coverage area (adjust for your region)
- **PIPELINE_INTERVAL_MIN**: Trade-off between data freshness and API usage
- **MAX_ARCHIVES**: Disk space management (older files auto-deleted)
- **NEO4J_DATABASE**: Must match active database in Neo4j Desktop

## Execution Modes

### Single Pipeline Run

```powershell
python run_pipeline.py
```

Executes all 5 stages sequentially. Use for:
- Initial database population
- Manual data collection
- Testing after configuration changes

Expected duration:
- First run: 3-5 minutes (includes CONNECTS_TO creation)
- Subsequent runs: 20-30 seconds (topology skipped)

### Automated Loop

```powershell
python run_loop.py
```

Continuous execution with interval-based scheduling. Use for:
- Production monitoring
- Long-term data collection
- Time series analysis

Behavior:
- Reads PIPELINE_INTERVAL_MIN from config/.env
- Sleeps between iterations (non-blocking)
- Logs execution status
- Ctrl+C to stop gracefully

### Database Cleanup

```powershell
python clean_all.py
```

Complete database reset. Deletes:
- All nodes (Segment, Measure)
- All relationships (HAS_MEASURE, CONNECTS_TO)
- Does NOT delete archive files or configuration

Warning: Irreversible operation. Use only for fresh starts.

## Maintenance Operations

### Topology Verification

```powershell
python src/gnn/check_topology.py
```

Validates graph structure:
- Coordinate presence (lat, lon)
- CONNECTS_TO relationship count
- Minimum connectivity threshold (100 relationships)

Output codes:
- `[OK] MEVCUT`: Topology valid
- `[WARN] YOK`: Missing coordinates or relationships
- `[WARN] YETERSIZ`: Insufficient connectivity

### Manual Topology Rebuild

```powershell
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
```

Force topology recreation. Only needed if:
- Topology corrupted
- Schema changed
- Different distance threshold required

### GNN Readiness Test

```powershell
python src/gnn/test_gnn_readiness.py
```

Validates data for graph neural networks:
- Node count (> 100 segments)
- Coordinate completeness (100%)
- Topology connectivity (CONNECTS_TO)
- Feature richness (traffic measurements)
- Temporal coverage (multiple timestamps)

## Visualization

### Static Map

Generated automatically by pipeline: `map.html`

Features:
- Color-coded traffic flow (green to red)
- Segment metadata (hover)
- Latest data snapshot
- No server required (standalone HTML)

### Live Web Dashboard

```powershell
python src/visualization/12_simple_web_server.py
```

Flask-based real-time visualization:
- URL: http://localhost:5000
- Auto-refresh: Pulls latest Neo4j data
- REST API: `/api/timestamps`, `/api/traffic`
- Supports concurrent access

## Performance Characteristics

### Database Growth

| Metric | First Run | After 100 Iterations | Notes |
|--------|-----------|---------------------|-------|
| Segments | 1,500-2,000 | 1,500-2,000 | Static (geographic coverage) |
| Measures | 1,500-2,000 | 150,000-200,000 | Linear growth per iteration |
| CONNECTS_TO | 2,500-3,000 | 2,500-3,000 | Created once, stable |
| Archive Files | 1 | 100 | Cyclic (MAX_ARCHIVES limit) |

### Execution Times

| Operation | Duration | Frequency | Notes |
|-----------|----------|-----------|-------|
| API Fetch | 2-5 sec | Per iteration | Network-dependent |
| GeoJSON Processing | 1-2 sec | Per iteration | CPU-bound |
| Neo4j Load | 5-10 sec | Per iteration | First load slower (indexes) |
| Topology Creation | 2-5 sec | First run only | Spatial calculation intensive |
| Topology Skip | 2-3 sec | Subsequent runs | Only coordinate update |

### Resource Usage

- **Disk**: ~1 MB per archive file (gzip compression recommended for storage)
- **Memory**: <500 MB Python process (peak during time series aggregation)
- **Neo4j**: ~100 MB per 100,000 Measure nodes (page cache dependent)
- **Network**: ~200 KB per HERE API call (gzip encoded)

## Troubleshooting

### Authentication Errors

**Symptom**: `Neo.ClientError.Security.Unauthorized`

**Cause**: Database parameter mismatch

**Solution**:
1. Verify Neo4j Desktop active database name
2. Update NEO4J_DATABASE in config/.env
3. Restart Neo4j database
4. Rerun pipeline

### Topology Not Created

**Symptom**: check_topology.py reports "YOK" repeatedly

**Cause**: Initial segment load without coordinates

**Solution**:
```powershell
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
```

### HERE API Rate Limits

**Symptom**: HTTP 429 or empty responses

**Cause**: Interval too short (< 1 minute not recommended)

**Solution**:
1. Increase PIPELINE_INTERVAL_MIN in config/.env
2. Check HERE API plan limits
3. Consider larger BBOX (fewer requests for same coverage)

### Archive Directory Growth

**Symptom**: Disk space warnings

**Cause**: MAX_ARCHIVES set too high

**Solution**:
1. Reduce MAX_ARCHIVES in config/.env
2. Run pipeline once (auto-cleanup triggers)
3. Or manually: `Remove-Item archive/flow_*.geojson -Force`

### Database Performance Degradation

**Symptom**: Slow queries after many iterations

**Cause**: Missing indexes or page cache too small

**Solution**:
1. Verify indexes exist (see Database Schema section)
2. Increase Neo4j heap size (neo4j.conf: `dbms.memory.heap.max_size`)
3. Enable query logging to identify slow operations

## Best Practices

### Production Deployment

1. **Set Appropriate Interval**: 15-30 minutes balances freshness vs. API usage
2. **Monitor Archive Size**: Enable automatic cleanup (MAX_ARCHIVES < 1000)
3. **Schedule Maintenance**: Weekly topology verification
4. **Backup Database**: Neo4j backup before major updates
5. **Log Rotation**: Monitor logs/ directory growth

### Development Workflow

1. **Test with Small BBOX**: Faster iterations during development
2. **Use Short Interval**: 1-3 minutes for testing (not production)
3. **Clean Database Often**: Fresh starts help identify issues
4. **Verify Each Stage**: Run pipeline steps individually before loop

### Data Quality

1. **Validate API Key**: Test with test_api.py before production
2. **Check Timestamps**: Ensure timezone configured correctly
3. **Monitor Confidence**: Filter low-confidence measurements if needed
4. **Archive Inspection**: Periodically verify GeoJSON files not corrupted

## Integration Examples

### Query Recent Traffic

```cypher
MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)
WHERE m.timestamp > datetime() - duration({hours: 1})
RETURN s.segment_id, s.road_name, avg(m.speed_kmh) AS avg_speed
ORDER BY avg_speed
LIMIT 10
```

### Find Traffic Hotspots

```cypher
MATCH (s:Segment)-[:HAS_MEASURE]->(m:Measure)
WHERE m.jam_factor > 5
WITH s, count(m) AS congestion_count
WHERE congestion_count > 10
RETURN s.road_name, s.direction, congestion_count
ORDER BY congestion_count DESC
```

### Analyze Traffic Propagation

```cypher
MATCH path = (s1:Segment)-[:CONNECTS_TO*1..3]->(s2:Segment)
WHERE s1.segment_id = 'start_segment_id'
WITH s2, length(path) AS hops
MATCH (s2)-[:HAS_MEASURE]->(m:Measure)
WHERE m.timestamp > datetime() - duration({minutes: 30})
RETURN s2.road_name, hops, avg(m.jam_factor) AS avg_jam
ORDER BY hops, avg_jam DESC
```

## Technical Support

For issues not covered in this documentation:
1. Check logs/ directory for error details
2. Verify configuration with test scripts (test_api.py, test_neo4j_connection.py)
3. Review Neo4j browser (http://localhost:7474) for data inspection
4. Consult HERE API documentation for data schema changes

## Version Information

This documentation reflects system version with:
- 2-command execution model (run_pipeline.py, run_loop.py)
- Optimized topology management (smart skip logic)
- Centralized configuration (config/.env)
- Professional codebase structure (src/ organization)

Last updated: 2025-10-21
