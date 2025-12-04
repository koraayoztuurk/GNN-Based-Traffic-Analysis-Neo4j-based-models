# Topology Management Guide

## Overview

Graph topology enables spatial analysis and graph neural network (GNN) applications. The CONNECTS_TO relationship network represents physical road connectivity based on geometric proximity.

## Topology Architecture

### Node Schema Enhancement

**Purpose**: Extract geographic coordinates from GeoJSON geometries

**Process**:
1. Reads segment geometries from Neo4j
2. Calculates centroid (lat, lon) for each segment
3. Updates Segment nodes with coordinate properties
4. Creates indexes for spatial queries

**Performance**: 1-2 seconds for 1,500-2,000 segments

**Script**: `src/gnn/run_step1_enhance_schema.py`

### Spatial Relationship Building

**Purpose**: Create CONNECTS_TO edges between adjacent road segments

**Algorithm**:
1. Extract all segment endpoints (first and last coordinates)
2. Build spatial index (R-tree) for efficient proximity queries
3. For each segment:
   - Find segments within 12-meter radius of endpoints
   - Create bidirectional CONNECTS_TO relationships
   - Store distance_m property

**Performance**: 2-5 seconds for typical urban network (2,500-3,000 connections)

**Script**: `src/gnn/run_step2_build_connects_to.py`

### Distance Threshold

**Default**: 12 meters

**Rationale**:
- Accounts for GPS accuracy variations
- Bridges minor data gaps between segments
- Prevents over-connection in complex intersections

**Tuning**:
- Increase (15-20m): More connectivity, risk of false connections
- Decrease (8-10m): Stricter matching, risk of fragmentation

Modify in `src/gnn/run_step2_build_connects_to.py`:
```python
DISTANCE_THRESHOLD = 12  # meters
```

## Smart Topology Management

### Automatic Optimization

The pipeline uses intelligent topology management via `ensure_topology.py`:

**Step 1: Schema Enhancement** (Always Runs)
- Updates coordinates for new segments
- Fast operation (1-2 seconds)
- Safe to run repeatedly

**Step 2: Relationship Building** (Conditional)
- Only runs if topology missing or insufficient
- Checks via `check_topology.py` subprocess
- Skips if CONNECTS_TO exists (saves 5-30 minutes)

### Skip Logic

```python
# Simplified logic from ensure_topology.py
output = subprocess.run(['python', 'src/gnn/check_topology.py'], capture_output=True)
if "YOK" in output.stdout or "YETERSIZ" in output.stdout:
    # Topology missing or insufficient - create it
    run_step2_build_connects_to()
else:
    # Topology exists - skip (saves time)
    print("Topology already exists, skipping...")
```

**Performance Impact**:
- Without skip: 30 minutes per iteration (wasteful)
- With skip: 2-3 seconds per iteration (optimal)

### Verification Tool

**Command**:
```powershell
python src/gnn/check_topology.py
```

**Output Codes**:

`[CHECK] Koordinat durumu...`
- `[OK] MEVCUT`: All segments have lat/lon coordinates
- `[WARN] YOK`: Missing coordinates (run step1)

`[CHECK] CONNECTS_TO ilişkileri...`
- `[OK] MEVCUT (N adet)`: N connections exist (sufficient)
- `[WARN] YOK`: No connections found (run step2)
- `[WARN] YETERSIZ (N adet)`: Insufficient connections (run step2)

**Minimum Threshold**: 100 CONNECTS_TO relationships

**Typical Values**:
- Small BBOX (test): 500-1,000 connections
- Medium BBOX: 2,500-5,000 connections
- Large city network: 10,000+ connections

## Manual Topology Operations

### Force Full Rebuild

When to use:
- Topology corruption suspected
- Changed distance threshold
- Major database schema changes

**Commands**:
```powershell
# Step 1: Delete existing topology
python -c "from neo4j import GraphDatabase; import os; from dotenv import load_dotenv; from pathlib import Path; load_dotenv(Path('config')/ '.env'); driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASS'))); driver.execute_query('MATCH ()-[r:CONNECTS_TO]->() DELETE r', database_=os.getenv('NEO4J_DATABASE'))"

# Step 2: Rebuild from scratch
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
```

### Verify Topology Quality

**Command**:
```powershell
python src/gnn/test_gnn_readiness.py
```

**Checks**:
1. Node count (>100 segments)
2. Coordinate coverage (100% with lat/lon)
3. Connectivity (CONNECTS_TO exists)
4. Feature richness (traffic measurements)
5. Temporal depth (multiple timestamps)

**Output**: GNN readiness score (0-100%)

**Target**: 100% for production use

## Topology Patterns

### Typical Network Structure

```
Segment A (lat, lon) ──CONNECTS_TO {distance_m: 8.3}──> Segment B
                 ╰──CONNECTS_TO {distance_m: 11.7}──> Segment C
```

### Query Examples

**Find connected segments:**
```cypher
MATCH (s1:Segment {segment_id: 'example_id'})-[c:CONNECTS_TO]->(s2:Segment)
RETURN s2.segment_id, s2.road_name, c.distance_m
ORDER BY c.distance_m
```

**Count connections per segment:**
```cypher
MATCH (s:Segment)-[c:CONNECTS_TO]->()
RETURN s.segment_id, s.road_name, count(c) AS connection_count
ORDER BY connection_count DESC
LIMIT 10
```

**Find isolated segments:**
```cypher
MATCH (s:Segment)
WHERE NOT (s)-[:CONNECTS_TO]-()
RETURN s.segment_id, s.road_name, s.lat, s.lon
```

**Network diameter (longest path):**
```cypher
MATCH path = (s1:Segment)-[:CONNECTS_TO*]->(s2:Segment)
RETURN length(path) AS path_length, s1.segment_id, s2.segment_id
ORDER BY path_length DESC
LIMIT 1
```

## Performance Optimization

### Spatial Index Usage

PostGIS-style queries not supported in Neo4j Community. Current implementation:
1. Extract all coordinates to Python
2. Build scipy.spatial.cKDTree
3. Query 12m radius per segment
4. Batch create relationships

**Alternative** (Enterprise only):
```cypher
-- Requires spatial plugin
CALL spatial.addPointLayer('segments')
CALL spatial.withinDistance('segments', {lat: 39.75, lon: 30.50}, 0.012)
```

### Batch Size Tuning

In `run_step2_build_connects_to.py`:

```python
BATCH_SIZE = 500  # Relationships per transaction
```

**Trade-offs**:
- Larger (1000+): Faster but more memory
- Smaller (100-500): Slower but safer for large networks

### Parallel Processing

Current: Sequential processing (single-threaded)

**Future enhancement**:
```python
from multiprocessing import Pool
# Partition segments by spatial grid
# Process each cell in parallel
# Merge results
```

## Troubleshooting

### "No CONNECTS_TO Created"

**Symptom**: step2 completes but 0 relationships in database

**Causes**:
1. Segments missing coordinates (run step1 first)
2. Distance threshold too strict
3. Segments too far apart (sparse network)

**Diagnosis**:
```powershell
python src/gnn/check_topology.py
```

**Fix**:
```powershell
# Verify coordinates exist
python src/gnn/run_step1_enhance_schema.py
# Increase threshold temporarily (test)
# Edit run_step2_build_connects_to.py: DISTANCE_THRESHOLD = 20
python src/gnn/run_step2_build_connects_to.py
```

### "Topology Takes Too Long"

**Symptom**: step2 runs for 30+ minutes

**Causes**:
1. Very large network (10,000+ segments)
2. Inefficient spatial queries
3. Small batch size

**Fix**:
```powershell
# Increase batch size
# Edit run_step2_build_connects_to.py: BATCH_SIZE = 1000
# Or split by geographic regions
```

### "Duplicate Relationships"

**Symptom**: Multiple CONNECTS_TO between same segment pair

**Cause**: Running step2 multiple times without cleanup

**Fix**:
```cypher
// Delete duplicates, keep one
MATCH (s1:Segment)-[r:CONNECTS_TO]->(s2:Segment)
WITH s1, s2, collect(r) AS rels
WHERE size(rels) > 1
FOREACH (r IN tail(rels) | DELETE r)
```

### "GNN Readiness Fails"

**Symptom**: test_gnn_readiness.py reports < 100%

**Common issues**:
1. Missing coordinates: Run step1
2. No topology: Run step2
3. No traffic data: Run pipeline first

**Fix sequence**:
```powershell
python run_pipeline.py  # Ensure data exists
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
python src/gnn/test_gnn_readiness.py
```

## Integration with Pipeline

### Automatic Topology in Loop

`run_loop.py` calls `run_pipeline.py`, which calls `ensure_topology.py`:

**First iteration**:
1. Fetch data → Load to Neo4j
2. ensure_topology.py runs
3. check_topology.py returns "YOK"
4. step1 runs (coordinates)
5. step2 runs (CONNECTS_TO created)
6. Total: 3-5 minutes

**Subsequent iterations**:
1. Fetch data → Load to Neo4j
2. ensure_topology.py runs
3. check_topology.py returns "MEVCUT"
4. step2 SKIPPED
5. Total: 20-30 seconds

### Manual Override

Disable automatic topology:
```python
# In run_pipeline.py, comment out:
# subprocess.run(['python', str(GNN_DIR / 'ensure_topology.py')])
```

### Custom Topology Schedule

Run topology rebuild weekly:
```powershell
# Daily: skip topology
python run_pipeline.py

# Sunday: force rebuild
python src/gnn/run_step1_enhance_schema.py
python src/gnn/run_step2_build_connects_to.py
```

## Best Practices

1. **Verify After Initial Load**
   ```powershell
   python src/gnn/test_gnn_readiness.py
   ```
   Ensure 100% before GNN training.

2. **Monitor Connection Density**
   ```cypher
   MATCH ()-[r:CONNECTS_TO]->()
   RETURN count(r) AS total_connections
   ```
   Should grow proportionally with segment count.

3. **Periodic Validation**
   Run `check_topology.py` weekly to catch degradation.

4. **Archive Topology State**
   Before major changes, export:
   ```cypher
   MATCH (s1:Segment)-[r:CONNECTS_TO]->(s2:Segment)
   RETURN s1.segment_id, s2.segment_id, r.distance_m
   ```

5. **Test Threshold Changes**
   Use small BBOX for experimentation before production.

## Advanced Topics

### Directional Topology

Current: Bidirectional connections (A→B and B→A)

**Future**: Respect road direction from HERE API
```python
if segment['direction'] == 'ONE_WAY':
    create_single_direction(start, end)
else:
    create_bidirectional(start, end)
```

### Weighted Topology

Current: Simple distance_m property

**Enhancement**: Traffic-aware weights
```cypher
MATCH (s1:Segment)-[c:CONNECTS_TO]->(s2:Segment)
MATCH (s2)-[:HAS_MEASURE]->(m:Measure)
SET c.traffic_weight = m.jam_factor * c.distance_m
```

### Temporal Topology

Current: Static topology (no time dimension)

**Future**: Time-varying connectivity
```cypher
CREATE (s1)-[:CONNECTS_TO {
  distance_m: 10.5,
  valid_from: datetime('2025-01-01T00:00:00'),
  valid_until: datetime('2025-12-31T23:59:59')
}]->(s2)
```

## Related Documentation

- **PIPELINE_README.md**: Full system architecture
- **QUICKSTART.md**: Initial setup and first run
- **test_gnn_readiness.py**: Validation script details

## Summary

Topology management is automatic and optimized:
- Creates relationships once (first run)
- Skips on subsequent runs (2-3 seconds vs 30 minutes)
- Verifies quality automatically
- Manual override available for special cases

Smart skip logic ensures fast iteration times without sacrificing data quality.
