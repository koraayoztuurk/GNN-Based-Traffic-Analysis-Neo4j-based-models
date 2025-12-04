#!/usr/bin/env python3
"""
neo4j_gnn_ingest.py
-------------------
Minimal-Cypher pipeline to store HERE Flow v7 data into Neo4j in a GNN/GCN-friendly schema.

Supports:
  • GeoJSON static segments (edges_static.geojson)
  • GeoJSON one-shot flow snapshots (flow_YYYYMMDD_HHMM.geojson)
  • Parquet time-series (single file or whole directory)

Dependencies (pip):
  pip install neo4j shapely pyproj scikit-learn python-dateutil pyarrow pandas

ENV (or edit constants below):
  NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")

Usage examples:
  # one-time constraints
  python neo4j_gnn_ingest.py --init-schema

  # load static segments
  python neo4j_gnn_ingest.py --load-segments edges_static.geojson

  # build adjacency by proximity (CONNECTS_TO is created via GNN pipeline)
  # Use: python src/gnn/run_step2_build_connects_to.py

  # load one flow snapshot (dynamic measures, GeoJSON)
  python neo4j_gnn_ingest.py --load-measure flow_20251003_1332.geojson --ts 2025-10-03T13:32:00Z

  # load time-series from a Parquet file
  python neo4j_gnn_ingest.py --load-parquet data/flow_2025-10-03.parquet --batch-size 10000

  # load time-series from all Parquet files under a directory (recursive)
  python neo4j_gnn_ingest.py --load-parquet-dir data/timeseries --batch-size 10000

  # map custom column names in Parquet
  python neo4j_gnn_ingest.py --load-parquet flow.parquet --col-map "segmentId=seg_id,ts=timestamp,freeFlow=ff_speed"

  # set default provider/source for Parquet rows missing those columns
  python neo4j_gnn_ingest.py --load-parquet flow.parquet --set-provider here --set-source archive
"""
import os, sys, json, hashlib, argparse, math, re
from datetime import datetime, timezone
from dateutil import parser as dtparser

from neo4j import GraphDatabase
from shapely.geometry import LineString
from shapely import wkt
import pyproj
from sklearn.neighbors import KDTree

# Parquet / DataFrame
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ---------- Helpers ----------
def normalize_segment_id(seg_id):
    """
    Normalize segment ID - sub-segment'leri ana yol ID'sine dönüştür
    edge:xxx:001 -> edge:xxx
    edge:xxx:002 -> edge:xxx
    edge:xxx -> edge:xxx (değişmez)
    """
    if not seg_id:
        return seg_id
    # Sub-segment varsa (ikiden fazla : varsa), son kısmı kaldır
    parts = seg_id.split(':')
    if len(parts) > 2:
        # edge:xxx:001 -> edge:xxx
        return ':'.join(parts[:2])
    return seg_id

def mk_segment_id(here_segment_id, coords):
    base = here_segment_id or json.dumps(coords, ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]

def line_to_wkt(coords):
    return LineString([(c[0], c[1]) for c in coords]).wkt

def snap_15(dt: datetime):
    m = (dt.minute // 15) * 15
    return dt.replace(minute=m, second=0, microsecond=0, tzinfo=timezone.utc)

def infer_ts_from_filename(path):
    # expects ..._YYYYMMDD_HHMM...
    m = re.search(r'(\d{8})[_-](\d{4})', os.path.basename(path))
    if not m:
        return None
    ymd, hm = m.group(1), m.group(2)
    dt = datetime.strptime(ymd+hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return dt

def connect():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_session():
    """Helper to create session with database parameter"""
    driver = connect()
    return driver.session(database=NEO4J_DATABASE)

# ---------- Cypher (minimal & parameterized) ----------
CYPHER_CONSTRAINTS = """
CREATE CONSTRAINT seg_id IF NOT EXISTS
FOR (s:Segment) REQUIRE s.segmentId IS UNIQUE;

CREATE INDEX measure_seg_ts IF NOT EXISTS
FOR (m:Measure) ON (m.segmentId, m.timestamp);

CREATE INDEX measure_timestamp IF NOT EXISTS
FOR (m:Measure) ON (m.timestamp);
"""

CYPHER_UPSERT_SEGMENTS = """
UNWIND $rows AS r
MERGE (s:Segment {segmentId:r.segmentId})
ON CREATE SET
  s.hereSegmentId = r.hereSegmentId,
  s.osmWayId      = r.osmWayId,
  s.frc           = r.frc,
  s.lengthM       = r.lengthM,
  s.name          = r.name,
  s.geom          = r.geom,
  s.startLat      = r.startLat,
  s.startLon      = r.startLon,
  s.endLat        = r.endLat,
  s.endLon        = r.endLon
ON MATCH SET
  s.hereSegmentId = r.hereSegmentId,
  s.osmWayId      = r.osmWayId,
  s.frc           = r.frc,
  s.lengthM       = r.lengthM,
  s.name          = r.name,
  s.geom          = r.geom
"""

# CYPHER_NEXT_TO removed - use CONNECTS_TO instead (created by run_step2_build_connects_to.py)

# TS15 removed - deprecated time bucket feature

CYPHER_UPSERT_MEASURE = """
UNWIND $rows AS r
MERGE (m:Measure {segmentId:r.segmentId, timestamp:r.timestamp})
  ON CREATE SET
    m.speed = r.speed,
    m.freeFlow = r.freeFlow,
    m.jamFactor = r.jamFactor,
    m.confidence = r.confidence,
    m.traversability = r.traversability,
    m.subSegments = r.subSegments,
    m.provider = r.provider,
    m.source = r.source,
    m.asOf = r.asOf
  ON MATCH SET
    m.speed = r.speed,
    m.freeFlow = r.freeFlow,
    m.jamFactor = r.jamFactor,
    m.confidence = r.confidence,
    m.traversability = r.traversability,
    m.subSegments = r.subSegments,
    m.provider = r.provider,
    m.source = r.source,
    m.asOf = r.asOf
WITH m
MATCH (s:Segment {segmentId:m.segmentId})
MERGE (s)-[:AT_TIME]->(m)
"""

# ---------- Core commands (your originals, unchanged behavior) ----------
def cmd_init_schema():
    with get_session() as s:
        statements = [stmt.strip() for stmt in CYPHER_CONSTRAINTS.split(';') if stmt.strip()]
        for stmt in statements:
            s.run(stmt)
    print("[OK] Constraints & indexes ensured.")

def cmd_load_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if "features" in data else data

    rows = []
    for feat in feats:
        props = feat.get("properties", {})
        coords = feat["geometry"]["coordinates"]
        # edge_id'yi öncelikle kullan (timeseries ile uyumlu)
        raw_seg_id = (
            props.get("edge_id") or 
            props.get("segmentId") or 
            mk_segment_id(props.get("hereSegmentId"), coords)
        )
        # NORMALIZE: edge:xxx:001 -> edge:xxx (sub-segment'leri ana yol ID'sine dönüştür)
        seg_id = normalize_segment_id(raw_seg_id)
        # Start ve end koordinatları
        start_lat = coords[0][1] if coords else None
        start_lon = coords[0][0] if coords else None
        end_lat = coords[-1][1] if coords else None
        end_lon = coords[-1][0] if coords else None
        
        rows.append({
            "segmentId": seg_id,
            "hereSegmentId": props.get("hereSegmentId"),
            "osmWayId": props.get("osmWayId"),
            "frc": props.get("frc"),
            "lengthM": props.get("lengthM"),
            "name": props.get("name"),
            "geom": line_to_wkt(coords),
            "startLat": start_lat,
            "startLon": start_lon,
            "endLat": end_lat,
            "endLon": end_lon
        })
    with get_session() as s:
        s.run(CYPHER_UPSERT_SEGMENTS, rows=rows)
    print(f"[OK] Segments upserted: {len(rows)}")

# cmd_build_next_to removed - NEXT_TO is deprecated
# Use CONNECTS_TO instead: python src/gnn/run_step2_build_connects_to.py

def cmd_load_measure(path, ts=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if "features" in data else data

    if ts:
        dt = dtparser.isoparse(ts)
    else:
        dt = infer_ts_from_filename(path)
        if not dt:
            raise ValueError(f"Timestamp dosya adından çıkarılamadı: {path}. --ts parametresi kullanın veya dosya adını 'flow_YYYYMMDD_HHMM.geojson' formatında adlandırın.")
    
    # snap_15() KALDIRILDI - gerçek timestamp kullan (ArangoDB/TigerGraph ile tutarlı)
    timestamp_iso = dt.isoformat()

    rows = []
    for feat in feats:
        p = feat.get("properties", {})
        # AYNI ÖNCEL İK SIRASI: edge_id → segmentId → hash (segment loader ile uyumlu)
        raw_seg_id = p.get("edge_id") or p.get("segmentId") or p.get("hereSegmentId")
        if not raw_seg_id:
            coords = feat["geometry"]["coordinates"]
            raw_seg_id = mk_segment_id(p.get("hereSegmentId"), coords)
        # NORMALIZE: edge:xxx:001 -> edge:xxx (measure'lar segment ile eşleşmeli)
        seg_id = normalize_segment_id(raw_seg_id)
        rows.append({
            "segmentId": seg_id,
            "timestamp": timestamp_iso,  # Gerçek timestamp (snap_15 kaldırıldı)
            "speed": p.get("speed") or p.get("currentSpeed"),
            "freeFlow": p.get("freeFlow") or p.get("freeFlowSpeed"),
            "jamFactor": p.get("jamFactor"),
            "confidence": p.get("confidence"),
            "traversability": p.get("traversability"),
            "subSegments": json.dumps(p.get("subSegments", None), ensure_ascii=False),
            # optional provenance
            "provider": p.get("provider"),
            "source": p.get("source"),
            "asOf": p.get("asOf"),
        })

    with get_session() as s:
        s.run(CYPHER_UPSERT_MEASURE, rows=rows)

    print(f"[OK] Measures upserted: {len(rows)} @ {timestamp_iso}")

# ---------- NEW: Parquet time-series ingest ----------
def _to_iso_bucket(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        dt = datetime.fromtimestamp(val, tz=timezone.utc)  # epoch
    elif isinstance(val, pd.Timestamp):
        dt = val.tz_convert("UTC").to_pydatetime() if val.tz is not None else val.to_pydatetime().replace(tzinfo=timezone.utc)
    elif isinstance(val, datetime):
        dt = val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    else:
        dt = dtparser.isoparse(str(val))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return snap_15(dt).isoformat()

# _ensure_ts15 removed - deprecated time bucket feature

def _chunk_iter(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i:i+size]

def _parse_col_map(s: str):
    # "segmentId=seg_id,ts=timestamp,freeFlow=ff_speed"
    m = {}
    if not s:
        return m
    for part in [p for p in s.split(",") if "=" in p]:
        k, v = [x.strip() for x in part.split("=", 1)]
        m[k] = v
    return m

def cmd_load_parquet_file(path, batch_size=5000, col_map=None, set_provider=None, set_source=None):
    tbl = pq.read_table(path)
    df = tbl.to_pandas()

    col_map = col_map or {}
    
    # Otomatik sütun algılama (birden fazla olası isim)
    seg_col = col_map.get("segmentId") or (
        "segmentId" if "segmentId" in df.columns else
        "edge_id" if "edge_id" in df.columns else
        "hereSegmentId" if "hereSegmentId" in df.columns else
        "seg_id" if "seg_id" in df.columns else None
    )
    
    ts_col = col_map.get("ts") or (
        "ts" if "ts" in df.columns else
        "t_utc" if "t_utc" in df.columns else
        "timestamp" if "timestamp" in df.columns else
        "datetime" if "datetime" in df.columns else None
    )
    
    ff_col = col_map.get("freeFlow") or (
        "freeFlow" if "freeFlow" in df.columns else
        "freeflow_kmh" if "freeflow_kmh" in df.columns else
        "freeFlowSpeed" if "freeFlowSpeed" in df.columns else
        "free_flow" if "free_flow" in df.columns else None
    )
    
    # Speed sütunu algılama
    speed_col = (
        "speed" if "speed" in df.columns else
        "speed_kmh" if "speed_kmh" in df.columns else
        "currentSpeed" if "currentSpeed" in df.columns else None
    )
    
    # JamFactor sütunu algılama
    jf_col = (
        "jamFactor" if "jamFactor" in df.columns else
        "jam_factor" if "jam_factor" in df.columns else None
    )

    if seg_col is None or ts_col is None:
        raise ValueError(f"Parquet must contain segment ID and timestamp columns.\n"
                        f"Available columns: {list(df.columns)}\n"
                        f"Detected: segmentId={seg_col}, ts={ts_col}")

    # ensure optional columns exist
    for c in ["confidence","traversability","subSegments","provider","source","asOf"]:
        if c not in df.columns: df[c] = None

    # default provider/source if missing
    if set_provider is not None:
        df["provider"] = df["provider"].fillna(set_provider)
    if set_source is not None:
        df["source"] = df["source"].fillna(set_source)

    # Gerçek timestamp'i ISO formatında sakla
    df["_timestamp"] = df[ts_col].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else str(x))
    df["_segmentId"] = df[seg_col]
    df["_speed"] = df[speed_col] if speed_col else None
    df["_freeFlow"] = df[ff_col] if ff_col else None
    df["_jamFactor"] = df[jf_col] if jf_col else None

    with get_session() as s:
        # TS15 bucket artık yok, direkt measure'ları yükle
        total = 0
        for chunk in _chunk_iter(df, batch_size):
            rows = []
            for _, r in chunk.iterrows():
                ts = r["_timestamp"]
                if not ts: continue
                rows.append({
                    "segmentId": r["_segmentId"],
                    "timestamp": ts,  # Gerçek zaman!
                    "speed": r["_speed"],
                    "freeFlow": r["_freeFlow"],
                    "jamFactor": r["_jamFactor"],
                    "confidence": r.get("confidence"),
                    "traversability": r.get("traversability"),
                    "subSegments": r.get("subSegments"),
                    "provider": r.get("provider"),
                    "source": r.get("source"),
                    "asOf": r.get("asOf"),
                })
            if rows:
                s.run(CYPHER_UPSERT_MEASURE, rows=rows)
                total += len(rows)
    print(f"[OK] Ingested parquet: {path} rows={len(df)}, uploaded={total}")

def cmd_load_parquet_dir(dir_path, batch_size=5000, col_map=None, set_provider=None, set_source=None):
    p = Path(dir_path)
    files = sorted([str(fp) for fp in p.rglob("*.parquet")])
    if not files:
        print(f"[WARN] No parquet files under: {dir_path}")
        return
    for f in files:
        cmd_load_parquet_file(f, batch_size=batch_size, col_map=col_map,
                              set_provider=set_provider, set_source=set_source)

def build_connects_to(threshold=12.0):
    """
    CONNECTS_TO ilişkilerini oluştur (spatial topology) - 4 YÖNLÜ KONTROL
    
    Python ile spatial grid optimizasyonu (Cypher O(n²) yerine O(n×k))
    
    Args:
        threshold: Maksimum mesafe (metre)
    """
    print("=" * 70)
    print(f"🔗 CONNECTS_TO ilişkileri oluşturuluyor (threshold={threshold}m)...")
    print("=" * 70)
    print("ℹ️  Spatial grid optimizasyonu kullanılıyor (4 yönlü kontrol)...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Tüm segment'leri al (start ve end koordinatları ile)
            result = session.run("""
                MATCH (s:Segment)
                WHERE s.startLat IS NOT NULL AND s.startLon IS NOT NULL
                  AND s.endLat IS NOT NULL AND s.endLon IS NOT NULL
                RETURN s.segmentId AS segmentId,
                       s.startLat AS startLat, s.startLon AS startLon,
                       s.endLat AS endLat, s.endLon AS endLon
            """)
            
            segments = []
            for record in result:
                segments.append({
                    "segmentId": record["segmentId"],
                    "startLat": record["startLat"],
                    "startLon": record["startLon"],
                    "endLat": record["endLat"],
                    "endLon": record["endLon"]
                })
            
            if not segments:
                print("⚠️  Segment bulunamadı")
                return
            
            print(f"ℹ️  {len(segments)} segment alındı")
            
            # Grid hücre boyutu
            grid_size = (threshold * 2) / 111320.0
            
            # Segment'leri grid'e yerleştir (her segment'in hem start hem end noktasını grid'e ekle)
            grid = {}
            for seg in segments:
                # Start noktasını grid'e ekle
                grid_x_start = int(seg["startLat"] / grid_size)
                grid_y_start = int(seg["startLon"] / grid_size)
                cell_start = (grid_x_start, grid_y_start)
                if cell_start not in grid:
                    grid[cell_start] = []
                if seg not in grid[cell_start]:
                    grid[cell_start].append(seg)
                
                # End noktasını grid'e ekle (farklı hücredeyse)
                grid_x_end = int(seg["endLat"] / grid_size)
                grid_y_end = int(seg["endLon"] / grid_size)
                cell_end = (grid_x_end, grid_y_end)
                if cell_end not in grid:
                    grid[cell_end] = []
                if seg not in grid[cell_end]:
                    grid[cell_end].append(seg)
            
            print(f"ℹ️  {len(grid)} grid hücresine dağıtıldı")
            
            # Edge'leri topla
            edges_to_insert = []
            processed = set()
            total_cells = len(grid)
            current_cell = 0
            
            for cell, segs_in_cell in grid.items():
                current_cell += 1
                if current_cell % 100 == 0:
                    print(f"  Progress: {current_cell}/{total_cells} hücre taranıyor...")
                gx, gy = cell
                neighbor_cells = [
                    (gx-1, gy-1), (gx-1, gy), (gx-1, gy+1),
                    (gx, gy-1), (gx, gy), (gx, gy+1),
                    (gx+1, gy-1), (gx+1, gy), (gx+1, gy+1)
                ]
                
                for seg1 in segs_in_cell:
                    id1 = seg1["segmentId"]
                    
                    for ncell in neighbor_cells:
                        if ncell not in grid:
                            continue
                        for seg2 in grid[ncell]:
                            id2 = seg2["segmentId"]
                            if id1 >= id2:
                                continue
                            
                            pair = (min(id1, id2), max(id1, id2))
                            if pair in processed:
                                continue
                            
                            # 4 yönlü kontrol: end→start, start→end, end→end, start→start
                            distances = [
                                _haversine_distance(
                                    seg1["endLat"], seg1["endLon"],
                                    seg2["startLat"], seg2["startLon"]
                                ),  # end→start
                                _haversine_distance(
                                    seg1["startLat"], seg1["startLon"],
                                    seg2["endLat"], seg2["endLon"]
                                ),  # start→end
                                _haversine_distance(
                                    seg1["endLat"], seg1["endLon"],
                                    seg2["endLat"], seg2["endLon"]
                                ),  # end→end
                                _haversine_distance(
                                    seg1["startLat"], seg1["startLon"],
                                    seg2["startLat"], seg2["startLon"]
                                )  # start→start
                            ]
                            
                            # En kısa mesafeyi al
                            min_distance = min(distances)
                            
                            processed.add(pair)
                            
                            if min_distance <= threshold:
                                edges_to_insert.append({
                                    "seg1": id1,
                                    "seg2": id2,
                                    "distance": min_distance
                                })
            
            # Batch insert (UNDIRECTED - tek yön yeter, Neo4j otomatik iki yönlü arama yapar)
            if edges_to_insert:
                print(f"\nℹ️  {len(edges_to_insert)} edge ekleniyor (batch mode)...")
                
                # Cypher batch upsert - TEK YÖN (ArangoDB/TigerGraph ile tutarlı)
                cypher_batch = """
                    UNWIND $edges AS edge
                    MATCH (a:Segment {segmentId: edge.seg1})
                    MATCH (b:Segment {segmentId: edge.seg2})
                    MERGE (a)-[r:CONNECTS_TO]-(b)
                    ON CREATE SET r.distance = edge.distance
                    ON MATCH SET r.distance = edge.distance
                """
                
                batch_size = 1000
                total_inserted = 0
                for i in range(0, len(edges_to_insert), batch_size):
                    batch = edges_to_insert[i:i+batch_size]
                    session.run(cypher_batch, edges=batch)
                    total_inserted += len(batch)
                    if i % 5000 == 0 and i > 0:
                        print(f"  Progress: {total_inserted:,}/{len(edges_to_insert):,} edge eklendi...")
                
                print(f"✅ {len(edges_to_insert):,} CONNECTS_TO ilişkisi eklendi!")
            else:
                print("ℹ️  Eşik değeri içinde edge bulunamadı")
    
    finally:
        driver.close()
    
    print()

def _haversine_distance(lat1, lon1, lat2, lon2):
    """
    İki GPS koordinatı arasındaki mesafeyi hesapla (metre)
    
    Args:
        lat1, lon1: İlk nokta (derece)
        lat2, lon2: İkinci nokta (derece)
    
    Returns:
        float: Mesafe (metre)
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Yarıçap (metre)
    R = 6371000
    
    # Derece → radyan
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    # Haversine formülü
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-schema", action="store_true")

    # GeoJSON
    ap.add_argument("--load-segments", type=str, help="edges_static.geojson")
    # --build-next-to removed (deprecated, use CONNECTS_TO instead)
    ap.add_argument("--load-measure", type=str, help="flow_YYYYMMDD_HHMM.geojson")
    ap.add_argument("--ts", type=str, help="ISO time for measure (optional)")

    # Parquet
    ap.add_argument("--load-parquet", type=str, help="Path to a Parquet file (time-series)")
    ap.add_argument("--load-parquet-dir", type=str, help="Ingest all *.parquet under directory (recursive)")
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--col-map", type=str, default="", help="Rename columns, e.g. 'segmentId=seg_id,ts=timestamp,freeFlow=ff'")
    ap.add_argument("--set-provider", type=str, default=None, help="Default provider if parquet column missing/NA (e.g., 'here')")
    ap.add_argument("--set-source", type=str, default=None, help="Default source if parquet column missing/NA (e.g., 'archive')")

    args = ap.parse_args()
    did = False

    if args.init_schema:
        cmd_init_schema(); did = True
    if args.load_segments:
        cmd_load_segments(args.load_segments); did = True
    # args.build_next_to removed (deprecated)
    if args.load_measure:
        cmd_load_measure(args.load_measure, ts=args.ts); did = True

    # parquet path(s)
    col_map = _parse_col_map(args.col_map)
    if args.load_parquet:
        cmd_load_parquet_file(args.load_parquet, batch_size=args.batch_size,
                              col_map=col_map, set_provider=args.set_provider, set_source=args.set_source)
        did = True
    if args.load_parquet_dir:
        cmd_load_parquet_dir(args.load_parquet_dir, batch_size=args.batch_size,
                             col_map=col_map, set_provider=args.set_provider, set_source=args.set_source)
        did = True

    if not did:
        print(__doc__)

if __name__ == "__main__":
    main()
