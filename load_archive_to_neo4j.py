#!/usr/bin/env python3
"""
load_archive_to_neo4j.py
-------------------------
Archive klasöründeki geçmiş GeoJSON dosyalarını Neo4j'ye yükler

Kullanım:
    # Tüm archive dosyalarını yükle
    python load_archive_to_neo4j.py
    
    # Sadece belirli tarih
    python load_archive_to_neo4j.py --date 20251127
    
    # Dry-run (yüklemeden önce test)
    python load_archive_to_neo4j.py --dry-run
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from neo4j import GraphDatabase
from dotenv import load_dotenv

# .env yükle
ENV_PATH = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)


class ArchiveLoader:
    """Archive GeoJSON → Neo4j yükleyici"""
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_pass: str = None,
        neo4j_database: str = None
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_pass = neo4j_pass or os.getenv("NEO4J_PASS", "123456789")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver = None
    
    def connect(self):
        """Neo4j bağlantısı"""
        print(f"📡 Neo4j'ye bağlanılıyor: {self.neo4j_uri}")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_pass)
        )
        
        # Bağlantı testi
        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        
        print("  ✓ Bağlantı başarılı!")
    
    def close(self):
        """Bağlantıyı kapat"""
        if self.driver:
            self.driver.close()
    
    def init_schema(self):
        """Schema ve index'leri oluştur"""
        print("\n🔧 Schema oluşturuluyor...")
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Constraint'ler (unique)
            session.run("""
                CREATE CONSTRAINT segment_id IF NOT EXISTS
                FOR (s:Segment) REQUIRE s.segmentId IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT measure_id IF NOT EXISTS
                FOR (m:Measure) REQUIRE m.measureId IS UNIQUE
            """)
            
            # Index'ler
            session.run("""
                CREATE INDEX measure_timestamp IF NOT EXISTS
                FOR (m:Measure) ON (m.timestamp)
            """)
            
            session.run("""
                CREATE INDEX segment_coords IF NOT EXISTS
                FOR (s:Segment) ON (s.start_lat, s.start_lon, s.end_lat, s.end_lon)
            """)
        
        print("  ✓ Schema hazır!")
    
    def load_geojson(self, file_path: Path, dry_run: bool = False) -> Dict:
        """Bir GeoJSON dosyasını yükle"""
        
        # Dosyadan timestamp'i parse et: flow_20251127_1542.geojson → 2025-11-27T15:42:00Z
        filename = file_path.stem  # flow_20251127_1542
        parts = filename.split('_')
        date_str = parts[1]  # 20251127
        time_str = parts[2]  # 1542
        
        timestamp = datetime.strptime(
            f"{date_str}_{time_str}", 
            "%Y%m%d_%H%M"
        ).strftime("%Y-%m-%dT%H:%M:00Z")
        
        # GeoJSON oku
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        
        if not features:
            return {'segments': 0, 'measures': 0, 'timestamp': timestamp}
        
        print(f"\n  📄 {file_path.name}")
        print(f"     Timestamp: {timestamp}")
        print(f"     Features: {len(features)}")
        
        if dry_run:
            return {'segments': len(features), 'measures': len(features), 'timestamp': timestamp}
        
        # Batch veri hazırla
        segments_data = []
        measures_data = []
        
        for feature in features:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [])
            
            if len(coords) < 2:
                continue
            
            # Segment ID (normalize et: edge:xxx:001 → edge:xxx)
            # Archive GeoJSON'da field adı 'edge_id' (küçük harf)
            segment_id = props.get('edge_id', props.get('SEGMENT_ID', ''))
            if ':' in segment_id:
                parts = segment_id.split(':')
                if len(parts) >= 2:
                    segment_id = f"{parts[0]}:{parts[1]}"
            
            # Koordinatlar
            start_lon, start_lat = coords[0]
            end_lon, end_lat = coords[-1]
            
            segments_data.append({
                'sid': segment_id,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'length': props.get('length_m', props.get('LENGTH', 0.0))
            })
            
            measures_data.append({
                'sid': segment_id,
                'mid': f"{segment_id}_{timestamp}",
                'ts': timestamp,
                'speed': props.get('speed', props.get('SPEED', 0.0)),
                'ff': props.get('freeFlow', props.get('FREE_FLOW', props.get('speed', 0.0))),
                'jf': props.get('jamFactor', props.get('JAM_FACTOR', 0.0)),
                'conf': props.get('confidence', props.get('CONFIDENCE', 0.0)),
                'trav': props.get('traversability', props.get('TRAVERSABILITY', 'OPEN'))
            })
        
        # Neo4j'ye batch yükle (çok daha hızlı!)
        with self.driver.session(database=self.neo4j_database) as session:
            # 1. Segment'leri batch yükle
            session.run("""
                UNWIND $segments AS seg
                MERGE (s:Segment {segmentId: seg.sid})
                ON CREATE SET
                    s.start_lat = seg.start_lat,
                    s.start_lon = seg.start_lon,
                    s.end_lat = seg.end_lat,
                    s.end_lon = seg.end_lon,
                    s.length_m = seg.length,
                    s.created_at = datetime()
                ON MATCH SET
                    s.updated_at = datetime()
            """, {'segments': segments_data})
            
            segments_created = len(segments_data)
            
            # 2. Measure'ları batch yükle
            session.run("""
                UNWIND $measures AS meas
                MATCH (s:Segment {segmentId: meas.sid})
                MERGE (m:Measure {measureId: meas.mid})
                ON CREATE SET
                    m.timestamp = meas.ts,
                    m.speed = meas.speed,
                    m.freeFlow = meas.ff,
                    m.jamFactor = meas.jf,
                    m.confidence = meas.conf,
                    m.traversability = meas.trav,
                    m.created_at = datetime()
                MERGE (s)-[:AT_TIME]->(m)
            """, {'measures': measures_data})
            
            measures_created = len(measures_data)
        
        print(f"     ✓ {segments_created} segments, {measures_created} measures")
        
        return {
            'segments': segments_created,
            'measures': measures_created,
            'timestamp': timestamp
        }
    
    def build_topology(self):
        """CONNECTS_TO ilişkilerini oluştur (4-way distance check)"""
        print("\n🔗 CONNECTS_TO topology oluşturuluyor...")
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Önce eski CONNECTS_TO'ları temizle
            session.run("MATCH ()-[r:CONNECTS_TO]-() DELETE r")
            
            # 4-way haversine check (existing code'dan)
            result = session.run("""
                MATCH (a:Segment), (b:Segment)
                WHERE a.segmentId < b.segmentId
                WITH a, b,
                     point({latitude: a.end_lat, longitude: a.end_lon}) AS a_end,
                     point({latitude: a.start_lat, longitude: a.start_lon}) AS a_start,
                     point({latitude: b.end_lat, longitude: b.end_lon}) AS b_end,
                     point({latitude: b.start_lat, longitude: b.start_lon}) AS b_start
                WITH a, b,
                     point.distance(a_end, b_start) AS dist_a_end_to_b_start,
                     point.distance(a_start, b_end) AS dist_a_start_to_b_end,
                     point.distance(a_end, b_end) AS dist_a_end_to_b_end,
                     point.distance(a_start, b_start) AS dist_a_start_to_b_start
                WHERE dist_a_end_to_b_start < 50 
                   OR dist_a_start_to_b_end < 50
                   OR dist_a_end_to_b_end < 50
                   OR dist_a_start_to_b_start < 50
                WITH a, b,
                     CASE 
                       WHEN dist_a_end_to_b_start < 50 THEN dist_a_end_to_b_start
                       WHEN dist_a_start_to_b_end < 50 THEN dist_a_start_to_b_end
                       WHEN dist_a_end_to_b_end < 50 THEN dist_a_end_to_b_end
                       ELSE dist_a_start_to_b_start
                     END AS min_dist
                MERGE (a)-[r:CONNECTS_TO]-(b)
                SET r.distance = min_dist
                RETURN COUNT(r) AS edges_created
            """)
            
            count = result.single()['edges_created']
            print(f"  ✓ {count} CONNECTS_TO edge oluşturuldu")


def main():
    parser = argparse.ArgumentParser(description='Archive GeoJSON → Neo4j loader')
    
    parser.add_argument('--archive_dir', type=str, default='archive',
                        help='Archive dizini (default: archive)')
    parser.add_argument('--date', type=str, default=None,
                        help='Sadece belirli tarih (örn: 20251127)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Yüklemeden önce test et')
    parser.add_argument('--skip-topology', action='store_true',
                        help='CONNECTS_TO oluşturma (hızlı test)')
    
    # Neo4j credentials (override .env)
    parser.add_argument('--neo4j-uri', type=str, default=None,
                        help='Neo4j URI (default: .env)')
    parser.add_argument('--neo4j-user', type=str, default=None,
                        help='Neo4j user (default: .env)')
    parser.add_argument('--neo4j-pass', type=str, default=None,
                        help='Neo4j password (default: .env)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📦 Archive GeoJSON → Neo4j Loader")
    print("="*70 + "\n")
    
    # Archive dizini
    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"❌ Archive dizini bulunamadı: {archive_dir}")
        sys.exit(1)
    
    # GeoJSON dosyalarını bul
    geojson_files = sorted(archive_dir.glob("flow_*.geojson"))
    
    # Tarih filtresi
    if args.date:
        geojson_files = [f for f in geojson_files if args.date in f.stem]
    
    if not geojson_files:
        print(f"❌ GeoJSON dosyası bulunamadı!")
        sys.exit(1)
    
    print(f"📁 {len(geojson_files)} dosya bulundu")
    
    # Tarih grupları
    dates = {}
    for f in geojson_files:
        date = f.stem.split('_')[1]  # 20251127
        dates[date] = dates.get(date, 0) + 1
    
    print(f"📅 Tarihler:")
    for date, count in sorted(dates.items()):
        formatted = datetime.strptime(date, "%Y%m%d").strftime("%d %B %Y")
        print(f"  - {formatted}: {count} dosya")
    
    if args.dry_run:
        print("\n⚠️  DRY-RUN mode - Neo4j'ye yüklenmeyecek!")
        total_segments = len(geojson_files)
        total_measures = len(geojson_files)
        print(f"\n📊 Tahmini:")
        print(f"  - ~{total_segments} unique segment")
        print(f"  - ~{total_measures} measure")
        print(f"\n✅ Gerçek yükleme için --dry-run olmadan çalıştırın")
        return
    
    # Loader oluştur
    loader = ArchiveLoader(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass
    )
    
    try:
        # Bağlan
        loader.connect()
        
        # Schema
        loader.init_schema()
        
        # Dosyaları yükle
        print("\n📤 Dosyalar yükleniyor...")
        
        total_segments = 0
        total_measures = 0
        
        for i, file_path in enumerate(geojson_files, 1):
            print(f"\n[{i}/{len(geojson_files)}]", end='')
            stats = loader.load_geojson(file_path)
            
            total_segments = max(total_segments, stats['segments'])  # max çünkü MERGE kullanıyoruz
            total_measures += stats['measures']
        
        print(f"\n\n✅ Tüm dosyalar yüklendi!")
        print(f"  - Segments: ~{total_segments} unique")
        print(f"  - Measures: {total_measures} total")
        
        # Topology
        if not args.skip_topology:
            loader.build_topology()
        else:
            print("\n⚠️  CONNECTS_TO atlandı (--skip-topology)")
        
        # Özet
        print("\n" + "="*70)
        print("📊 Neo4j Durum:")
        print("="*70)
        
        with loader.driver.session(database=loader.neo4j_database) as session:
            # Segment count
            result = session.run("MATCH (s:Segment) RETURN COUNT(s) AS count")
            seg_count = result.single()['count']
            
            # Measure count
            result = session.run("MATCH (m:Measure) RETURN COUNT(m) AS count")
            meas_count = result.single()['count']
            
            # CONNECTS_TO count
            result = session.run("MATCH ()-[r:CONNECTS_TO]-() RETURN COUNT(r) AS count")
            edge_count = result.single()['count']
            
            # Timestamp range
            result = session.run("""
                MATCH (m:Measure)
                RETURN MIN(m.timestamp) AS min_ts, MAX(m.timestamp) AS max_ts
            """)
            rec = result.single()
            min_ts = rec['min_ts']
            max_ts = rec['max_ts']
            
            print(f"\n  🔵 Segments: {seg_count:,}")
            print(f"  🔵 Measures: {meas_count:,}")
            print(f"  🔗 CONNECTS_TO: {edge_count:,}")
            print(f"  📅 Tarih aralığı: {min_ts} → {max_ts}")
        
        print("\n✅ Neo4j hazır! Şimdi model eğitimi yapabilirsiniz:")
        print("   python src/gnn/train.py --epochs 50")
        print()
        
    finally:
        loader.close()


if __name__ == "__main__":
    main()
