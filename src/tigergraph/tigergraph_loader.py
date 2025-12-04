#!/usr/bin/env python3
"""
tigergraph_loader.py
--------------------
TigerGraph veri yükleyici - Neo4j loader ile aynı interface

KULLANIM:
    from src.tigergraph.tigergraph_loader import TigerGraphLoader
    
    loader = TigerGraphLoader()
    loader.init_schema()
    loader.load_segments("data/processed/edges_static.geojson")
    loader.load_measurements("archive/flow_20251027_1244.geojson", "2025-10-27T12:44:13+00:00")
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as dtparser
from dotenv import load_dotenv

try:
    import pyTigerGraph as tg
except ImportError:
    print("⚠️  TigerGraph client not installed. Install: pip install pyTigerGraph")
    raise

# .env yükle
ENV_PATH = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)

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

# TigerGraph bağlantı bilgileri
TIGER_HOST = os.getenv("TIGER_HOST", "http://127.0.0.1")
TIGER_REST_PORT = int(os.getenv("TIGER_REST_PORT", "9000"))
TIGER_GSQL_PORT = int(os.getenv("TIGER_GSQL_PORT", "14240"))
TIGER_USERNAME = os.getenv("TIGER_USERNAME", "tigergraph")
TIGER_PASSWORD = os.getenv("TIGER_PASSWORD", "tigergraph")
TIGER_GRAPHNAME = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")


class TigerGraphLoader:
    """TigerGraph veri yükleyici (Neo4j loader API uyumlu)"""
    
    def __init__(self):
        """TigerGraph bağlantısı kur"""
        # REST API bağlantısı
        self.conn = tg.TigerGraphConnection(
            host=TIGER_HOST,
            restppPort=TIGER_REST_PORT,
            gsPort=TIGER_GSQL_PORT,
            username=TIGER_USERNAME,
            password=TIGER_PASSWORD,
            graphname=TIGER_GRAPHNAME
        )
        
        print(f"ℹ️  TigerGraph bağlantısı: {TIGER_HOST}:{TIGER_REST_PORT}")
        print(f"ℹ️  Graph adı: {TIGER_GRAPHNAME}")
        
    def init_schema(self):
        """
        GSQL ile schema oluştur
        
        TigerGraph'ta:
        - Vertex (node) tipi tanımla
        - Edge (ilişki) tipi tanımla
        - Graph oluştur
        
        AKILLI KONTROL: Eğer schema zaten varsa, önce siler sonra yeniden oluşturur
        """
        print("=" * 70)
        print("🔧 TigerGraph Schema Oluşturuluyor...")
        print("=" * 70)
        
        # ÖNCE GRAPH VARMI KONTROL ET (Neo4j/ArangoDB gibi)
        try:
            # Graph'ın mevcut olup olmadığını kontrol et
            graphs = self.conn.gsql("ls")
            if TIGER_GRAPHNAME in graphs:
                print(f"ℹ️  Graph zaten mevcut: {TIGER_GRAPHNAME}")
                print(f"ℹ️  Schema atlanıyor (veri korunuyor)")
                return
            else:
                print(f"🔧 Graph bulunamadı, yeni schema oluşturuluyor...")
        except Exception as e:
            print(f"⚠️  Graph kontrol hatası: {str(e)[:100]}")
            print(f"🔧 Schema oluşturmaya devam ediliyor...")
        
        # 2. GSQL schema tanımı - SCHEMA_CHANGE JOB ile (reset_tigergraph_schema.py ile aynı)
        gsql_schema = f"""
CREATE GRAPH {TIGER_GRAPHNAME}()

USE GRAPH {TIGER_GRAPHNAME}

CREATE SCHEMA_CHANGE JOB traffic_schema FOR GRAPH {TIGER_GRAPHNAME} {{
    
    // Vertex Types
    ADD VERTEX Segment (
        PRIMARY_ID segmentId STRING,
        hereSegmentId STRING,
        osmWayId STRING,
        frc INT,
        lengthM DOUBLE,
        name STRING,
        geom STRING,
        lat DOUBLE,
        lon DOUBLE,
        startLat DOUBLE,
        startLon DOUBLE,
        endLat DOUBLE,
        endLon DOUBLE
    ) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true";
    
    ADD VERTEX Measure (
        PRIMARY_ID measureId STRING,
        segmentId STRING,
        timestamp DATETIME,
        jamFactor DOUBLE,
        speed DOUBLE,
        freeFlow DOUBLE,
        confidence DOUBLE
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";
    
    // Edge Types
    ADD UNDIRECTED EDGE CONNECTS_TO (
        FROM Segment,
        TO Segment,
        distance DOUBLE
    );
    
    ADD DIRECTED EDGE AT_TIME (
        FROM Segment,
        TO Measure
    );
}}

RUN SCHEMA_CHANGE JOB traffic_schema
DROP JOB traffic_schema
"""
        
        try:
            # 3. Schema'yı uygula - reset_tigergraph_schema.py ile aynı yöntem
            result = self.conn.gsql(gsql_schema)
            print("✅ GSQL schema uygulandı")
            print(f"ℹ️  Graph oluşturuldu: {TIGER_GRAPHNAME}")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                print(f"⚠️  Schema zaten mevcut!")
                print(f"ℹ️  Manuel temizlik: python reset_tigergraph_schema.py")
            elif "404" in error_msg or "not found" in error_msg:
                print(f"⚠️  TigerGraph REST API yanıt vermiyor")
                print(f"ℹ️  Container başlatılıyor olabilir, birkaç dakika bekleyin")
                print(f"ℹ️  Veya: docker restart tigergraph")
            else:
                print(f"⚠️  Schema hatası: {e}")
        
        print()
    
    def load_segments(self, geojson_path):
        """
        Segment verilerini yükle (statik yol parçaları)
        
        Args:
            geojson_path: edges_static.geojson dosya yolu
        """
        print("=" * 70)
        print(f"📦 Segment yükleniyor: {geojson_path}")
        print("=" * 70)
        
        # GeoJSON oku
        geojson = json.loads(Path(geojson_path).read_text(encoding='utf-8'))
        features = geojson.get("features", geojson if isinstance(geojson, list) else [])
        
        # Segment verilerini hazırla
        segments = []
        for feat in features:
            props = feat.get("properties", {})
            coords = feat["geometry"]["coordinates"]
            
            # Segment ID
            raw_seg_id = props.get("edge_id") or props.get("segmentId") or props.get("segment_id")
            if not raw_seg_id:
                raw_seg_id = hashlib.sha1(json.dumps(coords).encode()).hexdigest()[:20]
            
            # NORMALIZE: edge:xxx:001 -> edge:xxx (sub-segment'leri ana yol ID'sine dönüştür)
            normalized_seg_id = normalize_segment_id(raw_seg_id)
            
            # TigerGraph için : karakterini _ ile değiştir (TigerGraph : ayırıcı olarak kullanır)
            seg_id = normalized_seg_id.replace(":", "_")
            
            # Segment başlangıç ve bitiş koordinatları
            start_lat = coords[0][1] if coords else None
            start_lon = coords[0][0] if coords else None
            end_lat = coords[-1][1] if coords else None
            end_lon = coords[-1][0] if coords else None
            
            # WKT geometry - TÜM koordinatları kullan
            coord_pairs = [f"{lon} {lat}" for lon, lat in coords]
            wkt_geom = f"LINESTRING ({', '.join(coord_pairs)})"
            
            segment = {
                "segmentId": seg_id,
                "hereSegmentId": props.get("hereSegmentId") or "",
                "osmWayId": props.get("osmWayId") or "",
                "frc": props.get("frc") or 0,
                "lengthM": props.get("length_m") or props.get("lengthM") or 0.0,
                "name": props.get("desc") or props.get("name") or props.get("road_name") or "",
                "geom": wkt_geom,
                "lat": start_lat or 0.0,
                "lon": start_lon or 0.0,
                "startLat": start_lat or 0.0,
                "startLon": start_lon or 0.0,
                "endLat": end_lat or 0.0,
                "endLon": end_lon or 0.0
            }
            
            segments.append(segment)
        
        # Batch upsert - MANUAL REST API (pyTigerGraph 1.9.1 uyumsuzluk workaround)
        if segments:
            try:
                # ⚠️ WORKAROUND: pyTigerGraph 1.9.1 batch upsert çalışmıyor (TG 4.2.2)
                # Direct REST API kullan - hızlı VE çalışıyor
                print("ℹ️  Manual REST API batch upsert (pyTigerGraph workaround)")
                
                import requests
                batch_size = 500
                total_inserted = 0
                failed_batches = 0
                url = f"{self.conn.restppUrl}/graph/{self.conn.graphname}"
                
                print(f"ℹ️  Batch yükleme başlıyor: {len(segments)} segment, batch_size={batch_size}")
                print(f"ℹ️  URL: {url}")
                
                for i in range(0, len(segments), batch_size):
                    batch = segments[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    
                    # REST API format: {"vertices": {"Segment": {"id1": {...}, "id2": {...}}}}
                    vertices_dict = {}
                    for seg in batch:
                        vertex_id = seg["segmentId"]
                        # Attribute'ları REST API formatına çevir: {"value": x}
                        attrs = {k: {"value": v} for k, v in seg.items()}
                        vertices_dict[vertex_id] = attrs
                    
                    payload = {"vertices": {"Segment": vertices_dict}}
                    
                    try:
                        response = requests.post(url, json=payload, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            # ⚠️ ERROR CHECK: API'den error geldi mi?
                            if result.get("error"):
                                failed_batches += 1
                                print(f"❌ Batch {batch_num}/{(len(segments)-1)//batch_size + 1} TigerGraph API hatası:")
                                print(f"   Message: {result.get('message')}")
                                print(f"   Code: {result.get('code')}")
                                print(f"   Batch size: {len(batch)} segment")
                                continue
                            accepted = result.get("results", [{}])[0].get("accepted_vertices", 0)
                            total_inserted += accepted
                            if batch_num % 2 == 0:  # Her 2 batch'te bir progress
                                print(f"✓ Batch {batch_num}: {accepted} segment eklendi (Toplam: {total_inserted})")
                        else:
                            failed_batches += 1
                            print(f"❌ Batch {batch_num} HTTP hatası: {response.status_code}")
                            print(f"   Body: {response.text[:500]}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "404" in error_msg:
                            print(f"⚠️  TigerGraph schema bulunamadı!")
                            print(f"ℹ️  Önce schema oluşturun: python src/pipeline/multi_db_loader.py --init-schema")
                            break
                        else:
                            print(f"⚠️  Batch hatası: {str(e)}")
                    
                
                # Final rapor
                print(f"\n📊 Yükleme Özeti:")
                print(f"   Toplam segment: {len(segments):,}")
                print(f"   Başarıyla yüklenen: {total_inserted:,}")
                print(f"   Başarısız batch: {failed_batches}")
                
                if total_inserted > 0:
                    print(f"✅ {total_inserted:,} segment yüklendi/güncellendi")
                else:
                    print(f"❌ HİÇ SEGMENT YÜKLENEMEDİ!")
            except Exception as e:
                print(f"⚠️  Toplu yükleme hatası: {e}")
        else:
            print("⚠️  Hiç segment bulunamadı!")
        
        print()
    
    def load_measurements(self, geojson_path, timestamp=None):
        """
        Trafik ölçümlerini yükle (dinamik veriler)
        
        Args:
            geojson_path: flow_YYYYMMDD_HHMM.geojson dosya yolu
            timestamp: ISO format timestamp (None ise dosya adından çıkar)
        """
        print("=" * 70)
        print(f"📊 Measure yükleniyor: {geojson_path}")
        print("=" * 70)
        
        # Timestamp belirle
        if timestamp:
            dt = dtparser.isoparse(timestamp)
        else:
            import re
            filename = Path(geojson_path).name
            match = re.search(r'(\d{8})[_-](\d{4})', filename)
            if match:
                ymd, hm = match.group(1), match.group(2)
                dt = datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            else:
                raise ValueError(f"Timestamp dosya adından çıkarılamadı: {geojson_path}. timestamp parametresi kullanın veya dosya adını 'flow_YYYYMMDD_HHMM.geojson' formatında adlandırın.")
        
        timestamp_str = dt.isoformat()
        print(f"ℹ️  Timestamp: {timestamp_str}")
        
        # GeoJSON oku
        geojson = json.loads(Path(geojson_path).read_text(encoding='utf-8'))
        features = geojson.get("features", geojson if isinstance(geojson, list) else [])
        
        # Measure verilerini hazırla
        measures = []
        at_time_edges = []
        
        for feat in features:
            props = feat.get("properties", {})
            
            # Segment ID bul
            raw_seg_id = props.get("segmentId") or props.get("segment_id") or props.get("edge_id")
            if not raw_seg_id:
                coords = feat["geometry"]["coordinates"]
                raw_seg_id = hashlib.sha1(json.dumps(coords).encode()).hexdigest()[:20]
            
            # NORMALIZE: edge:xxx:001 -> edge:xxx (measure'lar segment ile eşleşmeli)
            normalized_seg_id = normalize_segment_id(raw_seg_id)
            
            # TigerGraph için : karakterini _ ile değiştir
            seg_id = normalized_seg_id.replace(":", "_")
            
            # Measure ID oluştur
            measure_id = f"{seg_id}_{timestamp_str}".replace("+", "_").replace(":", "_").replace(".", "_")
            
            # Trafik değerleri - 0 değerini korumak için None check kullan
            speed_val = props.get("speed")
            if speed_val is None:
                speed_val = props.get("currentSpeed") or props.get("speed_kmh") or 0.0
            
            freeflow_val = props.get("freeFlow")
            if freeflow_val is None:
                freeflow_val = props.get("freeFlowSpeed") or props.get("free_flow_kmh") or 0.0
            
            jamfactor_val = props.get("jamFactor")
            if jamfactor_val is None:
                jamfactor_val = props.get("jam_factor")
            if jamfactor_val is None:
                jamfactor_val = 0.0
            
            measure = {
                "measureId": measure_id,
                "segmentId": seg_id,
                "timestamp": timestamp_str,
                "speed": speed_val,
                "freeFlow": freeflow_val,
                "jamFactor": jamfactor_val,
                "confidence": props.get("confidence") or 0.0
            }
            
            measures.append(measure)
            at_time_edges.append((seg_id, measure_id))
        
        # Batch insert - OPTIMIZED!
        if measures:
            try:
                # 1) Measure vertex'leri batch ekle
                batch_size = 1000
                total_measures = 0
                
                for i in range(0, len(measures), batch_size):
                    batch = measures[i:i+batch_size]
                    vertices_list = [(m["measureId"], m) for m in batch]
                    
                    try:
                        self.conn.upsertVertices("Measure", vertices_list)
                        total_measures += len(batch)
                        
                        if (i // batch_size + 1) % 5 == 0:
                            print(f"  Measure progress: {min(i+batch_size, len(measures)):,}/{len(measures):,}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "graph schema not found" in error_msg or "404" in error_msg:
                            print(f"⚠️  TigerGraph schema bulunamadı - ölçümler atlanıyor")
                            break
                        else:
                            print(f"⚠️  Measure batch hatası: {str(e)}")
                            # Fallback
                            for m in batch:
                                try:
                                    self.conn.upsertVertex("Measure", m["measureId"], attributes=m)
                                    total_measures += 1
                                except:
                                    pass
                
                # 2) AT_TIME edge'leri batch ekle
                total_edges = 0
                if total_measures > 0:
                    for i in range(0, len(at_time_edges), batch_size):
                        batch = at_time_edges[i:i+batch_size]
                        edges_list = [(seg_id, measure_id, {}) for seg_id, measure_id in batch]
                        
                        try:
                            self.conn.upsertEdges("Segment", "AT_TIME", "Measure", edges_list)
                            total_edges += len(batch)
                            
                            if (i // batch_size + 1) % 5 == 0:
                                print(f"  AT_TIME progress: {min(i+batch_size, len(at_time_edges)):,}/{len(at_time_edges):,}")
                        except Exception as e:
                            print(f"⚠️  AT_TIME batch hatası: {str(e)}")
                            # Fallback
                            for seg_id, measure_id in batch:
                                try:
                                    self.conn.upsertEdge("Segment", seg_id, "AT_TIME", "Measure", measure_id)
                                    total_edges += 1
                                except:
                                    pass
                
                if total_measures > 0:
                    print(f"✅ {total_measures:,} measure + {total_edges:,} AT_TIME ilişkisi yüklendi")
            except Exception as e:
                print(f"⚠️  Toplu yükleme hatası: {e}")
        else:
            print(f"⚠️  Hiç measure bulunamadı")
        print()
    
    def build_connects_to(self, threshold=12.0):
        """
        CONNECTS_TO ilişkilerini oluştur (spatial topology)
        
        Python ile spatial grid optimizasyonu (GSQL query atlanıyor - çok yavaş)
        
        Args:
            threshold: Maksimum mesafe (metre)
        """
        print("=" * 70)
        print(f"🔗 CONNECTS_TO ilişkileri oluşturuluyor (threshold={threshold}m)...")
        print("=" * 70)
        
        # PERFORMANS UYARISI
        print("⚠️  TigerGraph CONNECTS_TO ekleme YAVAŞ olabilir (REST API limiti)")
        print("ℹ️  Alternatif: TigerGraph'ı geçici olarak devre dışı bırakın")
        print("    (ACTIVE_DATABASES='neo4j,arangodb' olarak ayarlayın)")
        print()
        
        # GSQL query'yi ATLAYIP direkt Python spatial grid kullan
        print("ℹ️  Python ile spatial grid optimizasyonlu edge ekleniyor...")
        self._build_connects_to_direct(threshold)
        
        print()
    
    def _build_connects_to_direct(self, threshold=12.0):
        """
        Python'dan direkt edge ekle - SPATIAL GRID OPTİMİZASYONU
        4 yönlü kontrol: end→start, start→end, end→end, start→start
        """
        print("ℹ️  Python ile spatial grid optimizasyonlu edge ekleniyor (4 yönlü kontrol)...")
        
        # Tüm segmentleri al (start ve end koordinatları ile)
        # CRITICAL: limit=999999 ile TÜM segment'leri al (varsayılan limit ~1000)
        segments = self.conn.getVertices("Segment", limit=999999)
        
        if not segments:
            print("⚠️  Segment bulunamadı")
            return
        
        # Her segment için koordinatları al
        seg_coords = {}
        for seg in segments:
            seg_id = seg.get("v_id")
            # TigerGraph ID'den geri dönüşüm gerekmez (zaten _ ile saklandı)
            attrs = seg.get("attributes", {})
            start_lat = attrs.get("startLat", 0)
            start_lon = attrs.get("startLon", 0)
            end_lat = attrs.get("endLat", 0)
            end_lon = attrs.get("endLon", 0)
            
            if start_lat != 0 and start_lon != 0 and end_lat != 0 and end_lon != 0:
                seg_coords[seg_id] = {
                    "startLat": start_lat,
                    "startLon": start_lon,
                    "endLat": end_lat,
                    "endLon": end_lon
                }
        
        print(f"ℹ️  {len(seg_coords)} segment için koordinat alındı")
        
        # SPATIAL GRID - sadece yakın hücreler (4 yönlü için hem start hem end)
        grid_size = (threshold * 2) / 111320.0
        
        # Segment'leri grid hücrelerine yerleştir
        grid = {}
        for seg_id, coords in seg_coords.items():
            # Start noktasını grid'e ekle
            grid_x_start = int(coords["startLat"] / grid_size)
            grid_y_start = int(coords["startLon"] / grid_size)
            cell_start = (grid_x_start, grid_y_start)
            if cell_start not in grid:
                grid[cell_start] = []
            if (seg_id, coords) not in [(s[0], s[1]) for s in grid[cell_start]]:
                grid[cell_start].append((seg_id, coords))
            
            # End noktasını grid'e ekle
            grid_x_end = int(coords["endLat"] / grid_size)
            grid_y_end = int(coords["endLon"] / grid_size)
            cell_end = (grid_x_end, grid_y_end)
            if cell_end not in grid:
                grid[cell_end] = []
            if (seg_id, coords) not in [(s[0], s[1]) for s in grid[cell_end]]:
                grid[cell_end].append((seg_id, coords))
        
        print(f"ℹ️  {len(grid)} grid hücresine dağıtıldı")
        
        # Edge'leri topla (önce hepsini hesapla, sonra toplu ekle)
        edges_to_insert = []
        processed = set()
        total_cells = len(grid)
        current_cell = 0
        
        for cell, segments_in_cell in grid.items():
            current_cell += 1
            if current_cell % 100 == 0:
                print(f"  Progress: {current_cell}/{total_cells} hücre işlendi ({len(edges_to_insert):,} edge bulundu)")
            
            gx, gy = cell
            # Komşu hücreler (3x3 = 9 hücre)
            neighbor_cells = [
                (gx-1, gy-1), (gx-1, gy), (gx-1, gy+1),
                (gx, gy-1), (gx, gy), (gx, gy+1),
                (gx+1, gy-1), (gx+1, gy), (gx+1, gy+1)
            ]
            
            # Bu hücredeki her segment için
            for seg1_id, coords1 in segments_in_cell:
                # Komşu hücrelerdeki segment'leri kontrol et
                for ncell in neighbor_cells:
                    if ncell not in grid:
                        continue
                    for seg2_id, coords2 in grid[ncell]:
                        if seg1_id >= seg2_id:
                            continue
                        
                        pair = (min(seg1_id, seg2_id), max(seg1_id, seg2_id))
                        if pair in processed:
                            continue
                        processed.add(pair)
                        
                        # 4 yönlü kontrol: end→start, start→end, end→end, start→start
                        distances = [
                            self._haversine_distance(
                                coords1["endLat"], coords1["endLon"],
                                coords2["startLat"], coords2["startLon"]
                            ),  # end→start
                            self._haversine_distance(
                                coords1["startLat"], coords1["startLon"],
                                coords2["endLat"], coords2["endLon"]
                            ),  # start→end
                            self._haversine_distance(
                                coords1["endLat"], coords1["endLon"],
                                coords2["endLat"], coords2["endLon"]
                            ),  # end→end
                            self._haversine_distance(
                                coords1["startLat"], coords1["startLon"],
                                coords2["startLat"], coords2["startLon"]
                            )  # start→start
                        ]
                        
                        # En kısa mesafeyi al
                        min_distance = min(distances)
                        
                        if min_distance <= threshold:
                            # Edge'i listeye ekle (henüz DB'ye ekleme)
                            edges_to_insert.append({
                                "seg1": seg1_id,
                                "seg2": seg2_id,
                                "distance": min_distance
                            })
        
        # Edge insert (BATCH MODE - Liste of tuples formatı ile)
        print(f"\nℹ️  {len(edges_to_insert):,} edge DB'ye ekleniyor (batch mode)...")
        if edges_to_insert:
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges_to_insert), batch_size):
                batch = edges_to_insert[i:i+batch_size]
                
                # pyTigerGraph DOĞRU formatı: [(source_id, target_id, {attributes}), ...]
                edges_list = [
                    (edge["seg1"], edge["seg2"], {"distance": edge["distance"]})
                    for edge in batch
                ]
                
                try:
                    # UNDIRECTED edge için tek yön yeter
                    result = self.conn.upsertEdges("Segment", "CONNECTS_TO", "Segment", edges_list)
                    total_inserted += result
                    
                    if (i // batch_size + 1) % 5 == 0:
                        print(f"  Progress: {min(i+batch_size, len(edges_to_insert)):,}/{len(edges_to_insert):,} edge")
                except Exception as e:
                    print(f"⚠️  Batch hatası: {str(e)}")
                    # Fallback: Bu batch'i tek tek ekle
                    for edge in batch:
                        try:
                            self.conn.upsertEdge(
                                "Segment", edge["seg1"],
                                "CONNECTS_TO",
                                "Segment", edge["seg2"],
                                attributes={"distance": edge["distance"]}
                            )
                            total_inserted += 1
                        except:
                            pass
            
            print(f"✅ {total_inserted:,} CONNECTS_TO edge eklendi!")
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
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
    
    def close(self):
        """Bağlantıyı kapat (TigerGraph REST API stateless)"""
        pass


# Test fonksiyonu
def main():
    """Basit test"""
    print("\n" + "="*70)
    print("🧪 TigerGraph Loader Test")
    print("="*70 + "\n")
    
    loader = TigerGraphLoader()
    
    # Schema oluştur
    loader.init_schema()
    
    # Test verisi yükle
    root = Path(__file__).parent.parent.parent
    edges_file = root / "data" / "edges_static.geojson"
    
    if edges_file.exists():
        loader.load_segments(str(edges_file))
    else:
        print(f"⚠️  Test verisi bulunamadı: {edges_file}")
    
    loader.close()
    print("\n✅ Test tamamlandı!\n")


if __name__ == "__main__":
    main()
