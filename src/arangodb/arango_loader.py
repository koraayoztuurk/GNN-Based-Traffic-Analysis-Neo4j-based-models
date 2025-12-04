#!/usr/bin/env python3
"""
arango_loader.py
----------------
ArangoDB veri yükleyici - Neo4j loader ile aynı interface

KULLANIM:
    from src.arangodb.arango_loader import ArangoLoader
    
    loader = ArangoLoader()
    loader.init_schema()
    loader.load_segments("data/edges_static.geojson")
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
    from arango import ArangoClient
except ImportError:
    print("⚠️  ArangoDB client not installed. Install: pip install python-arango")
    raise

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

# .env yükle
ENV_PATH = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)

# ArangoDB bağlantı bilgileri
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "123456789")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE", "traffic_db")


class ArangoLoader:
    """ArangoDB veri yükleyici (Neo4j loader API uyumlu)"""
    
    def __init__(self):
        """ArangoDB bağlantısı kur"""
        self.client = ArangoClient(hosts=ARANGO_HOST)
        self.sys_db = self.client.db('_system', username=ARANGO_USER, password=ARANGO_PASS)
        
        # Veritabanı oluştur (yoksa)
        if not self.sys_db.has_database(ARANGO_DATABASE):
            self.sys_db.create_database(ARANGO_DATABASE)
            print(f"✅ Veritabanı oluşturuldu: {ARANGO_DATABASE}")
        
        # Ana veritabanına bağlan
        self.db = self.client.db(ARANGO_DATABASE, username=ARANGO_USER, password=ARANGO_PASS)
        
    def init_schema(self):
        """
        Collections (tablolar) ve indexes oluştur
        Neo4j'deki Nodes → ArangoDB Collections
        Neo4j'deki Relationships → ArangoDB Edge Collections
        """
        print("=" * 70)
        print("🔧 ArangoDB Schema Oluşturuluyor...")
        print("=" * 70)
        
        # 1. Segment collection (document collection)
        if not self.db.has_collection('Segment'):
            self.segments = self.db.create_collection('Segment')
            print("✅ Collection oluşturuldu: Segment")
        else:
            self.segments = self.db.collection('Segment')
            print("ℹ️  Collection mevcut: Segment")
        
        # 2. Measure collection (document collection)
        if not self.db.has_collection('Measure'):
            self.measures = self.db.create_collection('Measure')
            print("✅ Collection oluşturuldu: Measure")
        else:
            self.measures = self.db.collection('Measure')
            print("ℹ️  Collection mevcut: Measure")
        
        # 3. CONNECTS_TO edge collection (ilişki)
        if not self.db.has_collection('CONNECTS_TO'):
            self.connects_to = self.db.create_collection('CONNECTS_TO', edge=True)
            print("✅ Edge Collection oluşturuldu: CONNECTS_TO")
        else:
            self.connects_to = self.db.collection('CONNECTS_TO')
            print("ℹ️  Edge Collection mevcut: CONNECTS_TO")
        
        # 4. AT_TIME edge collection (temporal ilişki)
        if not self.db.has_collection('AT_TIME'):
            self.at_time = self.db.create_collection('AT_TIME', edge=True)
            print("✅ Edge Collection oluşturuldu: AT_TIME")
        else:
            self.at_time = self.db.collection('AT_TIME')
            print("ℹ️  Edge Collection mevcut: AT_TIME")
        
        # 5. Indexes oluştur (performans için)
        try:
            # Segment için index
            self.segments.add_hash_index(fields=['segmentId'], unique=True)
            print("✅ Index oluşturuldu: Segment.segmentId")
            
            # Measure için composite index
            self.measures.add_hash_index(fields=['segmentId', 'timestamp'], unique=True)
            print("✅ Index oluşturuldu: Measure(segmentId, timestamp)")
            
        except Exception as e:
            print(f"ℹ️  Index zaten mevcut: {e}")
        
        # 6. Graph yapısını oluştur (benchmark için gerekli)
        self._ensure_graph()
        
        print()
        print("✅ Schema hazırlama tamamlandı!")
        print()
    
    def clear_all_data(self):
        """Tüm collection'ları temizle (schema'yı koruyarak)"""
        print("🧹 Collection'lar temizleniyor...")
        self._ensure_collections()
        
        try:
            self.segments.truncate()
            print("  ✅ Segment temizlendi")
        except:
            pass
        
        try:
            self.measures.truncate()
            print("  ✅ Measure temizlendi")
        except:
            pass
        
        try:
            self.connects_to.truncate()
            print("  ✅ CONNECTS_TO temizlendi")
        except:
            pass
        
        try:
            self.at_time.truncate()
            print("  ✅ AT_TIME temizlendi")
        except:
            pass
        
        print("✅ Tüm veriler temizlendi!")
        print()

    def _ensure_collections(self):
        """
        İç kullanım: collection referanslarının mevcut olduğunu doğrula.
        Eğer `init_schema()` bu oturumda çağrılmadıysa, varolan collection'lara bağlanır.
        """
        # Segment
        if not hasattr(self, 'segments'):
            if self.db.has_collection('Segment'):
                self.segments = self.db.collection('Segment')
            else:
                self.segments = self.db.create_collection('Segment')

        # Measure
        if not hasattr(self, 'measures'):
            if self.db.has_collection('Measure'):
                self.measures = self.db.collection('Measure')
            else:
                self.measures = self.db.create_collection('Measure')

        # Edge collections
        if not hasattr(self, 'connects_to'):
            if self.db.has_collection('CONNECTS_TO'):
                self.connects_to = self.db.collection('CONNECTS_TO')
            else:
                self.connects_to = self.db.create_collection('CONNECTS_TO', edge=True)

        if not hasattr(self, 'at_time'):
            if self.db.has_collection('AT_TIME'):
                self.at_time = self.db.collection('AT_TIME')
            else:
                self.at_time = self.db.create_collection('AT_TIME', edge=True)
    
    def _ensure_graph(self):
        """
        Graph yapısını kontrol et ve gerekirse oluştur.
        Benchmark testleri için 'traffic_flow_graph' gerekli.
        """
        graph_name = 'traffic_flow_graph'
        
        try:
            if self.db.has_graph(graph_name):
                print(f"ℹ️  Graph mevcut: {graph_name}")
            else:
                print(f"🔧 Graph oluşturuluyor: {graph_name}")
                
                # Graph oluştur
                graph = self.db.create_graph(graph_name)
                
                # Edge definition ekle (CONNECTS_TO: Segment → Segment)
                graph.create_edge_definition(
                    edge_collection='CONNECTS_TO',
                    from_vertex_collections=['Segment'],
                    to_vertex_collections=['Segment']
                )
                
                print(f"✅ Graph oluşturuldu: {graph_name}")
        except Exception as e:
            print(f"⚠️  Graph kontrol/oluşturma hatası: {e}")
    
    def load_segments(self, geojson_path):
        """
        Segment verilerini yükle (statik yol parçaları)
        
        Args:
            geojson_path: edges_static.geojson dosya yolu
        """
        print("=" * 70)
        print(f"📦 Segment yükleniyor: {geojson_path}")
        print("=" * 70)
        
        # Ensure collections exist in this loader instance
        self._ensure_collections()

        # GeoJSON oku
        geojson = json.loads(Path(geojson_path).read_text(encoding='utf-8'))
        features = geojson.get("features", geojson if isinstance(geojson, list) else [])
        
        # Segment dökümanları hazırla
        segments = []
        for feat in features:
            props = feat.get("properties", {})
            coords = feat["geometry"]["coordinates"]
            
            # Segment ID (edge_id veya segmentId)
            raw_seg_id = props.get("edge_id") or props.get("segmentId") or props.get("segment_id")
            if not raw_seg_id:
                # Fallback: koordinatlardan hash
                raw_seg_id = hashlib.sha1(json.dumps(coords).encode()).hexdigest()[:20]
            
            # NORMALIZE: edge:xxx:001 -> edge:xxx (sub-segment'leri ana yol ID'sine dönüştür)
            seg_id = normalize_segment_id(raw_seg_id)
            
            # WKT geometry oluştur - TÜM koordinatları kullan
            coord_pairs = [f"{lon} {lat}" for lon, lat in coords]
            wkt_geom = f"LINESTRING ({', '.join(coord_pairs)})"
            
            # Key formatını normalize et (: karakterini _ ile değiştir)
            normalized_key = seg_id.replace("+", "_").replace("-", "_").replace(":", "_")
            
            segment = {
                "_key": normalized_key,
                "segmentId": seg_id,
                "hereSegmentId": props.get("hereSegmentId"),
                "osmWayId": props.get("osmWayId"),
                "frc": props.get("frc"),
                "lengthM": props.get("length_m") or props.get("lengthM"),
                "name": props.get("desc") or props.get("name") or props.get("road_name"),
                "geom": wkt_geom,
                # Koordinatlar (GNN için ve 4 yönlü topoloji için)
                "lat": coords[0][1] if coords else None,
                "lon": coords[0][0] if coords else None,
                "startLat": coords[0][1] if coords else None,
                "startLon": coords[0][0] if coords else None,
                "endLat": coords[-1][1] if coords else None,
                "endLon": coords[-1][0] if coords else None,
            }
            
            segments.append(segment)
        
        # Batch insert (upsert)
        if segments:
            for seg in segments:
                try:
                    # overwrite_mode="replace" ile upsert davranışı
                    self.segments.insert(seg, overwrite_mode="replace")
                except Exception as e:
                    print(f"⚠️  Segment insert hatası ({seg['_key']}): {e}")
            
            print(f"✅ {len(segments)} segment yüklendi/güncellendi")
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
        
        # Ensure collections exist in this loader instance
        self._ensure_collections()

        # Timestamp belirle
        if timestamp:
            dt = dtparser.isoparse(timestamp)
        else:
            # Dosya adından çıkar: flow_20251027_1244.geojson
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
        
        # Measure dökümanları hazırla
        measures = []
        edges = []  # AT_TIME ilişkileri
        
        for feat in features:
            props = feat.get("properties", {})
            
            # Segment ID bul
            raw_seg_id = props.get("segmentId") or props.get("segment_id") or props.get("edge_id")
            if not raw_seg_id:
                coords = feat["geometry"]["coordinates"]
                raw_seg_id = hashlib.sha1(json.dumps(coords).encode()).hexdigest()[:20]
            
            # NORMALIZE: edge:xxx:001 -> edge:xxx (measure'lar segment ile eşleşmeli)
            seg_id = normalize_segment_id(raw_seg_id)
            
            # Measure key (segmentId + timestamp) - tüm özel karakterleri normalize et
            measure_key = f"{seg_id}_{timestamp_str}".replace("+", "_").replace("-", "_").replace(":", "_").replace(".", "_")
            
            # Değerleri al - 0 değerini korumak için if-else kullan
            speed_val = props.get("speed")
            if speed_val is None:
                speed_val = props.get("currentSpeed") or props.get("speed_kmh")
            
            freeflow_val = props.get("freeFlow")
            if freeflow_val is None:
                freeflow_val = props.get("freeFlowSpeed") or props.get("free_flow_kmh")
            
            jamfactor_val = props.get("jamFactor")
            if jamfactor_val is None:
                jamfactor_val = props.get("jam_factor")
            
            measure = {
                "_key": measure_key,
                "segmentId": seg_id,
                "timestamp": timestamp_str,
                "speed": speed_val,
                "freeFlow": freeflow_val,
                "jamFactor": jamfactor_val,
                "confidence": props.get("confidence"),
                "traversability": props.get("traversability"),
            }
            
            measures.append(measure)
            
            # AT_TIME edge (Segment → Measure) - key'leri normalize et
            segment_key = seg_id.replace("+", "_").replace("-", "_").replace(":", "_")
            edge = {
                "_from": f"Segment/{segment_key}",
                "_to": f"Measure/{measure_key}",
                "timestamp": timestamp_str
            }
            edges.append(edge)
        
        # Batch insert
        if measures:
            for measure in measures:
                try:
                    self.measures.insert(measure, overwrite=True)
                except Exception:
                    self.measures.update({"_key": measure["_key"]}, measure)
            
            print(f"✅ {len(measures)} measure yüklendi/güncellendi")
            
            # AT_TIME ilişkileri ekle
            for edge in edges:
                try:
                    self.at_time.insert(edge)
                except Exception:
                    pass  # Zaten varsa geç
            
            print(f"✅ {len(edges)} AT_TIME ilişkisi eklendi")
        else:
            print("⚠️  Hiç measure bulunamadı!")
        
        print()
    
    def build_connects_to(self, threshold=12.0):
        """
        CONNECTS_TO ilişkilerini oluştur (spatial topology)
        
        Python ile spatial grid optimizasyonu (AQL çok yavaş olabilir)
        
        Args:
            threshold: Maksimum mesafe (metre)
        """
        print("=" * 70)
        print(f"🔗 CONNECTS_TO ilişkileri oluşturuluyor (threshold={threshold}m)...")
        print("=" * 70)
        
        # Ensure collections exist
        self._ensure_collections()

        print("ℹ️  Spatial grid optimizasyonu kullanılıyor (4 yönlü kontrol)...")
        
        # Tüm segment'leri al (start ve end koordinatları ile)
        query = "FOR s IN Segment RETURN {_key: s._key, segmentId: s.segmentId, startLat: s.startLat, startLon: s.startLon, endLat: s.endLat, endLon: s.endLon}"
        cursor = self.db.aql.execute(query)
        segments = list(cursor)
        
        if not segments:
            print("⚠️  Segment bulunamadı")
            return
        
        print(f"ℹ️  {len(segments)} segment alındı")
        
        # Grid hücre boyutu
        grid_size = (threshold * 2) / 111320.0
        
        # Segment'leri grid'e yerleştir (her segment'in hem start hem end noktasını grid'e ekle)
        grid = {}
        for seg in segments:
            start_lat = seg.get("startLat")
            start_lon = seg.get("startLon")
            end_lat = seg.get("endLat")
            end_lon = seg.get("endLon")
            
            if not (start_lat and start_lon and end_lat and end_lon):
                continue
            
            # Start noktasını grid'e ekle
            grid_x_start = int(start_lat / grid_size)
            grid_y_start = int(start_lon / grid_size)
            cell_start = (grid_x_start, grid_y_start)
            if cell_start not in grid:
                grid[cell_start] = []
            if seg not in grid[cell_start]:
                grid[cell_start].append(seg)
            
            # End noktasını grid'e ekle (farklı hücredeyse)
            grid_x_end = int(end_lat / grid_size)
            grid_y_end = int(end_lon / grid_size)
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
                key1 = seg1["_key"]
                s1_start_lat = seg1["startLat"]
                s1_start_lon = seg1["startLon"]
                s1_end_lat = seg1["endLat"]
                s1_end_lon = seg1["endLon"]
                
                for ncell in neighbor_cells:
                    if ncell not in grid:
                        continue
                    for seg2 in grid[ncell]:
                        key2 = seg2["_key"]
                        if key1 >= key2:
                            continue
                        
                        pair = (min(key1, key2), max(key1, key2))
                        if pair in processed:
                            continue
                        
                        s2_start_lat = seg2["startLat"]
                        s2_start_lon = seg2["startLon"]
                        s2_end_lat = seg2["endLat"]
                        s2_end_lon = seg2["endLon"]
                        
                        # 4 yönlü kontrol: end→start, start→end, end→end, start→start
                        distances = [
                            self._haversine_distance(s1_end_lat, s1_end_lon, s2_start_lat, s2_start_lon),  # end→start
                            self._haversine_distance(s1_start_lat, s1_start_lon, s2_end_lat, s2_end_lon),  # start→end
                            self._haversine_distance(s1_end_lat, s1_end_lon, s2_end_lat, s2_end_lon),      # end→end
                            self._haversine_distance(s1_start_lat, s1_start_lon, s2_start_lat, s2_start_lon)  # start→start
                        ]
                        
                        # En kısa mesafeyi al
                        min_distance = min(distances)
                        
                        processed.add(pair)
                        
                        if min_distance <= threshold:
                            edges_to_insert.append({
                                "_from": f"Segment/{key1}",
                                "_to": f"Segment/{key2}",
                                "distance": min_distance
                            })
        
        # Batch insert
        if edges_to_insert:
            print(f"ℹ️  {len(edges_to_insert)} edge ekleniyor...")
            batch_size = 1000
            for i in range(0, len(edges_to_insert), batch_size):
                batch = edges_to_insert[i:i+batch_size]
                for edge in batch:
                    try:
                        self.connects_to.insert(edge)
                    except:
                        pass  # Zaten varsa geç
                if (i // batch_size) % 10 == 0:
                    print(f"  Progress: {i}/{len(edges_to_insert)}")
        
        print(f"✅ {len(edges_to_insert)} CONNECTS_TO edge eklendi!")
        
        # Toplam ilişki sayısı
        try:
            count_query = "RETURN LENGTH(CONNECTS_TO)"
            cursor = self.db.aql.execute(count_query)
            count = list(cursor)[0] if cursor else 0
            print(f"📊 Toplam CONNECTS_TO: {count:,}")
        except Exception as e:
            print(f"ℹ️  Count query hatası (normal olabilir): {e}")
        print()
    
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
        """Bağlantıyı kapat"""
        self.client.close()


# Test fonksiyonu
def main():
    """Basit test"""
    print("\n" + "="*70)
    print("🧪 ArangoDB Loader Test")
    print("="*70 + "\n")
    
    loader = ArangoLoader()
    
    # Schema oluştur
    loader.init_schema()
    
    # Test verisi yükle (mevcut dosyalar varsa)
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
