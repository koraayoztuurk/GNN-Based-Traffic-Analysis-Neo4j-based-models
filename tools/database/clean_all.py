#!/usr/bin/env python3
"""
clean_everything.py - TÜM VERİLERİ TEMİZLE
----------------------------------------------
UYARI: Bu script şunları siler:
  ✓ Tüm veritabanlarındaki tüm veriler (Neo4j, ArangoDB, TigerGraph)
  ✓ Archive klasöründeki tüm GeoJSON dosyaları
  ✓ Data klasöründeki timeseries dosyaları
  ✓ here_flow_raw.json dosyası
  ✓ Oluşturulan harita dosyaları

KORUNANLAR:
  ✓ edges_static.geojson (statik segment verileri)
  ✓ Config dosyaları (.env, requirements.txt)
  ✓ Kaynak kodlar (src/, scripts/)
  ✓ Dokümantasyon (docs/, *.md)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(ENV_PATH)

# Hangi DB'ler aktif?
ACTIVE_DBS = os.getenv("ACTIVE_DATABASES", "neo4j").split(",")
ACTIVE_DBS = [db.strip().lower() for db in ACTIVE_DBS]

print("\n" + "=" * 80)
print("🧹 TÜM VERİLERİ TEMİZLE")
print("=" * 80)
print()
print("⚠️  UYARI: Bu işlem GERİ ALINAMAZ!")
print()
print("📋 Silinecekler:")
print("   ❌ Neo4j veritabanı (tüm node ve ilişkiler)")
print("   ❌ ArangoDB veritabanı (tüm collections)")
print("   ❌ TigerGraph veritabanı (tüm vertex ve edge'ler)")
print("   ❌ Archive klasöründeki tüm flow_*.geojson dosyaları")
print("   ❌ data/timeseries.parquet")
print("   ❌ data/timeseries.csv")
print("   ❌ here_flow_raw.json")
print("   ❌ src/visualization/map.html")
print()
print("✅ Korunacaklar:")
print("   ✓ data/edges_static.geojson (statik veriler)")
print("   ✓ Config dosyaları")
print("   ✓ Kaynak kodlar")
print()

# Onay al
response = input("⚠️  Devam etmek istediğinize EMİN MİSİNİZ? (EVET yazın): ")

if response != "EVET":
    print("\n❌ İşlem iptal edildi")
    sys.exit(0)

print()

# ============================================================================
# 1. NEO4J TEMİZLEME
# ============================================================================
if "neo4j" in ACTIVE_DBS:
    print("=" * 80)
    print("🔵 NEO4J TEMİZLENİYOR")
    print("=" * 80)
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASS", "123456789")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session(database=database) as session:
            # Tüm ilişkileri sil
            print("🔗 İlişkiler siliniyor...")
            result = session.run("MATCH ()-[r]->() DELETE r RETURN count(r) AS cnt")
            rel_count = result.single()["cnt"]
            print(f"   ✅ {rel_count:,} ilişki silindi")
            
            # Tüm node'ları sil
            print("📦 Node'lar siliniyor...")
            result = session.run("MATCH (n) DELETE n RETURN count(n) AS cnt")
            node_count = result.single()["cnt"]
            print(f"   ✅ {node_count:,} node silindi")
            
            # TS15 node'larını özellikle temizle (deprecated özellik)
            print("🧹 TS15 node'ları temizleniyor (deprecated)...")
            try:
                result = session.run("MATCH (t:TS15) DETACH DELETE t RETURN count(t) AS cnt")
                ts15_count = result.single()["cnt"]
                if ts15_count > 0:
                    print(f"   ✅ {ts15_count} TS15 node silindi")
            except:
                pass  # Zaten yoksa devam et
            
            # İndeksleri bırak (opsiyonel - schema korunur)
            # print("📋 İndeksler temizleniyor...")
            # result = session.run("SHOW INDEXES")
            # ...
        
        driver.close()
        print("✅ Neo4j temizlendi!\n")
        
    except Exception as e:
        print(f"⚠️  Neo4j hatası: {e}")
        print("   (Veritabanı çalışmıyor olabilir)\n")

# ============================================================================
# 2. ARANGODB TEMİZLEME
# ============================================================================
if "arangodb" in ACTIVE_DBS:
    print("=" * 80)
    print("🟢 ARANGODB TEMİZLENİYOR")
    print("=" * 80)
    
    try:
        from arango import ArangoClient
        
        host = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
        user = os.getenv("ARANGO_USER", "root")
        password = os.getenv("ARANGO_PASS", "1234")
        database = os.getenv("ARANGO_DATABASE", "traffic_db")
        
        client = ArangoClient(hosts=host)
        db = client.db(database, username=user, password=password)
        
        # Collection'ları truncate et (schema korunur)
        collections = ['Segment', 'Measure', 'CONNECTS_TO', 'AT_TIME']
        
        for coll_name in collections:
            if db.has_collection(coll_name):
                coll = db.collection(coll_name)
                coll.truncate()
                print(f"   ✅ {coll_name} temizlendi")
        
        client.close()
        print("✅ ArangoDB temizlendi!\n")
        
    except Exception as e:
        print(f"⚠️  ArangoDB hatası: {e}")
        print("   (Veritabanı çalışmıyor olabilir)\n")

# ============================================================================
# 3. TIGERGRAPH TEMİZLEME
# ============================================================================
if "tigergraph" in ACTIVE_DBS:
    print("=" * 80)
    print("🟠 TIGERGRAPH TEMİZLENİYOR")
    print("=" * 80)
    
    try:
        import pyTigerGraph as tg
        
        host = os.getenv("TIGER_HOST", "http://127.0.0.1")
        rest_port = os.getenv("TIGER_REST_PORT", "9000")
        username = os.getenv("TIGER_USERNAME", "tigergraph")
        password = os.getenv("TIGER_PASSWORD", "tigergraph")
        graphname = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")
        
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graphname
        )
        
        # Token al
        try:
            conn.apiToken = conn.getToken(conn.createSecret())[0]
        except:
            pass
        
        # Tüm vertex'leri sil (en hızlı yöntem: graph drop + recreate yerine REST API delete)
        print("📦 Tüm veriler siliniyor...")
        
        deleted_total = 0
        
        try:
            # Tüm vertex'leri sil (permanent=True ile kalıcı silme)
            # where parametresini kullanmadan tüm vertex'leri silmek için limit çok yüksek ayarla
            result_seg = conn.delVertices("Segment", limit="999999", permanent=True)
            result_meas = conn.delVertices("Measure", limit="999999", permanent=True)
            
            deleted_total = result_seg + result_meas
            print(f"   ✅ {result_seg} Segment + {result_meas} Measure vertex silindi")
            
        except Exception as e:
            print(f"   ⚠️  Toplu silme hatası: {e}")
            print(f"   ℹ️  Alternatif yöntem deneniyor...")
            
            # Alternatif: Tüm vertex ID'leri çekip tek tek sil
            try:
                # Segment'leri sil
                segments = conn.getVertices("Segment", limit=999999)
                for seg in segments:
                    try:
                        conn.delVertices("Segment", where=f"primary_id==\"{seg['v_id']}\"", permanent=True)
                        deleted_total += 1
                    except:
                        pass
                
                # Measure'leri sil
                measures = conn.getVertices("Measure", limit=999999)
                for meas in measures:
                    try:
                        conn.delVertices("Measure", where=f"primary_id==\"{meas['v_id']}\"", permanent=True)
                        deleted_total += 1
                    except:
                        pass
                
                print(f"   ✅ {deleted_total} vertex silindi (tek tek)")
            except Exception as e2:
                print(f"   ⚠️  Tek tek silme hatası: {e2}")
        
        # Sonuç kontrolü
        try:
            seg_count = conn.getVertexCount("Segment")
            meas_count = conn.getVertexCount("Measure")
            
            if seg_count == 0 and meas_count == 0:
                print(f"   ✅ Graph başarıyla temizlendi!")
            else:
                print(f"   ⚠️  Kalan veriler: {seg_count} Segment, {meas_count} Measure")
                print(f"   ℹ️  Tam temizlik için: python tools/database/reset_tigergraph.py")
        except:
            pass
        
        print("✅ TigerGraph temizleme tamamlandı!\n")
        
    except Exception as e:
        print(f"⚠️  TigerGraph hatası: {e}")
        print("   (Veritabanı çalışmıyor olabilir)\n")

# ============================================================================
# 4. DOSYA TEMİZLEME
# ============================================================================
print("=" * 80)
print("📁 DOSYALAR TEMİZLENİYOR")
print("=" * 80)

deleted_files = 0

# 4.1 Archive klasörü
archive_dir = ROOT_DIR / "archive"
if archive_dir.exists():
    flow_files = list(archive_dir.glob("flow_*.geojson"))
    for f in flow_files:
        try:
            f.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"⚠️  {f.name} silinemedi: {e}")
    print(f"✅ Archive temizlendi ({len(flow_files)} dosya)")

# 4.2 Data klasörü
data_dir = ROOT_DIR / "data"
files_to_remove = [
    "timeseries.csv",
    "timeseries.parquet",
    "features_window.csv",
    "pyg_graph.npz"
]

for fname in files_to_remove:
    fpath = data_dir / fname
    if fpath.exists():
        try:
            fpath.unlink()
            deleted_files += 1
            print(f"✅ {fname} silindi")
        except Exception as e:
            print(f"⚠️  {fname} silinemedi: {e}")

# 4.3 HERE API raw output
raw_json = ROOT_DIR / "data" / "raw" / "here_flow_raw.json"
if raw_json.exists():
    try:
        raw_json.unlink()
        deleted_files += 1
        print(f"✅ here_flow_raw.json silindi")
    except Exception as e:
        print(f"⚠️  here_flow_raw.json silinemedi: {e}")

# 4.4 Visualization outputs
viz_files = [
    ROOT_DIR / "src/visualization/map.html",
    ROOT_DIR / "neo4j_traffic_map.html",
]

for vf in viz_files:
    if vf.exists():
        try:
            vf.unlink()
            deleted_files += 1
            print(f"✅ {vf.name} silindi")
        except Exception as e:
            print(f"⚠️  {vf.name} silinemedi: {e}")

print()
print(f"✅ Toplam {deleted_files} dosya silindi")
print()

# ============================================================================
# ÖZET
# ============================================================================
print("=" * 80)
print("🎉 TEMİZLEME TAMAMLANDI!")
print("=" * 80)
print()
print("✅ Tüm veritabanları temizlendi")
print("✅ Tüm flow arşivi silindi")
print("✅ Tüm geçici dosyalar silindi")
print()
print("📝 Korunan dosyalar:")
print("   ✓ data/edges_static.geojson")
print("   ✓ config/.env")
print("   ✓ config/requirements.txt")
print("   ✓ Tüm kaynak kodlar (src/)")
print()
print("🚀 Yeni başlangıç için:")
print("   python run_pipeline.py")
print()
print("=" * 80)
