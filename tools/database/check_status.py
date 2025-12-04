#!/usr/bin/env python3
"""
check_dbs_status.py - Database Durumlarını Kontrol Et
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(ENV_PATH)

print("\n" + "=" * 80)
print("📊 DATABASE DURUM KONTROLÜ")
print("=" * 80 + "\n")

# ============================================================================
# NEO4J
# ============================================================================
print("🔵 NEO4J")
print("-" * 80)

try:
    from neo4j import GraphDatabase
    
    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASS", "123456789")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        # Node sayısı
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        node_count = result.single()["cnt"]
        
        # İlişki sayısı
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        rel_count = result.single()["cnt"]
        
        # Node tipleri
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt")
        node_types = {record["label"]: record["cnt"] for record in result}
        
        print(f"✅ Bağlantı: BAŞARILI")
        print(f"📦 Toplam Node: {node_count:,}")
        print(f"🔗 Toplam İlişki: {rel_count:,}")
        
        if node_types:
            print(f"📋 Node Tipleri:")
            for label, cnt in node_types.items():
                print(f"   - {label}: {cnt:,}")
        else:
            print(f"   ⚠️  Hiç node yok")
    
    driver.close()
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()

# ============================================================================
# ARANGODB
# ============================================================================
print("🟢 ARANGODB")
print("-" * 80)

try:
    from arango import ArangoClient
    
    host = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
    user = os.getenv("ARANGO_USER", "root")
    password = os.getenv("ARANGO_PASS", "1234")
    database = os.getenv("ARANGO_DATABASE", "traffic_db")
    
    client = ArangoClient(hosts=host)
    db = client.db(database, username=user, password=password)
    
    collections = ['Segment', 'Measure', 'CONNECTS_TO', 'AT_TIME']
    
    print(f"✅ Bağlantı: BAŞARILI")
    print(f"📋 Collection'lar:")
    
    total_docs = 0
    for coll_name in collections:
        if db.has_collection(coll_name):
            coll = db.collection(coll_name)
            count = coll.count()
            total_docs += count
            print(f"   - {coll_name}: {count:,}")
        else:
            print(f"   - {coll_name}: ⚠️  YOK")
    
    print(f"📦 Toplam Döküman: {total_docs:,}")
    
    # Graph kontrolü
    if db.has_graph('traffic_flow_graph'):
        print(f"🔗 Graph: traffic_flow_graph ✅")
    else:
        print(f"🔗 Graph: traffic_flow_graph ⚠️  YOK")
    
    client.close()
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()

# ============================================================================
# TIGERGRAPH
# ============================================================================
print("🟠 TIGERGRAPH")
print("-" * 80)

try:
    import requests
    
    host = os.getenv("TIGER_HOST", "http://127.0.0.1")
    rest_port = os.getenv("TIGER_REST_PORT", "9000")
    graphname = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")
    
    base_url = f"{host}:{rest_port}/graph/{graphname}/vertices"
    
    # Segment count - REST API ile
    try:
        url_seg = f"{base_url}/Segment?limit=1"
        response_seg = requests.get(url_seg, timeout=5)
        if response_seg.status_code == 200:
            data_seg = response_seg.json()
            # İlk 1000 segment al ve say
            url_seg_all = f"{base_url}/Segment?limit=10000"
            response_all = requests.get(url_seg_all, timeout=10)
            seg_count = len(response_all.json().get("results", []))
        else:
            seg_count = 0
    except:
        seg_count = 0
    
    # Measure count
    try:
        url_meas_all = f"{base_url}/Measure?limit=10000"
        response_meas = requests.get(url_meas_all, timeout=10)
        meas_count = len(response_meas.json().get("results", []))
    except:
        meas_count = 0
    
    print(f"✅ Bağlantı: BAŞARILI")
    print(f"📋 Vertex Tipleri:")
    print(f"   - Segment: {seg_count:,}")
    print(f"   - Measure: {meas_count:,}")
    print(f"📦 Toplam Vertex: {seg_count + meas_count:,}")
    print(f"🔗 Edge istatistikleri alınamadı")
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()
print("=" * 80)
print("✅ KONTROL TAMAMLANDI")
print("=" * 80 + "\n")
