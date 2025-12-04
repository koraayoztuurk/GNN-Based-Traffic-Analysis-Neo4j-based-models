#!/usr/bin/env python3
"""
detailed_comparison.py - Detaylı Database Karşılaştırması
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(ENV_PATH)

print("\n" + "=" * 80)
print("🔍 DETAYLI DATABASE KARŞILAŞTIRMASI")
print("=" * 80 + "\n")

# Örnek segmentler seçelim
sample_segment_ids = []

# ============================================================================
# 1. Sample Segment ID'leri topla
# ============================================================================
print("📋 Sample segment ID'leri toplanıyor...\n")

try:
    from neo4j import GraphDatabase
    
    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASS", "123456789")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        result = session.run("MATCH (s:Segment) RETURN s.segmentId AS id LIMIT 3")
        sample_segment_ids = [record["id"] for record in result]
    
    driver.close()
    print(f"✅ Sample ID'ler: {sample_segment_ids}\n")
    
except Exception as e:
    print(f"❌ Neo4j sample alma hatası: {e}\n")

# ============================================================================
# 2. NEO4J - Detaylı Veri
# ============================================================================
print("🔵 NEO4J - Detaylı Veri")
print("-" * 80)

try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        for seg_id in sample_segment_ids[:1]:  # İlk segment
            # Segment bilgileri
            query = """
            MATCH (s:Segment {segmentId: $seg_id})
            OPTIONAL MATCH (s)-[:AT_TIME]->(m:Measure)
            OPTIONAL MATCH (s)-[:CONNECTS_TO]-(neighbor:Segment)
            RETURN s, 
                   count(DISTINCT m) AS measure_count,
                   count(DISTINCT neighbor) AS neighbor_count
            """
            result = session.run(query, seg_id=seg_id)
            record = result.single()
            
            if record:
                seg = record["s"]
                print(f"\n📍 Segment: {seg_id}")
                print(f"   - Name: {seg.get('name', 'N/A')}")
                print(f"   - FRC: {seg.get('frc', 'N/A')}")
                print(f"   - Length: {seg.get('lengthM', 'N/A'):.1f}m")
                print(f"   - Coordinates: ({seg.get('lat', 0):.5f}, {seg.get('lon', 0):.5f})")
                print(f"   - Measure Count: {record['measure_count']}")
                print(f"   - Neighbor Count: {record['neighbor_count']}")
                
                # İlk measure
                result2 = session.run("""
                    MATCH (s:Segment {segmentId: $seg_id})-[:AT_TIME]->(m:Measure)
                    RETURN m
                    LIMIT 1
                """, seg_id=seg_id)
                
                meas_record = result2.single()
                if meas_record:
                    m = meas_record["m"]
                    print(f"\n   📊 Sample Measure:")
                    print(f"      - Timestamp: {m.get('timestamp', 'N/A')}")
                    print(f"      - Speed: {m.get('speed', 'N/A'):.1f} km/h")
                    print(f"      - JamFactor: {m.get('jamFactor', 'N/A'):.2f}")
                    print(f"      - FreeFlow: {m.get('freeFlow', 'N/A'):.1f} km/h")
    
    driver.close()
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()

# ============================================================================
# 3. ARANGODB - Detaylı Veri
# ============================================================================
print("🟢 ARANGODB - Detaylı Veri")
print("-" * 80)

try:
    from arango import ArangoClient
    
    host = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
    user = os.getenv("ARANGO_USER", "root")
    password = os.getenv("ARANGO_PASS", "1234")
    database_name = os.getenv("ARANGO_DATABASE", "traffic_db")
    
    client = ArangoClient(hosts=host)
    db = client.db(database_name, username=user, password=password)
    
    for seg_id in sample_segment_ids[:1]:  # İlk segment
        # Segment bilgileri
        seg_coll = db.collection('Segment')
        segments = list(seg_coll.find({'segmentId': seg_id}))
        
        if segments:
            seg = segments[0]
            print(f"\n📍 Segment: {seg_id}")
            print(f"   - Name: {seg.get('name', 'N/A')}")
            print(f"   - FRC: {seg.get('frc', 'N/A')}")
            print(f"   - Length: {seg.get('lengthM', 'N/A'):.1f}m")
            print(f"   - Coordinates: ({seg.get('lat', 0):.5f}, {seg.get('lon', 0):.5f})")
            
            # AT_TIME sayısı
            at_time_count = db.collection('AT_TIME').find({'_from': seg['_id']}).count()
            print(f"   - Measure Count: {at_time_count}")
            
            # CONNECTS_TO sayısı (graph query)
            query = """
            FOR v, e, p IN 1..1 ANY @start_vertex GRAPH 'traffic_flow_graph'
                RETURN DISTINCT v._id
            """
            cursor = db.aql.execute(query, bind_vars={'start_vertex': seg['_id']})
            neighbor_count = len(list(cursor)) - 1  # Kendisi hariç
            print(f"   - Neighbor Count: {neighbor_count}")
            
            # İlk measure
            at_times = list(db.collection('AT_TIME').find({'_from': seg['_id']}, limit=1))
            if at_times:
                measure_id = at_times[0]['_to']
                measure = db.collection('Measure').get(measure_id.split('/')[-1])
                
                if measure:
                    print(f"\n   📊 Sample Measure:")
                    print(f"      - Timestamp: {measure.get('timestamp', 'N/A')}")
                    print(f"      - Speed: {measure.get('speed', 'N/A'):.1f} km/h")
                    print(f"      - JamFactor: {measure.get('jamFactor', 'N/A'):.2f}")
                    print(f"      - FreeFlow: {measure.get('freeFlow', 'N/A'):.1f} km/h")
    
    client.close()
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()

# ============================================================================
# 4. TIGERGRAPH - Detaylı Veri
# ============================================================================
print("🟠 TIGERGRAPH - Detaylı Veri")
print("-" * 80)

try:
    import pyTigerGraph as tg
    
    host = os.getenv("TIGER_HOST", "http://127.0.0.1")
    graphname = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")
    
    conn = tg.TigerGraphConnection(host=host, graphname=graphname)
    
    # Token al (opsiyonel)
    try:
        conn.apiToken = conn.getToken(conn.createSecret())[0]
    except:
        pass
    
    for seg_id in sample_segment_ids[:1]:  # İlk segment
        # Segment bilgileri
        vertices = conn.getVertices("Segment", where=f'segmentId=="{seg_id}"')
        
        if vertices:
            seg = vertices[0]
            attrs = seg.get('attributes', {})
            
            print(f"\n📍 Segment: {seg_id}")
            print(f"   - Name: {attrs.get('name', 'N/A')}")
            print(f"   - FRC: {attrs.get('frc', 'N/A')}")
            print(f"   - Length: {attrs.get('lengthM', 'N/A'):.1f}m")
            print(f"   - Coordinates: ({attrs.get('lat', 0):.5f}, {attrs.get('lon', 0):.5f})")
            
            # Edge sayıları (detaylı query gerekebilir)
            try:
                # AT_TIME edges
                edges = conn.getEdges("Segment", seg['v_id'], edgeType="AT_TIME")
                print(f"   - Measure Count: {len(edges)}")
                
                # CONNECTS_TO edges
                neighbors = conn.getEdges("Segment", seg['v_id'], edgeType="CONNECTS_TO")
                print(f"   - Neighbor Count: {len(neighbors)}")
                
                # İlk measure
                if edges:
                    first_measure_id = edges[0]
                    measures = conn.getVertices("Measure", where=f'primary_id=="{first_measure_id}"')
                    
                    if measures:
                        m = measures[0].get('attributes', {})
                        print(f"\n   📊 Sample Measure:")
                        print(f"      - Timestamp: {m.get('timestamp', 'N/A')}")
                        print(f"      - Speed: {m.get('speed', 'N/A'):.1f} km/h")
                        print(f"      - JamFactor: {m.get('jamFactor', 'N/A'):.2f}")
                        print(f"      - FreeFlow: {m.get('freeFlow', 'N/A'):.1f} km/h")
            except Exception as e:
                print(f"   ⚠️  Edge detayları alınamadı: {e}")
    
except Exception as e:
    print(f"❌ HATA: {e}")

print()
print("=" * 80)
print("✅ DETAYLI KARŞILAŞTIRMA TAMAMLANDI")
print("=" * 80 + "\n")
