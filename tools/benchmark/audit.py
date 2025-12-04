#!/usr/bin/env python3
"""
KAPSAMLI DATABASE AUDIT - TÜM VERİTABANLARINI KONTROL ET
- Neo4j
- ArangoDB  
- TigerGraph

Tüm ilişkiler, node/vertex sayıları, veri bütünlüğü kontrolü
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv('config/.env')

# Hangi DB'ler aktif?
ACTIVE_DBS = os.getenv("ACTIVE_DATABASES", "neo4j,arangodb,tigergraph").split(",")
ACTIVE_DBS = [db.strip().lower() for db in ACTIVE_DBS]

print("\n" + "=" * 100)
print("🔍 KAPSAMLI MULTI-DATABASE AUDIT")
print("=" * 100)
print(f"ℹ️  Aktif veritabanları: {', '.join(ACTIVE_DBS)}")
print()

# Global sonuçlar
db_results = {}


# ============================================================================
# NEO4J AUDIT
# ============================================================================
def audit_neo4j():
    """Neo4j veritabanını denetle"""
    print("=" * 100)
    print("🔵 NEO4J AUDIT")
    print("=" * 100)
    
    result = {
        'segments': 0,
        'measures': 0,
        'connects_to': 0,
        'at_time': 0,
        'errors': [],
        'warnings': [],
        'infos': [],
        'score': 0
    }
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASS', '123456789')
        database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        session = driver.session(database=database)
        
        # 1. Node sayıları
        print("\n📦 1. NODE SAYILARI")
        print("-" * 100)
        
        query_result = session.run("MATCH (n:Segment) RETURN count(n) AS cnt")
        result['segments'] = query_result.single()['cnt']
        
        query_result = session.run("MATCH (n:Measure) RETURN count(n) AS cnt")
        result['measures'] = query_result.single()['cnt']
        
        print(f"   Segment: {result['segments']:,}")
        print(f"   Measure: {result['measures']:,}")
        
        # 2. İlişki sayıları
        print("\n🔗 2. İLİŞKİ SAYILARI")
        print("-" * 100)
        
        query_result = session.run("MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS cnt")
        result['connects_to'] = query_result.single()['cnt']
        
        query_result = session.run("MATCH ()-[r:AT_TIME]->() RETURN count(r) AS cnt")
        result['at_time'] = query_result.single()['cnt']
        
        print(f"   CONNECTS_TO: {result['connects_to']:,}")
        print(f"   AT_TIME: {result['at_time']:,}")
        
        # 3. Veri bütünlüğü kontrolleri
        print("\n✅ 3. VERİ BÜTÜNLÜĞÜ")
        print("-" * 100)
        
        # AT_TIME = Measure?
        if result['at_time'] == result['measures'] and result['measures'] > 0:
            result['infos'].append("✅ AT_TIME = Measure (1:1 eşleşme)")
            print("   ✅ AT_TIME = Measure (1:1 eşleşme)")
        elif result['measures'] > 0:
            result['errors'].append(f"❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
            print(f"   ❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
        
        # Koordinat kontrolü
        query_result = session.run("""
            MATCH (s:Segment)
            WHERE s.startLon IS NULL OR s.startLat IS NULL
               OR s.endLon IS NULL OR s.endLat IS NULL
            RETURN count(s) AS cnt
        """)
        no_coords = query_result.single()['cnt']
        
        if no_coords == 0:
            result['infos'].append("✅ Tüm segment'lerde koordinat var")
            print("   ✅ Tüm segment'lerde koordinat var")
        else:
            result['errors'].append(f"❌ {no_coords} segment'de koordinat eksik")
            print(f"   ❌ {no_coords} segment'de koordinat eksik")
        
        # İzole segment
        query_result = session.run("""
            MATCH (s:Segment)
            WHERE NOT exists((s)-[:CONNECTS_TO]->())
              AND NOT exists((s)<-[:CONNECTS_TO]-())
            RETURN count(s) AS cnt
        """)
        isolated = query_result.single()['cnt']
        isolated_pct = (isolated / result['segments'] * 100) if result['segments'] > 0 else 0
        
        print(f"   İzole segment: {isolated:,} ({isolated_pct:.1f}%)")
        
        if isolated_pct > 5:
            result['warnings'].append(f"⚠️  İzole segment oranı yüksek: {isolated_pct:.1f}%")
        elif isolated_pct < 1:
            result['infos'].append(f"✅ İzole segment çok az: {isolated_pct:.1f}%")
        
        # Skor hesapla
        checks = [
            result['segments'] > 0,
            result['measures'] > 0,
            result['connects_to'] > 0,
            result['at_time'] > 0,
            result['at_time'] == result['measures'],
            no_coords == 0,
            isolated_pct < 5
        ]
        result['score'] = sum(checks) / len(checks) * 100
        
        session.close()
        driver.close()
        
        print(f"\n   📊 Sağlık Skoru: {result['score']:.0f}%")
        
    except Exception as e:
        result['errors'].append(f"❌ Bağlantı hatası: {e}")
        print(f"\n❌ Hata: {e}")
    
    return result


# ============================================================================
# ARANGODB AUDIT
# ============================================================================
def audit_arangodb():
    """ArangoDB veritabanını denetle"""
    print("\n" + "=" * 100)
    print("🟢 ARANGODB AUDIT")
    print("=" * 100)
    
    result = {
        'segments': 0,
        'measures': 0,
        'connects_to': 0,
        'at_time': 0,
        'errors': [],
        'warnings': [],
        'infos': [],
        'score': 0
    }
    
    try:
        from arango import ArangoClient
        
        host = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
        user = os.getenv("ARANGO_USER", "root")
        password = os.getenv("ARANGO_PASS", "1234")
        database = os.getenv("ARANGO_DATABASE", "traffic_db")
        
        client = ArangoClient(hosts=host)
        db = client.db(database, username=user, password=password)
        
        # 1. Collection sayıları
        print("\n📦 1. COLLECTION SAYILARI")
        print("-" * 100)
        
        if db.has_collection('Segment'):
            result['segments'] = db.collection('Segment').count()
        if db.has_collection('Measure'):
            result['measures'] = db.collection('Measure').count()
        
        print(f"   Segment: {result['segments']:,}")
        print(f"   Measure: {result['measures']:,}")
        
        # 2. Edge collection sayıları
        print("\n🔗 2. EDGE COLLECTION SAYILARI")
        print("-" * 100)
        
        if db.has_collection('CONNECTS_TO'):
            result['connects_to'] = db.collection('CONNECTS_TO').count()
        if db.has_collection('AT_TIME'):
            result['at_time'] = db.collection('AT_TIME').count()
        
        print(f"   CONNECTS_TO: {result['connects_to']:,}")
        print(f"   AT_TIME: {result['at_time']:,}")
        
        # 3. Veri bütünlüğü kontrolleri
        print("\n✅ 3. VERİ BÜTÜNLÜĞÜ")
        print("-" * 100)
        
        # AT_TIME = Measure?
        if result['at_time'] == result['measures'] and result['measures'] > 0:
            result['infos'].append("✅ AT_TIME = Measure (1:1 eşleşme)")
            print("   ✅ AT_TIME = Measure (1:1 eşleşme)")
        elif result['measures'] > 0:
            result['errors'].append(f"❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
            print(f"   ❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
        
        # Koordinat kontrolü
        no_coords = 0
        if db.has_collection('Segment'):
            cursor = db.aql.execute("""
                FOR s IN Segment
                FILTER s.startLon == null OR s.startLat == null
                    OR s.endLon == null OR s.endLat == null
                COLLECT WITH COUNT INTO cnt
                RETURN cnt
            """)
            no_coords = next(cursor, 0)
            
            if no_coords == 0:
                result['infos'].append("✅ Tüm segment'lerde koordinat var")
                print("   ✅ Tüm segment'lerde koordinat var")
            else:
                result['errors'].append(f"❌ {no_coords} segment'de koordinat eksik")
                print(f"   ❌ {no_coords} segment'de koordinat eksik")
        
        # İzole segment
        isolated = 0
        isolated_pct = 0
        if db.has_collection('Segment') and db.has_collection('CONNECTS_TO'):
            cursor = db.aql.execute("""
                LET connected = (
                    FOR edge IN CONNECTS_TO
                    RETURN DISTINCT [edge._from, edge._to]
                )
                LET connected_ids = FLATTEN(connected)
                
                FOR s IN Segment
                FILTER s._id NOT IN connected_ids
                COLLECT WITH COUNT INTO cnt
                RETURN cnt
            """)
            isolated = next(cursor, 0)
            isolated_pct = (isolated / result['segments'] * 100) if result['segments'] > 0 else 0
            
            print(f"   İzole segment: {isolated:,} ({isolated_pct:.1f}%)")
            
            if isolated_pct > 5:
                result['warnings'].append(f"⚠️  İzole segment oranı yüksek: {isolated_pct:.1f}%")
            elif isolated_pct < 1:
                result['infos'].append(f"✅ İzole segment çok az: {isolated_pct:.1f}%")
        
        # Skor hesapla
        checks = [
            result['segments'] > 0,
            result['measures'] > 0,
            result['connects_to'] > 0,
            result['at_time'] > 0,
            result['at_time'] == result['measures'],
            no_coords == 0,
            isolated_pct < 5
        ]
        result['score'] = sum(checks) / len(checks) * 100
        
        client.close()
        
        print(f"\n   📊 Sağlık Skoru: {result['score']:.0f}%")
        
    except Exception as e:
        result['errors'].append(f"❌ Bağlantı hatası: {e}")
        print(f"\n❌ Hata: {e}")
    
    return result


# ============================================================================
# TIGERGRAPH AUDIT
# ============================================================================
def audit_tigergraph():
    """TigerGraph veritabanını denetle"""
    print("\n" + "=" * 100)
    print("🟠 TIGERGRAPH AUDIT")
    print("=" * 100)
    
    result = {
        'segments': 0,
        'measures': 0,
        'connects_to': 0,
        'at_time': 0,
        'errors': [],
        'warnings': [],
        'infos': [],
        'score': 0
    }
    
    try:
        import pyTigerGraph as tg
        
        host = os.getenv("TIGER_HOST", "http://127.0.0.1")
        rest_port = int(os.getenv("TIGER_REST_PORT", "9000"))
        gsql_port = int(os.getenv("TIGER_GSQL_PORT", "14240"))
        username = os.getenv("TIGER_USERNAME", "tigergraph")
        password = os.getenv("TIGER_PASSWORD", "tigergraph")
        graphname = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")
        
        conn = tg.TigerGraphConnection(
            host=host,
            restppPort=rest_port,
            gsPort=gsql_port,
            username=username,
            password=password,
            graphname=graphname
        )
        
        # 1. Vertex sayıları
        print("\n📦 1. VERTEX SAYILARI")
        print("-" * 100)
        
        try:
            segments = conn.getVertices("Segment", limit=999999)
            result['segments'] = len(segments)
        except:
            result['segments'] = 0
        
        try:
            measures = conn.getVertices("Measure", limit=999999)
            result['measures'] = len(measures)
        except:
            result['measures'] = 0
        
        print(f"   Segment: {result['segments']:,}")
        print(f"   Measure: {result['measures']:,}")
        
        # 2. Edge sayıları
        print("\n🔗 2. EDGE SAYILARI")
        print("-" * 100)
        
        try:
            # CONNECTS_TO sayısı
            result['connects_to'] = conn.getEdgeCount("CONNECTS_TO")
        except:
            result['connects_to'] = 0
        
        try:
            # AT_TIME sayısı
            result['at_time'] = conn.getEdgeCount("AT_TIME")
        except:
            result['at_time'] = 0
        
        print(f"   CONNECTS_TO: {result['connects_to']:,}")
        print(f"   AT_TIME: {result['at_time']:,}")
        
        # 3. Veri bütünlüğü kontrolleri
        print("\n✅ 3. VERİ BÜTÜNLÜĞÜ")
        print("-" * 100)
        
        # AT_TIME = Measure?
        if result['at_time'] == result['measures'] and result['measures'] > 0:
            result['infos'].append("✅ AT_TIME = Measure (1:1 eşleşme)")
            print("   ✅ AT_TIME = Measure (1:1 eşleşme)")
        elif result['measures'] > 0:
            result['errors'].append(f"❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
            print(f"   ❌ AT_TIME ({result['at_time']}) ≠ Measure ({result['measures']})")
        
        # Koordinat kontrolü
        no_coords = 0
        if result['segments'] > 0:
            try:
                segments_list = conn.getVertices("Segment", limit=999999)
                for seg in segments_list:
                    attrs = seg.get('attributes', {})
                    if not attrs.get('startLon') or not attrs.get('startLat') or \
                       not attrs.get('endLon') or not attrs.get('endLat'):
                        no_coords += 1
            except:
                pass
        
        if no_coords == 0:
            result['infos'].append("✅ Tüm segment'lerde koordinat var")
            print("   ✅ Tüm segment'lerde koordinat var")
        elif result['segments'] > 0:
            result['errors'].append(f"❌ {no_coords} segment'de koordinat eksik")
            print(f"   ❌ {no_coords} segment'de koordinat eksik")
        
        # İzole segment
        isolated = 0
        isolated_pct = 0
        if result['segments'] > 0 and result['connects_to'] > 0:
            try:
                # Tüm segment'leri al
                all_segments = set()
                segments_list = conn.getVertices("Segment", limit=999999)
                for seg in segments_list:
                    all_segments.add(seg.get('v_id'))
                
                # CONNECTS_TO'da olan segment'leri al
                connected_segments = set()
                edges = conn.getEdgesByType("CONNECTS_TO", fmt="json")
                for edge in edges:
                    connected_segments.add(edge.get('from_id'))
                    connected_segments.add(edge.get('to_id'))
                
                isolated = len(all_segments - connected_segments)
                isolated_pct = (isolated / result['segments'] * 100)
            except:
                pass
        
        print(f"   İzole segment: {isolated:,} ({isolated_pct:.1f}%)")
        
        if isolated_pct > 5:
            result['warnings'].append(f"⚠️  İzole segment oranı yüksek: {isolated_pct:.1f}%")
        elif isolated_pct < 1:
            result['infos'].append(f"✅ İzole segment çok az: {isolated_pct:.1f}%")
        
        # Skor hesapla
        checks = [
            result['segments'] > 0,
            result['measures'] > 0,
            result['connects_to'] > 0,
            result['at_time'] > 0,
            result['at_time'] == result['measures'],
            no_coords == 0,
            isolated_pct < 5
        ]
        result['score'] = sum(checks) / len(checks) * 100
        
        print(f"\n   📊 Sağlık Skoru: {result['score']:.0f}%")
        
    except Exception as e:
        result['errors'].append(f"❌ Bağlantı hatası: {e}")
        print(f"\n❌ Hata: {e}")
    
    return result


# ============================================================================
# ANA PROGRAM
# ============================================================================
def main():
    """Tüm aktif veritabanlarını denetle"""
    
    # Her DB için audit yap
    if 'neo4j' in ACTIVE_DBS:
        db_results['neo4j'] = audit_neo4j()
    
    if 'arangodb' in ACTIVE_DBS:
        db_results['arangodb'] = audit_arangodb()
    
    if 'tigergraph' in ACTIVE_DBS:
        db_results['tigergraph'] = audit_tigergraph()
    
    # Karşılaştırma tablosu
    print("\n" + "=" * 100)
    print("📊 VERİTABANLARI KARŞILAŞTIRMA")
    print("=" * 100)
    print()
    
    # Tablo başlığı
    print(f"{'Metric':<20}", end="")
    for db_name in db_results.keys():
        print(f"{db_name.upper():<20}", end="")
    print()
    print("-" * 100)
    
    # Satırlar
    metrics = [
        ('Segment', 'segments'),
        ('Measure', 'measures'),
        ('CONNECTS_TO', 'connects_to'),
        ('AT_TIME', 'at_time'),
        ('Skor', 'score')
    ]
    
    for label, key in metrics:
        print(f"{label:<20}", end="")
        for db_name, result in db_results.items():
            value = result.get(key, 0)
            if key == 'score':
                print(f"{value:.0f}%{'':<16}", end="")
            else:
                print(f"{value:,}{'':<20}"[:20], end="")
        print()
    
    print()
    
    # Tutarlılık kontrolü
    print("🔍 TUTARLILIK KONTROLÜ")
    print("-" * 100)
    
    if len(db_results) > 1:
        # Segment sayıları
        segment_counts = [r['segments'] for r in db_results.values() if r['segments'] > 0]
        if segment_counts and len(set(segment_counts)) == 1:
            print(f"   ✅ Segment sayıları tutarlı: {segment_counts[0]:,}")
        elif segment_counts:
            print(f"   ⚠️  Segment sayıları farklı: {segment_counts}")
        
        # Measure sayıları
        measure_counts = [r['measures'] for r in db_results.values() if r['measures'] > 0]
        if measure_counts and len(set(measure_counts)) == 1:
            print(f"   ✅ Measure sayıları tutarlı: {measure_counts[0]:,}")
        elif measure_counts:
            print(f"   ⚠️  Measure sayıları farklı: {measure_counts}")
    else:
        print("   ℹ️  Tek veritabanı aktif, karşılaştırma yok")
    
    print()
    
    # Genel sağlık skoru
    print("🎯 GENEL SAĞLIK SKORU")
    print("-" * 100)
    
    if db_results:
        avg_score = sum(r['score'] for r in db_results.values()) / len(db_results)
        print(f"   Ortalama: {avg_score:.0f}%")
        
        if avg_score >= 90:
            print(f"   🎉 MÜKEMMEL! Tüm veritabanları sağlıklı")
        elif avg_score >= 70:
            print(f"   ✅ İyi! Küçük iyileştirmeler yapılabilir")
        elif avg_score >= 50:
            print(f"   ⚠️  Orta! Bazı problemler var")
        else:
            print(f"   ❌ Zayıf! Ciddi problemler var")
    
    print()
    
    # Tüm hatalar ve uyarılar
    all_errors = []
    all_warnings = []
    
    for db_name, result in db_results.items():
        for err in result['errors']:
            all_errors.append(f"[{db_name.upper()}] {err}")
        for warn in result['warnings']:
            all_warnings.append(f"[{db_name.upper()}] {warn}")
    
    if all_errors:
        print("❌ TÜM HATALAR:")
        for err in all_errors:
            print(f"   {err}")
        print()
    
    if all_warnings:
        print("⚠️  TÜM UYARILAR:")
        for warn in all_warnings:
            print(f"   {warn}")
        print()
    
    if not all_errors and not all_warnings:
        print("✅ HİÇBİR SORUN YOK! Tüm veritabanları tamamen sağlıklı!")
        print()
    
    print("=" * 100)
    print("✨ Multi-Database Audit Tamamlandı!")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
