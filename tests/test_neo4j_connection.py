#!/usr/bin/env python3
"""
Neo4j bağlantısını ve veri durumunu test eder
"""
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# .env dosyasını config/ dizininden yükle
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def test_connection():
    print("=" * 70)
    print("  NEO4J BAĞLANTI VE VERİ TESTİ")
    print("=" * 70)
    print()
    print(f"🔗 Bağlantı Bilgileri:")
    print(f"   URI:  {NEO4J_URI}")
    print(f"   User: {NEO4J_USER}")
    print()
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        print("✅ Neo4j'ye bağlantı başarılı!")
        print()
        
        with driver.session(database=NEO4J_DATABASE) as session:
            # Segment sayısını kontrol et
            result = session.run("MATCH (s:Segment) RETURN count(s) AS count")
            segment_count = result.single()["count"]
            print(f"📍 Segment sayısı: {segment_count}")
            
            # Measure sayısını kontrol et
            result = session.run("MATCH (m:Measure) RETURN count(m) AS count")
            measure_count = result.single()["count"]
            print(f"📊 Measure sayısı: {measure_count}")
            
            # Timestamp'leri listele
            result = session.run("""
                MATCH (m:Measure)
                RETURN DISTINCT m.timestamp AS ts
                ORDER BY ts DESC
                LIMIT 10
            """)
            timestamps = [r["ts"] for r in result]
            print(f"🕒 Son 10 timestamp:")
            for ts in timestamps:
                print(f"   - {ts}")
            print()
            
            # Örnek bir Measure kaydı göster
            result = session.run("""
                MATCH (m:Measure)
                RETURN m
                LIMIT 1
            """)
            sample = result.single()
            if sample:
                print("📝 Örnek Measure kaydı:")
                measure = sample["m"]
                for key, value in dict(measure).items():
                    print(f"   {key}: {value}")
            else:
                print("⚠️  Hiç Measure kaydı bulunamadı!")
            print()
            
            # Örnek bir Segment kaydı göster
            result = session.run("""
                MATCH (s:Segment)
                RETURN s
                LIMIT 1
            """)
            sample = result.single()
            if sample:
                print("📝 Örnek Segment kaydı:")
                segment = sample["s"]
                for key, value in dict(segment).items():
                    if key == "geom":
                        print(f"   {key}: {str(value)[:50]}...")
                    else:
                        print(f"   {key}: {value}")
            else:
                print("⚠️  Hiç Segment kaydı bulunamadı!")
            print()
            
            # Segment ve Measure ilişkisini kontrol et
            result = session.run("""
                MATCH (s:Segment)
                OPTIONAL MATCH (m:Measure {segmentId: s.segmentId})
                RETURN s.segmentId AS sid, count(m) AS measureCount
                LIMIT 5
            """)
            print("🔗 Segment-Measure İlişkisi (ilk 5):")
            for record in result:
                print(f"   Segment {record['sid']}: {record['measureCount']} measure")
        
        driver.close()
        print()
        print("=" * 70)
        print("✅ Test tamamlandı!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ HATA: {e}")
        print()
        print("Olası Sorunlar:")
        print("  1. Neo4j çalışmıyor olabilir")
        print("  2. Şifre yanlış olabilir")
        print("  3. URI yanlış olabilir")
        print()
        print("Çözüm önerileri:")
        print("  - Neo4j Desktop'tan veritabanının çalıştığını kontrol edin")
        print("  - Şifre ve URI bilgilerini kontrol edin")
        print("  - 'neo4j_loader.py --init-schema' komutunu çalıştırın")

if __name__ == "__main__":
    test_connection()
