#!/usr/bin/env python3
"""
REAL-TIME LOOP MONITORING - Detaylı İzleme Dashboard
Loop çalışırken her 30 saniyede bir database durumunu gösterir
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv('config/.env')

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def clear_screen():
    """Konsolu temizle"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_stats(session):
    """Database istatistiklerini al"""
    stats = {}
    
    # Node sayıları
    result = session.run("MATCH (s:Segment) RETURN count(s) AS cnt")
    stats['segments'] = result.single()["cnt"]
    
    result = session.run("MATCH (m:Measure) RETURN count(m) AS cnt")
    stats['measures'] = result.single()["cnt"]
    
    # Relationship sayıları
    result = session.run("MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS cnt")
    stats['connects_to'] = result.single()["cnt"]
    
    result = session.run("MATCH ()-[r:AT_TIME]->() RETURN count(r) AS cnt")
    stats['at_time'] = result.single()["cnt"]
    
    # Timestamp bilgisi
    result = session.run("""
        MATCH (m:Measure)
        RETURN 
            min(m.timestamp) AS minTs,
            max(m.timestamp) AS maxTs,
            count(DISTINCT m.timestamp) AS uniqueTs
    """)
    rec = result.single()
    stats['min_ts'] = rec["minTs"]
    stats['max_ts'] = rec["maxTs"]
    stats['unique_ts'] = rec["uniqueTs"]
    
    # İzolasyon analizi
    result = session.run("""
        MATCH (s:Segment)
        WHERE NOT exists((s)-[:CONNECTS_TO]->())
          AND NOT exists((s)<-[:CONNECTS_TO]-())
        RETURN count(s) AS isolated
    """)
    stats['isolated'] = result.single()["isolated"]
    
    # Avg degree
    if stats['segments'] > 0:
        stats['avg_degree'] = stats['connects_to'] / stats['segments']
    else:
        stats['avg_degree'] = 0.0
    
    # Isolation percentage
    if stats['segments'] > 0:
        stats['isolation_pct'] = (stats['isolated'] / stats['segments']) * 100
    else:
        stats['isolation_pct'] = 0.0
    
    # Measures per segment
    if stats['segments'] > 0:
        stats['measures_per_segment'] = stats['measures'] / stats['segments']
    else:
        stats['measures_per_segment'] = 0.0
    
    return stats

def print_dashboard(stats, iteration):
    """Dashboard'u göster"""
    clear_screen()
    
    print("=" * 80)
    print("📊 REAL-TIME DATABASE MONITORING")
    print("=" * 80)
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔄 Refresh: #{iteration} (her 30 saniye)")
    print(f"⏸️  Durdurmak için: Ctrl + C")
    print("=" * 80)
    print()
    
    # Node istatistikleri
    print("📦 NODE SAYILARI")
    print("-" * 80)
    print(f"   Segment:  {stats['segments']:>8,}")
    print(f"   Measure:  {stats['measures']:>8,}")
    print(f"   ─────────────────")
    print(f"   TOPLAM:   {stats['segments'] + stats['measures']:>8,}")
    print()
    
    # Relationship istatistikleri
    print("🔗 RELATIONSHIP SAYILARI")
    print("-" * 80)
    print(f"   CONNECTS_TO:  {stats['connects_to']:>10,}")
    print(f"   AT_TIME:      {stats['at_time']:>10,}")
    print(f"   ─────────────────────")
    print(f"   TOPLAM:       {stats['connects_to'] + stats['at_time']:>10,}")
    print()
    
    # Temporal coverage
    print("📅 TEMPORAL COVERAGE")
    print("-" * 80)
    if stats['min_ts']:
        print(f"   İlk ölçüm:      {stats['min_ts']}")
        print(f"   Son ölçüm:      {stats['max_ts']}")
        print(f"   Unique zaman:   {stats['unique_ts']:,} timestamp")
        print(f"   Segment başına: {stats['measures_per_segment']:.2f} ölçüm")
    else:
        print("   ⚠️  Henüz ölçüm yok")
    print()
    
    # Topology quality
    print("🌐 TOPOLOGY KALITESI")
    print("-" * 80)
    print(f"   Bağlı segment:    {stats['segments'] - stats['isolated']:,} / {stats['segments']:,}")
    print(f"   İzole segment:    {stats['isolated']:,} ({stats['isolation_pct']:.1f}%)")
    print(f"   Ortalama derece:  {stats['avg_degree']:.2f} komşu/segment")
    
    # Durum değerlendirmesi
    if stats['isolation_pct'] < 1:
        quality = "🎉 MÜKEMMEL"
    elif stats['isolation_pct'] < 5:
        quality = "✅ İYİ"
    elif stats['isolation_pct'] < 20:
        quality = "⚠️  KABUL EDİLEBİLİR"
    else:
        quality = "❌ SORUNLU"
    
    print(f"   Durum:            {quality}")
    print()
    
    # GNN Readiness
    print("🎯 GNN HAZIRLIK DURUMU")
    print("-" * 80)
    
    checks = []
    checks.append(("Segment var", stats['segments'] > 0))
    checks.append(("Measure var", stats['measures'] > 0))
    checks.append(("CONNECTS_TO var", stats['connects_to'] > 0))
    checks.append(("AT_TIME var", stats['at_time'] > 0))
    checks.append(("İzolasyon < %5", stats['isolation_pct'] < 5))
    checks.append(("AT_TIME = Measure", stats['at_time'] == stats['measures']))
    
    ready_count = sum(1 for _, status in checks if status)
    total_checks = len(checks)
    
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")
    
    print()
    print(f"   📊 Skor: {ready_count}/{total_checks} ({ready_count/total_checks*100:.0f}%)")
    
    if ready_count == total_checks:
        print(f"   🎉 SİSTEM TAMAMEN HAZIR!")
    elif ready_count >= total_checks * 0.8:
        print(f"   ✅ İyi durumda, eksikler tamamlanıyor...")
    else:
        print(f"   ⏳ Veri toplanıyor, bekleyin...")
    
    print()
    print("=" * 80)
    print("💡 Pipeline çalışıyor, veriler otomatik güncelleniyor...")
    print("=" * 80)

def main():
    print("Monitoring başlıyor...")
    print("Neo4j'ye bağlanılıyor...")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        iteration = 0
        
        while True:
            iteration += 1
            
            with driver.session(database=NEO4J_DATABASE) as session:
                stats = get_stats(session)
            
            print_dashboard(stats, iteration)
            
            time.sleep(30)  # 30 saniye bekle
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring durduruldu!")
        print("✨ İyi günler!")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    main()
