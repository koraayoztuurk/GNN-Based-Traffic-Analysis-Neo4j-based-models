#!/usr/bin/env python3
"""
compare_benchmark_files.py - İki Benchmark Dosyasını Karşılaştır
"""
import json
from pathlib import Path

print("\n" + "=" * 90)
print("🔍 BENCHMARK DOSYALARI KARŞILAŞTIRMASI")
print("=" * 90)
print()

# 1. benchmark_databases.py sonuçları
db_results_file = Path("benchmark_results.json")
if db_results_file.exists():
    with open(db_results_file) as f:
        db_results = json.load(f)
    print("📄 benchmark_databases.py (benchmark_results.json):")
    print(f"   Tarih: {db_results.get('timestamp', 'N/A')}")
    print(f"   Süre: {db_results.get('duration_seconds', 'N/A'):.2f}s")
    print()
    
    if 'results' in db_results:
        print("   Test Edilen Database'ler:")
        for db in db_results['results'].keys():
            print(f"   - {db}")
        print()
        
        # Örnek sonuçlar
        print("   Örnek Sonuçlar (Connection Speed):")
        for db, data in db_results['results'].items():
            if 'Connection Speed' in data:
                time_ms = data['Connection Speed']['Time']['value']
                print(f"   - {db}: {time_ms:.2f}ms")
        print()
else:
    print("⚠️  benchmark_results.json bulunamadı")
    print()

# 2. benchmark_comprehensive.py sonuçları
comp_results_file = Path("comprehensive_benchmark_results.json")
if comp_results_file.exists():
    with open(comp_results_file) as f:
        comp_results = json.load(f)
    
    print("📄 benchmark_comprehensive.py (comprehensive_benchmark_results.json):")
    print(f"   Tarih: {comp_results['metadata']['timestamp']}")
    print(f"   Profil: {comp_results['metadata']['profile']}")
    print()
    
    print("   Test Edilen Database'ler:")
    for db in comp_results['metadata']['databases_tested']:
        print(f"   - {db}")
    print()
    
    # Örnek sonuçlar
    print("   Örnek Sonuçlar (Connection Speed - Mean):")
    for db in comp_results['results'].keys():
        mean_time = comp_results['results'][db]['Connection Speed']['Time']['statistics']['mean']
        print(f"   - {db}: {mean_time:.2f}ms")
    print()
else:
    print("⚠️  comprehensive_benchmark_results.json bulunamadı")
    print()

print("=" * 90)
print("🔍 ANA FARKLAR:")
print("=" * 90)
print()

print("1. 📊 TEST METODOLOJİSİ:")
print("   benchmark_databases.py:")
print("   - TEK İTERASYON (her test 1 kez çalışır)")
print("   - HIZLI TEST (saniyeler içinde biter)")
print("   - Basit metrikler (time, count)")
print("   - Warmup yok")
print()
print("   benchmark_comprehensive.py:")
print("   - ÇOK İTERASYON (10+ tekrar)")
print("   - WARMUP runs (3 kez ısınma)")
print("   - İstatistiksel analiz (mean, median, std, p90, p95, p99)")
print("   - Stress test, concurrent users")
print()

print("2. 🎯 TEST KAPSAMİ:")
print("   benchmark_databases.py:")
print("   - Temel CRUD işlemleri")
print("   - Basit graph traversal")
print("   - Bellek kullanımı")
print()
print("   benchmark_comprehensive.py:")
print("   - 8 farklı test kategorisi")
print("   - Graph traversal (1-hop, 2-hop, 3-hop)")
print("   - Shortest path algoritmaları")
print("   - Concurrent reads (20 kullanıcı)")
print("   - Stress test (30 saniye)")
print("   - Write performance")
print()

print("3. ⏱️  ÖLÇÜM YÖNTEMİ:")
print("   benchmark_databases.py:")
print("   - Tek ölçüm → tek sonuç")
print("   - Cache'e bağımlı (ilk çalıştırma yavaş olabilir)")
print("   - Tutarsız sonuçlar verebilir")
print()
print("   benchmark_comprehensive.py:")
print("   - 10 ölçüm → istatistiksel ortalama")
print("   - Warmup ile cache optimize edilir")
print("   - Güvenilir, tekrarlanabilir sonuçlar")
print()

print("4. 🏆 KAZANAN BELİRLEME:")
print("   benchmark_databases.py:")
print("   - En düşük tek değer kazanır")
print("   - Şansa bağlı olabilir")
print()
print("   benchmark_comprehensive.py:")
print("   - En düşük ORTALAMA kazanır")
print("   - İstatistiksel olarak anlamlı")
print()

print("=" * 90)
print("💡 SONUÇ:")
print("=" * 90)
print()
print("FARKLI SONUÇLARIN NEDENİ:")
print()
print("1. ⚡ CACHE ETKİSİ:")
print("   - İlk test: Database cache'i boş → yavaş")
print("   - İkinci test: Cache dolu → hızlı")
print("   - benchmark_databases.py cache'e çok duyarlı")
print("   - benchmark_comprehensive.py warmup ile cache'i optimize eder")
print()
print("2. 📊 ÖRNEKLEMİN BÜYÜKLÜĞÜ:")
print("   - Tek ölçüm: Anlık sistem durumuna bağlı")
print("   - CPU kullanımı, I/O yükü, network latency")
print("   - 10+ ölçüm: Bu varyasyonları ortalayarak daha doğru sonuç")
print()
print("3. 🎲 TEST ZAMANLAMA:")
print("   - Testler farklı zamanlarda çalıştırıldı")
print("   - Sistem kaynakları değişmiş olabilir")
print("   - Arka planda çalışan uygulamalar")
print()
print("4. 🔄 QUERY OPTİMİZASYONU:")
print("   - Database'ler query planlarını cache'ler")
print("   - Aynı query ikinci kez daha hızlı çalışır")
print("   - benchmark_comprehensive.py bunu dikkate alır")
print()

print("=" * 90)
print("✅ HANGİ SONUÇLARA GÜVENMELİYİZ?")
print("=" * 90)
print()
print("🏆 benchmark_comprehensive.py DAHA GÜVENİLİR çünkü:")
print()
print("   ✓ İstatistiksel analiz (mean, std, percentiles)")
print("   ✓ Warmup ile cache optimize edilmiş")
print("   ✓ Çoklu iterasyon ile varyasyon elimine edilmiş")
print("   ✓ Stress test ve concurrent load testleri")
print("   ✓ Gerçek dünya senaryolarına daha yakın")
print()
print("   benchmark_databases.py:")
print("   ✓ Hızlı genel bakış için iyi")
print("   ✗ Tek ölçüm güvenilir değil")
print("   ✗ Cache etkisine çok duyarlı")
print()

print("=" * 90)
print("📝 ÖNERİ:")
print("=" * 90)
print()
print("Performans kararları için:")
print("   → benchmark_comprehensive.py sonuçlarını kullanın")
print("   → --profile standard veya --profile performance")
print("   → En az 10 iterasyon")
print("   → Warmup ile başlayın")
print()
print("Hızlı kontrol için:")
print("   → benchmark_databases.py kullanılabilir")
print("   → Ama sonuçlara %100 güvenmeyin")
print("   → Birden fazla çalıştırıp ortalama alın")
print()

print("=" * 90)
