#!/usr/bin/env python3
"""
analyze_benchmark.py - Benchmark Sonuçlarını Analiz Et
"""
import json
from pathlib import Path

# Benchmark sonuçlarını yükle
ROOT_DIR = Path(__file__).parent.parent.parent
results_file = ROOT_DIR / "outputs" / "benchmarks" / "comprehensive_benchmark_results.json"

with open(results_file, 'r') as f:
    data = json.load(f)

print("\n" + "=" * 80)
print("📊 BENCHMARK SONUÇLARI ANALİZİ")
print("=" * 80)
print()

# Metadata
meta = data['metadata']
print(f"🕐 Tarih: {meta['timestamp']}")
print(f"📋 Profil: {meta['profile'].upper()}")
print(f"💾 Test Edilen Database'ler: {', '.join(meta['databases_tested'])}")
print()

results = data['results']

# Tüm test kategorileri
test_categories = {
    'Connection Speed': ('Time', 'ms', False),  # (metric, unit, higher_is_better)
    'Read Performance': ('segments', 'ms', False),
    'Graph Traversal': ('Time', 'ms', False),
    'Shortest Path': ('Time', 'ms', False),
    'Aggregation': ('Time', 'ms', False),
    'Write Performance': ('time_per_write', 'ms', False),
    'Concurrent Reads': ('throughput', 'req/s', True),
    'Stress Test': ('throughput', 'req/s', True),
}

print("=" * 80)
print("🏆 PERFORMANS KARŞILAŞTIRMASI (Ortalama Değerler)")
print("=" * 80)
print()

all_scores = {db: 0 for db in results.keys()}
category_count = 0

for category, (metric_key, unit, higher_better) in test_categories.items():
    # Her kategori için database'leri karşılaştır
    category_values = {}
    
    for db in results.keys():
        try:
            if category == 'Read Performance':
                value = results[db][category]['segments']['statistics']['mean']
            elif category == 'Write Performance':
                value = results[db][category]['time_per_write']['statistics']['mean']
            elif category == 'Concurrent Reads':
                value = results[db][category]['throughput']['statistics']['mean']
            elif category == 'Stress Test':
                value = results[db][category]['throughput']['statistics']['mean']
            else:
                value = results[db][category]['Time']['statistics']['mean']
            
            category_values[db] = value
        except (KeyError, TypeError):
            category_values[db] = None
    
    # Geçerli değerleri olanları filtrele
    valid_values = {db: v for db, v in category_values.items() if v is not None}
    
    if not valid_values:
        continue
    
    category_count += 1
    
    # Kazananı belirle
    if higher_better:
        winner = max(valid_values, key=valid_values.get)
        best_value = max(valid_values.values())
    else:
        winner = min(valid_values, key=valid_values.get)
        best_value = min(valid_values.values())
    
    # Skorları güncelle
    all_scores[winner] += 1
    
    print(f"📌 {category}")
    print(f"   {'─' * 70}")
    
    # Değerleri sırala
    sorted_dbs = sorted(valid_values.items(), 
                       key=lambda x: x[1], 
                       reverse=higher_better)
    
    for db, value in sorted_dbs:
        is_winner = db == winner
        symbol = "🥇" if is_winner else "  "
        
        # Yüzdesel fark hesapla
        if higher_better:
            pct_diff = ((value / best_value) - 1) * 100
        else:
            pct_diff = ((value / best_value) - 1) * 100
        
        pct_str = "" if is_winner else f" (+{pct_diff:.1f}%)" if pct_diff > 0 else f" ({pct_diff:.1f}%)"
        
        print(f"   {symbol} {db:12s}: {value:10.3f} {unit}{pct_str}")
    
    print()

# Genel skor tablosu
print("=" * 80)
print("🏆 GENEL PERFORMANS SKORU")
print("=" * 80)
print()

sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

for rank, (db, score) in enumerate(sorted_scores, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    percentage = (score / category_count * 100) if category_count > 0 else 0
    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    print(f"{medal} {rank}. {db:12s}: {score}/{category_count} kazanım {bar} {percentage:.1f}%")

print()

# Detaylı istatistikler
print("=" * 80)
print("📈 DETAYLI İSTATİSTİKLER")
print("=" * 80)
print()

for db in results.keys():
    print(f"🔸 {db.upper()}")
    print(f"   {'─' * 70}")
    
    # Connection speed
    try:
        conn_time = results[db]['Connection Speed']['Time']['statistics']
        print(f"   Bağlantı Hızı: {conn_time['mean']:.2f}ms (min: {conn_time['min']:.2f}, max: {conn_time['max']:.2f})")
    except:
        pass
    
    # Read throughput
    try:
        seg_time = results[db]['Read Performance']['segments']['statistics']['mean']
        meas_time = results[db]['Read Performance']['measures']['statistics']['mean']
        print(f"   Okuma Hızı: Segment={seg_time:.2f}ms, Measure={meas_time:.2f}ms")
    except:
        pass
    
    # Graph operations
    try:
        trav_time = results[db]['Graph Traversal']['Time']['statistics']['mean']
        path_time = results[db]['Shortest Path']['Time']['statistics']['mean']
        print(f"   Graph İşlemleri: Traversal={trav_time:.2f}ms, Shortest Path={path_time:.2f}ms")
    except:
        pass
    
    # Write performance
    try:
        write_time = results[db]['Write Performance']['time_per_write']['statistics']['mean']
        print(f"   Yazma Hızı: {write_time:.2f}ms per write")
    except:
        pass
    
    # Concurrent & Stress
    try:
        conc_throughput = results[db]['Concurrent Reads']['throughput']['statistics']['mean']
        stress_throughput = results[db]['Stress Test']['throughput']['statistics']['mean']
        print(f"   Eşzamanlı Performans: {conc_throughput:.1f} req/s (stress: {stress_throughput:.1f} req/s)")
    except:
        pass
    
    print()

print("=" * 80)
print("✅ ANALİZ TAMAMLANDI")
print("=" * 80)
print()
