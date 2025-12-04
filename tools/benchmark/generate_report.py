#!/usr/bin/env python3
"""
full_benchmark_report.py - Tam Benchmark Raporu
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
results_file = ROOT_DIR / "outputs" / "benchmarks" / "comprehensive_benchmark_results.json"

with open(results_file, 'r') as f:
    data = json.load(f)

print("\n" + "=" * 90)
print("📊 KAPSAMLI BENCHMARK RAPORU - TÜM DATABASE'LER")
print("=" * 90)
print()

# Metadata
meta = data['metadata']
print(f"🕐 Tarih: {meta['timestamp']}")
print(f"📋 Profil: {meta['profile'].upper()}")
print(f"💾 Database'ler: {', '.join([db.upper() for db in meta['databases_tested']])}")
print()

results = data['results']

# Her test için detaylı karşılaştırma
print("=" * 90)
print("🏁 TEST SONUÇLARI (Tüm Metrikler)")
print("=" * 90)
print()

# 1. CONNECTION SPEED
print("1️⃣  CONNECTION SPEED (Bağlantı Hızı) - Düşük daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Connection Speed']['Time']['statistics']
    print(f"  {db.upper():12s}: Mean={stats['mean']:7.3f}ms  Min={stats['min']:6.3f}ms  "
          f"Max={stats['max']:6.3f}ms  StdDev={stats['std']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Connection Speed']['Time']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 2. READ PERFORMANCE
print("2️⃣  READ PERFORMANCE (Okuma Performansı) - Düşük daha iyi")
print("─" * 90)
print("  📦 Segment Okuma:")
for db in results.keys():
    stats = results[db]['Read Performance']['segments']['statistics']
    print(f"     {db.upper():12s}: Mean={stats['mean']:7.3f}ms  Min={stats['min']:6.3f}ms  "
          f"Max={stats['max']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Read Performance']['segments']['statistics']['mean'])
print(f"     🏆 Kazanan: {winner.upper()}")

print("  📊 Measure Okuma:")
for db in results.keys():
    stats = results[db]['Read Performance']['measures']['statistics']
    print(f"     {db.upper():12s}: Mean={stats['mean']:7.3f}ms  Min={stats['min']:6.3f}ms  "
          f"Max={stats['max']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Read Performance']['measures']['statistics']['mean'])
print(f"     🏆 Kazanan: {winner.upper()}")
print()

# 3. GRAPH TRAVERSAL
print("3️⃣  GRAPH TRAVERSAL (Graf Dolaşımı) - Düşük daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Graph Traversal']['Time']['statistics']
    count = results[db]['Graph Traversal']['Count']['statistics']['mean']
    print(f"  {db.upper():12s}: Mean={stats['mean']:8.3f}ms  Visited={count:6.1f} nodes  "
          f"StdDev={stats['std']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Graph Traversal']['Time']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 4. SHORTEST PATH
print("4️⃣  SHORTEST PATH (En Kısa Yol) - Düşük daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Shortest Path']['Time']['statistics']
    path_len = results[db]['Shortest Path']['Path Length']['statistics']['mean']
    print(f"  {db.upper():12s}: Mean={stats['mean']:7.3f}ms  Path Length={path_len:4.1f}  "
          f"StdDev={stats['std']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Shortest Path']['Time']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 5. AGGREGATION
print("5️⃣  AGGREGATION (Toplama İşlemleri) - Düşük daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Aggregation']['Time']['statistics']
    result_val = results[db]['Aggregation']['Result']['statistics']['mean']
    print(f"  {db.upper():12s}: Mean={stats['mean']:8.3f}ms  Avg Speed={result_val:6.2f} km/h  "
          f"StdDev={stats['std']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Aggregation']['Time']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 6. WRITE PERFORMANCE
print("6️⃣  WRITE PERFORMANCE (Yazma Performansı) - Düşük daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Write Performance']['time_per_write']['statistics']
    total_time = results[db]['Write Performance']['total_time']['statistics']['mean']
    print(f"  {db.upper():12s}: Per Write={stats['mean']:7.3f}ms  Total={total_time:8.1f}ms  "
          f"StdDev={stats['std']:6.3f}ms")
winner = min(results.keys(), key=lambda db: results[db]['Write Performance']['time_per_write']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 7. CONCURRENT READS
print("7️⃣  CONCURRENT READS (Eşzamanlı Okuma) - Yüksek daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Concurrent Reads']['throughput']['statistics']
    avg_time = results[db]['Concurrent Reads']['avg_response_time']['statistics']['mean']
    print(f"  {db.upper():12s}: Throughput={stats['mean']:8.2f} req/s  "
          f"Avg Response={avg_time:7.3f}ms  StdDev={stats['std']:6.2f}")
winner = max(results.keys(), key=lambda db: results[db]['Concurrent Reads']['throughput']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# 8. STRESS TEST
print("8️⃣  STRESS TEST (Stres Testi) - Yüksek daha iyi")
print("─" * 90)
for db in results.keys():
    stats = results[db]['Stress Test']['throughput']['statistics']
    error_rate = results[db]['Stress Test']['error_rate']['statistics']['mean']
    print(f"  {db.upper():12s}: Throughput={stats['mean']:8.2f} req/s  "
          f"Error Rate={error_rate:5.2f}%  StdDev={stats['std']:6.2f}")
winner = max(results.keys(), key=lambda db: results[db]['Stress Test']['throughput']['statistics']['mean'])
print(f"  🏆 Kazanan: {winner.upper()}")
print()

# GENEL KAZANAN
print("=" * 90)
print("🏆 GENEL SKOR TABLOSU")
print("=" * 90)
print()

# Her test kategorisinde kazananı say
scores = {db: 0 for db in results.keys()}

# Connection Speed
winner = min(results.keys(), key=lambda db: results[db]['Connection Speed']['Time']['statistics']['mean'])
scores[winner] += 1

# Read Segments
winner = min(results.keys(), key=lambda db: results[db]['Read Performance']['segments']['statistics']['mean'])
scores[winner] += 1

# Read Measures
winner = min(results.keys(), key=lambda db: results[db]['Read Performance']['measures']['statistics']['mean'])
scores[winner] += 1

# Graph Traversal
winner = min(results.keys(), key=lambda db: results[db]['Graph Traversal']['Time']['statistics']['mean'])
scores[winner] += 1

# Shortest Path
winner = min(results.keys(), key=lambda db: results[db]['Shortest Path']['Time']['statistics']['mean'])
scores[winner] += 1

# Aggregation
winner = min(results.keys(), key=lambda db: results[db]['Aggregation']['Time']['statistics']['mean'])
scores[winner] += 1

# Write Performance
winner = min(results.keys(), key=lambda db: results[db]['Write Performance']['time_per_write']['statistics']['mean'])
scores[winner] += 1

# Concurrent Reads
winner = max(results.keys(), key=lambda db: results[db]['Concurrent Reads']['throughput']['statistics']['mean'])
scores[winner] += 1

# Stress Test
winner = max(results.keys(), key=lambda db: results[db]['Stress Test']['throughput']['statistics']['mean'])
scores[winner] += 1

# Sıralama
total_categories = 9
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

for rank, (db, score) in enumerate(sorted_scores, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    percentage = (score / total_categories) * 100
    bar = "█" * int(percentage / 2.5) + "░" * (40 - int(percentage / 2.5))
    print(f"{medal} {rank}. {db.upper():12s}: {score}/{total_categories} kazanım")
    print(f"   {bar} {percentage:.1f}%")
    print()

print("=" * 90)
print("📝 SONUÇ VE ÖNERİLER")
print("=" * 90)
print()

# En iyi performansı belirle
best_db = sorted_scores[0][0]
print(f"✅ En İyi Genel Performans: {best_db.upper()}")
print()

print("📊 Kullanım Senaryolarına Göre Öneriler:")
print()
print("  🔸 Hızlı Bağlantı & Düşük Latency:")
print(f"     → {min(results.keys(), key=lambda db: results[db]['Connection Speed']['Time']['statistics']['mean']).upper()}")
print()
print("  🔸 Graf İşlemleri (Traversal, Shortest Path):")
graph_winner = min(results.keys(), key=lambda db: results[db]['Graph Traversal']['Time']['statistics']['mean'])
print(f"     → {graph_winner.upper()}")
print()
print("  🔸 Yüksek Eşzamanlılık (Concurrent Operations):")
concurrent_winner = max(results.keys(), key=lambda db: results[db]['Concurrent Reads']['throughput']['statistics']['mean'])
print(f"     → {concurrent_winner.upper()}")
print()
print("  🔸 Yazma Yoğun İşlemler (Write-Heavy):")
write_winner = min(results.keys(), key=lambda db: results[db]['Write Performance']['time_per_write']['statistics']['mean'])
print(f"     → {write_winner.upper()}")
print()

print("=" * 90)
print("✅ RAPOR TAMAMLANDI")
print("=" * 90)
print()
