#!/usr/bin/env python3
"""
check_benchmark_completeness.py - Benchmark Tamamlılık Kontrolü
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
results_file = ROOT_DIR / "outputs" / "benchmarks" / "comprehensive_benchmark_results.json"

with open(results_file, 'r') as f:
    data = json.load(f)

print("\n" + "=" * 80)
print("🔍 BENCHMARK TAMAMLILIK KONTROLÜ")
print("=" * 80)
print()

results = data['results']

# Beklenen tüm testler
expected_tests = [
    'Connection Speed',
    'Read Performance',
    'Graph Traversal',
    'Shortest Path',
    'Aggregation',
    'Write Performance',
    'Concurrent Reads',
    'Stress Test'
]

print("📋 Test Durumu:\n")

for test in expected_tests:
    print(f"  {test}:")
    for db in results.keys():
        status = "✅ OK" if test in results[db] else "❌ EKSIK"
        print(f"    - {db:12s}: {status}")
    print()

# Eksik testleri topla
missing = {}
for db in results.keys():
    missing[db] = [test for test in expected_tests if test not in results[db]]

print("=" * 80)
print("📊 ÖZET")
print("=" * 80)
print()

for db, tests in missing.items():
    if tests:
        print(f"⚠️  {db}: {len(tests)} test eksik")
        for test in tests:
            print(f"   - {test}")
    else:
        print(f"✅ {db}: Tüm testler tamamlandı")
    print()

# Başarılı test istatistikleri
total_tests = len(expected_tests)
for db in results.keys():
    completed = total_tests - len(missing[db])
    percentage = (completed / total_tests) * 100
    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    print(f"{db:12s}: {bar} {completed}/{total_tests} ({percentage:.0f}%)")

print()
print("=" * 80)
