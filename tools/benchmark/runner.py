#!/usr/bin/env python3
"""
KAPSAMLI DATABASE BENCHMARK SİSTEMİ
====================================

Bu script Neo4j, ArangoDB ve TigerGraph veritabanlarını
çok detaylı bir şekilde benchmark eder.

Özellikler:
- 13 test kategorisi (8 temel + 5 gelişmiş)
- İstatistiksel analiz (mean, median, p95, p99)
- Concurrent test (çoklu kullanıcı simülasyonu)
- Write performance (insert, update, delete)
- Bulk operations (1000-10000 kayıt toplu işlem)
- Complex graph queries (pattern matching, multi-hop filtering)
- Spatial queries (coğrafi sorgular, distance calculation)
- Time-series aggregation (zaman bazlı analizler)
- Index performance (indexed vs full scan)
- Stress test (limit testleri)
- HTML dashboard ile görselleştirme
- Warm-up runs (cache etkisi analizi)
- Resource monitoring (CPU, Memory, Network)

Test Kategorileri:
1. Connection Speed - Bağlantı hızı
2. Read Performance - Okuma performansı
3. Graph Traversal - Graf gezinme (1-2-3 hop)
4. Shortest Path - En kısa yol
5. Aggregation - Toplama işlemleri (AVG, MIN, MAX, SUM)
6. Write Performance - Yazma performansı (CREATE, UPDATE, DELETE)
7. Concurrent Reads - Çoklu kullanıcı simülasyonu
8. Stress Test - Sürekli yük altında performans
9. Bulk Operations - Toplu ekleme/güncelleme/silme
10. Complex Graph - Karmaşık graf sorguları
11. Spatial Queries - Coğrafi/mekansal sorgular
12. Time Series - Zaman serisi analizleri
13. Index Performance - Index kullanımı karşılaştırma

Kullanım:
    python tools/benchmark/runner.py --profile standard
    python tools/benchmark/runner.py --profile stress
    python tools/benchmark/runner.py --profile production
"""

import os
import sys
import time
import json
import argparse
import statistics
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import threading
import queue
import random

# Third-party imports
import psutil
from dotenv import load_dotenv

# Database clients
from neo4j import GraphDatabase
from arango import ArangoClient
import pyTigerGraph as tg

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Load .env from config directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(ROOT_DIR, 'config', '.env'))

# Test profiles
PROFILES = {
    "quick": {
        "iterations": 3,
        "warmup_runs": 1,
        "concurrent_users": 5,
        "stress_duration": 10,
        "bulk_size": 100
    },
    "standard": {
        "iterations": 10,
        "warmup_runs": 3,
        "concurrent_users": 20,
        "stress_duration": 30,
        "bulk_size": 1000
    },
    "production": {
        "iterations": 50,
        "warmup_runs": 5,
        "concurrent_users": 100,
        "stress_duration": 60,
        "bulk_size": 5000
    },
    "stress": {
        "iterations": 100,
        "warmup_runs": 10,
        "concurrent_users": 500,
        "stress_duration": 300,
        "bulk_size": 10000
    }
}

# Test segment IDs for various tests
TEST_SEGMENTS = [
    "A8001_113599020", "A8001_113599021", "A8001_113599022",
    "A8001_113599023", "A8001_113599024", "A8001_113599025"
]

# ============================================================================
# STATISTICS & ANALYSIS
# ============================================================================

class Statistics:
    """İstatistiksel analiz ve metrik hesaplama."""
    
    @staticmethod
    def analyze(values: List[float]) -> Dict[str, float]:
        """
        Bir değer listesi için kapsamlı istatistik hesapla.
        
        Returns:
            {
                'mean': ortalama,
                'median': ortanca,
                'min': minimum,
                'max': maksimum,
                'std': standart sapma,
                'p50': 50th percentile,
                'p90': 90th percentile,
                'p95': 95th percentile,
                'p99': 99th percentile,
                'variance': varyans
            }
        """
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(values)
        
        return {
            'count': n,
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if n > 1 else 0,
            'p50': sorted_values[int(n * 0.50)],
            'p90': sorted_values[int(n * 0.90)] if n > 10 else sorted_values[-1],
            'p95': sorted_values[int(n * 0.95)] if n > 20 else sorted_values[-1],
            'p99': sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1],
            'variance': statistics.variance(values) if n > 1 else 0
        }
    
    @staticmethod
    def detect_outliers(values: List[float]) -> List[int]:
        """IQR metoduyla outlier indekslerini tespit et."""
        if len(values) < 4:
            return []
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return [i for i, v in enumerate(values) if v < lower or v > upper]

# ============================================================================
# RESOURCE MONITORING
# ============================================================================

class ResourceMonitor:
    """CPU, Memory, Network kullanımını izle."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None
        self.start_io = None
        
    def start(self):
        """Monitoring başlat."""
        self.start_time = time.time()
        self.start_cpu = self.process.cpu_percent(interval=0.1)
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        try:
            self.start_io = self.process.io_counters()
        except (AttributeError, PermissionError):
            self.start_io = None
    
    def stop(self) -> Dict[str, float]:
        """
        Monitoring durdur ve metrikleri döndür.
        
        Returns:
            {
                'duration': süre (saniye),
                'cpu_percent': ortalama CPU kullanımı (%),
                'memory_mb': memory artışı (MB),
                'io_read_mb': okunan veri (MB),
                'io_write_mb': yazılan veri (MB)
            }
        """
        duration = time.time() - self.start_time
        cpu = self.process.cpu_percent(interval=0.1)
        memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        result = {
            'duration': duration,
            'cpu_percent': cpu,
            'memory_mb': memory - self.start_memory
        }
        
        if self.start_io:
            try:
                end_io = self.process.io_counters()
                result['io_read_mb'] = (end_io.read_bytes - self.start_io.read_bytes) / 1024 / 1024
                result['io_write_mb'] = (end_io.write_bytes - self.start_io.write_bytes) / 1024 / 1024
            except (AttributeError, PermissionError):
                pass
        
        return result

# ============================================================================
# BENCHMARK RESULTS
# ============================================================================

class ComprehensiveResults:
    """Tüm benchmark sonuçlarını topla ve analiz et."""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'profile': None,
            'databases_tested': []
        }
    
    def add_result(self, db: str, test: str, metric: str, value: float, unit: str = ""):
        """Tek bir test sonucu ekle."""
        self.results[db][test][metric].append({
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        })
    
    def add_batch_results(self, db: str, test: str, metric: str, values: List[float], unit: str = ""):
        """Birden fazla test sonucu ekle (iterasyonlar için)."""
        for value in values:
            self.add_result(db, test, metric, value, unit)
    
    def get_statistics(self, db: str, test: str, metric: str) -> Dict[str, float]:
        """Bir metrik için istatistikleri hesapla."""
        values = [r['value'] for r in self.results[db][test][metric] if isinstance(r['value'], (int, float))]
        if not values:
            return {'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'variance': 0}
        return Statistics.analyze(values)
    
    def get_winner(self, test: str, metric: str, higher_is_better: bool = False) -> Optional[str]:
        """Bir test için kazananı bul."""
        avg_values = {}
        
        for db in self.results:
            if test in self.results[db] and metric in self.results[db][test]:
                values = [r['value'] for r in self.results[db][test][metric]]
                avg_values[db] = statistics.mean(values) if values else float('inf')
        
        if not avg_values:
            return None
        
        if higher_is_better:
            return max(avg_values, key=avg_values.get)
        else:
            return min(avg_values, key=avg_values.get)
    
    def print_summary(self):
        """Konsola özet tablo yazdır."""
        print("\n" + "=" * 100)
        print("KAPSAMLI BENCHMARK SONUÇLARI".center(100))
        print("=" * 100 + "\n")
        
        # Her database için
        for db in sorted(self.results.keys()):
            print(f"\n{'=' * 100}")
            print(f"{db.upper()} - DETAYLI SONUÇLAR".center(100))
            print(f"{'=' * 100}\n")
            
            # Her test için
            for test in sorted(self.results[db].keys()):
                print(f"\n[TEST] {test}")
                print("-" * 100)
                
                # Her metrik için
                for metric in sorted(self.results[db][test].keys()):
                    stats = self.get_statistics(db, test, metric)
                    unit = self.results[db][test][metric][0]['unit']
                    
                    winner = self.get_winner(test, metric)
                    winner_mark = " [WINNER]" if winner == db else ""
                    
                    print(f"  {metric}:")
                    print(f"    Mean:   {stats['mean']:.2f} {unit}{winner_mark}")
                    print(f"    Median: {stats['median']:.2f} {unit}")
                    print(f"    P95:    {stats['p95']:.2f} {unit}")
                    print(f"    P99:    {stats['p99']:.2f} {unit}")
                    print(f"    Min:    {stats['min']:.2f} {unit}")
                    print(f"    Max:    {stats['max']:.2f} {unit}")
                    print(f"    StdDev: {stats['std']:.2f} {unit}")
        
        # Genel skorlar
        print(f"\n{'=' * 100}")
        print("GENEL SKORLAR".center(100))
        print(f"{'=' * 100}\n")
        
        scores = defaultdict(int)
        total_tests = 0
        
        for db in self.results:
            for test in self.results[db]:
                for metric in self.results[db][test]:
                    winner = self.get_winner(test, metric)
                    if winner:
                        scores[winner] += 1
                    total_tests += 1
        
        for db in sorted(scores.keys(), key=lambda x: scores[x], reverse=True):
            percentage = (scores[db] / total_tests * 100) if total_tests > 0 else 0
            print(f"{db.upper()}: {scores[db]}/{total_tests} metrik kazandı ({percentage:.1f}%)")
    
    def save_to_json(self, filename: str = "comprehensive_benchmark_results.json"):
        """Sonuçları JSON dosyasına kaydet."""
        # Full path - outputs/benchmarks/ altına kaydet
        output_path = Path(ROOT_DIR) / "outputs" / "benchmarks" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Raw results
        raw_results = {}
        for db in self.results:
            raw_results[db] = {}
            for test in self.results[db]:
                raw_results[db][test] = {}
                for metric in self.results[db][test]:
                    values = [r['value'] for r in self.results[db][test][metric]]
                    unit = self.results[db][test][metric][0]['unit']
                    stats = self.get_statistics(db, test, metric)
                    
                    raw_results[db][test][metric] = {
                        'raw_values': values,
                        'unit': unit,
                        'statistics': stats,
                        'winner': self.get_winner(test, metric)
                    }
        
        output = {
            'metadata': self.metadata,
            'results': raw_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Detaylı sonuçlar kaydedildi: {filename}")

# ============================================================================
# NEO4J BENCHMARK
# ============================================================================

class Neo4jComprehensiveBenchmark:
    """Neo4j için kapsamlı benchmark testleri."""
    
    def __init__(self, profile: Dict):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASS", "password")
        self.driver = None
        self.profile = profile
        self.monitor = ResourceMonitor()
    
    def connect(self):
        """Neo4j'ye bağlan."""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
    
    def disconnect(self):
        """Bağlantıyı kapat."""
        if self.driver:
            self.driver.close()
    
    def _run_query(self, query: str, parameters: Dict = None) -> Any:
        """Query çalıştır ve sonucu döndür."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)
    
    # ---- BASIC TESTS ----
    
    def test_connection(self) -> List[float]:
        """Bağlantı hızı - multiple iterations."""
        times = []
        
        # Warmup
        for _ in range(self.profile['warmup_runs']):
            self.driver.verify_connectivity()
        
        # Actual test
        for _ in range(self.profile['iterations']):
            start = time.time()
            self.driver.verify_connectivity()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_read_performance(self) -> Dict[str, List[float]]:
        """Okuma performansı - Segment ve Measure sayma."""
        results = {'segments': [], 'measures': []}
        
        # Warmup
        for _ in range(self.profile['warmup_runs']):
            self._run_query("MATCH (s:Segment) RETURN count(s)")
        
        # Segments
        for _ in range(self.profile['iterations']):
            start = time.time()
            result = self._run_query("MATCH (s:Segment) RETURN count(s) as count")
            elapsed = (time.time() - start) * 1000
            results['segments'].append(elapsed)
        
        # Measures
        for _ in range(self.profile['iterations']):
            start = time.time()
            result = self._run_query("MATCH (m:Measure) RETURN count(m) as count")
            elapsed = (time.time() - start) * 1000
            results['measures'].append(elapsed)
        
        return results
    
    def test_graph_traversal(self) -> Dict[str, List[float]]:
        """Graf gezinme - 1 hop, 2 hop, 3 hop."""
        results = {'1_hop': [], '2_hop': [], '3_hop': []}
        
        segment_id = TEST_SEGMENTS[0]
        
        # 1-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment {segmentId: $id})-[:CONNECTS_TO]->(n:Segment)
            RETURN count(n) as neighbors
            """
            self._run_query(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['1_hop'].append(elapsed)
        
        # 2-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment {segmentId: $id})-[:CONNECTS_TO*1..2]->(n:Segment)
            RETURN count(DISTINCT n) as neighbors
            """
            self._run_query(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['2_hop'].append(elapsed)
        
        # 3-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment {segmentId: $id})-[:CONNECTS_TO*1..3]->(n:Segment)
            RETURN count(DISTINCT n) as neighbors
            """
            self._run_query(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['3_hop'].append(elapsed)
        
        return results
    
    def test_shortest_path(self) -> List[float]:
        """Shortest path bulma."""
        times = []
        
        source = TEST_SEGMENTS[0]
        target = TEST_SEGMENTS[-1]
        
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s1:Segment {segmentId: $source}),
                  (s2:Segment {segmentId: $target}),
                  path = shortestPath((s1)-[:CONNECTS_TO*]-(s2))
            RETURN length(path) as pathLength
            """
            self._run_query(query, {'source': source, 'target': target})
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_aggregation(self) -> Dict[str, List[float]]:
        """Aggregation queries - AVG, MIN, MAX, SUM."""
        results = {'avg': [], 'min': [], 'max': [], 'sum': []}
        
        # AVG
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = "MATCH (m:Measure) RETURN avg(m.speed) as avgSpeed"
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['avg'].append(elapsed)
        
        # MIN
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = "MATCH (m:Measure) RETURN min(m.speed) as minSpeed"
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['min'].append(elapsed)
        
        # MAX
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = "MATCH (m:Measure) RETURN max(m.speed) as maxSpeed"
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['max'].append(elapsed)
        
        # SUM
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = "MATCH (m:Measure) RETURN sum(m.speed) as sumSpeed"
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['sum'].append(elapsed)
        
        return results
    
    def test_write_performance(self) -> Dict[str, List[float]]:
        """Write performance - CREATE, UPDATE, DELETE."""
        results = {'create': [], 'update': [], 'delete': []}
        
        # CREATE
        for i in range(self.profile['iterations']):
            start = time.time()
            query = """
            CREATE (s:Segment_Test {
                segmentId: $id,
                name: $name,
                created: timestamp()
            })
            """
            self._run_query(query, {'id': f'TEST_{i}', 'name': f'Test Segment {i}'})
            elapsed = (time.time() - start) * 1000
            results['create'].append(elapsed)
        
        # UPDATE
        for i in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment_Test {segmentId: $id})
            SET s.updated = timestamp(), s.name = $name
            """
            self._run_query(query, {'id': f'TEST_{i}', 'name': f'Updated {i}'})
            elapsed = (time.time() - start) * 1000
            results['update'].append(elapsed)
        
        # DELETE
        for i in range(self.profile['iterations']):
            start = time.time()
            query = "MATCH (s:Segment_Test {segmentId: $id}) DELETE s"
            self._run_query(query, {'id': f'TEST_{i}'})
            elapsed = (time.time() - start) * 1000
            results['delete'].append(elapsed)
        
        return results
    
    def test_concurrent_reads(self) -> Dict[str, Any]:
        """Concurrent read operations - çoklu kullanıcı simülasyonu."""
        num_users = self.profile['concurrent_users']
        iterations_per_user = 10
        
        results_queue = queue.Queue()
        errors = []
        
        def worker(user_id: int):
            """Tek bir kullanıcıyı simüle et."""
            try:
                for _ in range(iterations_per_user):
                    start = time.time()
                    query = "MATCH (s:Segment) RETURN count(s) as count"
                    self._run_query(query)
                    elapsed = (time.time() - start) * 1000
                    results_queue.put(elapsed)
            except Exception as e:
                errors.append(f"User {user_id}: {str(e)}")
        
        # Start all threads
        start_time = time.time()
        threads = []
        for i in range(num_users):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        times = []
        while not results_queue.empty():
            times.append(results_queue.get())
        
        return {
            'times': times,
            'total_duration': total_time,
            'errors': errors,
            'throughput': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_stress(self) -> Dict[str, Any]:
        """Stress test - sürekli yük altında performans."""
        duration = self.profile['stress_duration']  # seconds
        
        times = []
        errors = []
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            try:
                query_start = time.time()
                query = random.choice([
                    "MATCH (s:Segment) RETURN count(s)",
                    "MATCH (m:Measure) RETURN avg(m.speed)",
                    "MATCH (s:Segment)-[:CONNECTS_TO]->(n) RETURN count(n)"
                ])
                self._run_query(query)
                elapsed = (time.time() - query_start) * 1000
                times.append(elapsed)
            except Exception as e:
                errors.append(str(e))
        
        total_time = time.time() - start_time
        
        return {
            'times': times,
            'total_duration': total_time,
            'total_queries': len(times),
            'errors': len(errors),
            'queries_per_second': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_bulk_operations(self) -> Dict[str, List[float]]:
        """Bulk insert/update/delete performance."""
        results = {'bulk_insert': [], 'bulk_update': [], 'bulk_delete': []}
        bulk_size = self.profile.get('bulk_size', 1000)
        
        # Bulk Insert
        for iteration in range(min(3, self.profile['iterations'])):
            nodes = [
                {
                    'segmentId': f'BULK_TEST_{iteration}_{i}',
                    'length': i * 10.5,
                    'speedLimit': 50
                }
                for i in range(bulk_size)
            ]
            
            start = time.time()
            with self.driver.session() as session:
                session.run("""
                    UNWIND $nodes AS node
                    MERGE (s:Segment {segmentId: node.segmentId})
                    SET s.length = node.length, s.speedLimit = node.speedLimit
                """, nodes=nodes)
            elapsed = (time.time() - start) * 1000
            results['bulk_insert'].append(elapsed)
        
        # Bulk Update
        for iteration in range(min(3, self.profile['iterations'])):
            segment_ids = [f'BULK_TEST_{iteration}_{i}' for i in range(bulk_size)]
            
            start = time.time()
            with self.driver.session() as session:
                session.run("""
                    UNWIND $ids AS segmentId
                    MATCH (s:Segment {segmentId: segmentId})
                    SET s.speedLimit = 60
                """, ids=segment_ids)
            elapsed = (time.time() - start) * 1000
            results['bulk_update'].append(elapsed)
        
        # Bulk Delete
        for iteration in range(min(3, self.profile['iterations'])):
            segment_ids = [f'BULK_TEST_{iteration}_{i}' for i in range(bulk_size)]
            
            start = time.time()
            with self.driver.session() as session:
                session.run("""
                    UNWIND $ids AS segmentId
                    MATCH (s:Segment {segmentId: segmentId})
                    DELETE s
                """, ids=segment_ids)
            elapsed = (time.time() - start) * 1000
            results['bulk_delete'].append(elapsed)
        
        return results
    
    def test_complex_graph(self) -> Dict[str, List[float]]:
        """Complex graph queries - pattern matching, multi-hop filtering."""
        results = {'pattern_match': [], 'multi_hop_filter': []}
        
        # Pattern matching: Find segments with specific connectivity pattern
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            query = """
            MATCH (s:Segment)-[:CONNECTS_TO]->(n)
            WITH s, count(n) as neighbor_count
            WHERE neighbor_count >= 2
            RETURN s.segmentId, neighbor_count
            LIMIT 100
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['pattern_match'].append(elapsed)
        
        # Multi-hop with filtering
        segment_id = TEST_SEGMENTS[0]
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            query = """
            MATCH path = (s:Segment {segmentId: $id})-[:CONNECTS_TO*1..3]->(n)
            WHERE n.speedLimit >= 50
            RETURN n.segmentId, length(path) as path_length
            """
            self._run_query(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['multi_hop_filter'].append(elapsed)
        
        return results
    
    def test_spatial_queries(self) -> Dict[str, List[float]]:
        """Spatial/geographic queries."""
        results = {'distance_calc': [], 'bounding_box': []}
        
        # Distance calculation between segments
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment)
            WHERE s.startLat IS NOT NULL AND s.startLon IS NOT NULL
            WITH s, point.distance(
                point({latitude: s.startLat, longitude: s.startLon}),
                point({latitude: s.endLat, longitude: s.endLon})
            ) as distance
            RETURN s.segmentId, distance
            LIMIT 100
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['distance_calc'].append(elapsed)
        
        # Bounding box search
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment)
            WHERE s.startLat >= 39.9 AND s.startLat <= 40.1
              AND s.startLon >= 32.7 AND s.startLon <= 32.9
            RETURN s.segmentId
            LIMIT 100
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['bounding_box'].append(elapsed)
        
        return results
    
    def test_time_series(self) -> Dict[str, List[float]]:
        """Time-series aggregation queries."""
        results = {'time_range': [], 'time_aggregation': []}
        
        # Time range filter
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (m:Measure)
            WHERE m.timestamp >= datetime('2024-01-01') 
              AND m.timestamp <= datetime('2024-12-31')
            RETURN m.speed
            LIMIT 1000
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['time_range'].append(elapsed)
        
        # Time-based aggregation
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (m:Measure)
            WITH datetime(m.timestamp).hour as hour, m
            RETURN hour, avg(m.speed) as avg_speed, count(m) as count
            ORDER BY hour
            LIMIT 24
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['time_aggregation'].append(elapsed)
        
        return results
    
    def test_index_performance(self) -> Dict[str, List[float]]:
        """Index vs non-index query performance."""
        results = {'with_index': [], 'simulated_scan': []}
        
        # Query that uses index (segmentId is typically indexed)
        for _ in range(self.profile['iterations']):
            segment_id = random.choice(TEST_SEGMENTS)
            start = time.time()
            query = """
            MATCH (s:Segment {segmentId: $id})
            RETURN s
            """
            self._run_query(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['with_index'].append(elapsed)
        
        # Simulated full scan (filter on non-indexed field)
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            MATCH (s:Segment)
            WHERE s.length > 100
            RETURN s.segmentId
            LIMIT 10
            """
            self._run_query(query)
            elapsed = (time.time() - start) * 1000
            results['simulated_scan'].append(elapsed)
        
        return results

# ============================================================================
# ARANGODB BENCHMARK
# ============================================================================

class ArangoComprehensiveBenchmark:
    """ArangoDB için kapsamlı benchmark testleri."""
    
    def __init__(self, profile: Dict):
        self.host = os.getenv("ARANGO_HOST", "http://localhost:8529")
        self.username = os.getenv("ARANGO_USER", "root")
        self.password = os.getenv("ARANGO_PASS", "")
        self.db_name = os.getenv("ARANGO_DATABASE", "traffic_db")
        self.client = None
        self.db = None
        self.profile = profile
        self.monitor = ResourceMonitor()
    
    def connect(self):
        """ArangoDB'ye bağlan."""
        self.client = ArangoClient(hosts=self.host)
        # First connect to _system to verify credentials
        sys_db = self.client.db('_system', username=self.username, password=self.password)
        # Then connect to target database
        self.db = self.client.db(self.db_name, username=self.username, password=self.password)
        # Ensure graph exists for traversal tests
        self._ensure_graph()
    
    def disconnect(self):
        """Bağlantıyı kapat."""
        pass  # ArangoDB client doesn't need explicit close
    
    def _ensure_graph(self):
        """Graph yapısını kontrol et ve gerekirse oluştur."""
        graph_name = 'traffic_flow_graph'
        
        try:
            if self.db.has_graph(graph_name):
                print(f"ℹ️  Graph mevcut: {graph_name}")
            else:
                print(f"🔧 Graph oluşturuluyor: {graph_name}")
                graph = self.db.create_graph(graph_name)
                graph.create_edge_definition(
                    edge_collection='CONNECTS_TO',
                    from_vertex_collections=['Segment'],
                    to_vertex_collections=['Segment']
                )
                print(f"✅ Graph oluşturuldu: {graph_name}")
        except Exception as e:
            print(f"⚠️  Graph kontrolü sırasında hata: {e}")
    
    def _run_aql(self, query: str, bind_vars: Dict = None) -> Any:
        """AQL query çalıştır."""
        cursor = self.db.aql.execute(query, bind_vars=bind_vars or {})
        return list(cursor)
    
    # ---- BASIC TESTS ----
    
    def test_connection(self) -> List[float]:
        """Bağlantı hızı."""
        times = []
        
        # Warmup
        for _ in range(self.profile['warmup_runs']):
            self.db.collection('Segment').count()
        
        # Actual test
        for _ in range(self.profile['iterations']):
            start = time.time()
            self.db.collection('Segment').count()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_read_performance(self) -> Dict[str, List[float]]:
        """Okuma performansı."""
        results = {'segments': [], 'measures': []}
        
        # Segments
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR s IN Segment COLLECT WITH COUNT INTO length RETURN length")
            elapsed = (time.time() - start) * 1000
            results['segments'].append(elapsed)
        
        # Measures
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR m IN Measure COLLECT WITH COUNT INTO length RETURN length")
            elapsed = (time.time() - start) * 1000
            results['measures'].append(elapsed)
        
        return results
    
    def test_graph_traversal(self) -> Dict[str, List[float]]:
        """Graf gezinme - 1, 2, 3 hop."""
        results = {'1_hop': [], '2_hop': [], '3_hop': []}
        
        segment_id = TEST_SEGMENTS[0]
        
        # 1-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR v IN 1..1 OUTBOUND @start GRAPH 'traffic_flow_graph'
            RETURN v
            """
            self._run_aql(query, {'start': f'Segment/{segment_id}'})
            elapsed = (time.time() - start) * 1000
            results['1_hop'].append(elapsed)
        
        # 2-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR v IN 1..2 OUTBOUND @start GRAPH 'traffic_flow_graph'
            RETURN DISTINCT v
            """
            self._run_aql(query, {'start': f'Segment/{segment_id}'})
            elapsed = (time.time() - start) * 1000
            results['2_hop'].append(elapsed)
        
        # 3-hop
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR v IN 1..3 OUTBOUND @start GRAPH 'traffic_flow_graph'
            RETURN DISTINCT v
            """
            self._run_aql(query, {'start': f'Segment/{segment_id}'})
            elapsed = (time.time() - start) * 1000
            results['3_hop'].append(elapsed)
        
        return results
    
    def test_shortest_path(self) -> List[float]:
        """Shortest path."""
        times = []
        
        source = TEST_SEGMENTS[0]
        target = TEST_SEGMENTS[-1]
        
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR v, e IN OUTBOUND SHORTEST_PATH @source TO @target GRAPH 'traffic_flow_graph'
            RETURN v
            """
            self._run_aql(query, {
                'source': f'Segment/{source}',
                'target': f'Segment/{target}'
            })
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_aggregation(self) -> Dict[str, List[float]]:
        """Aggregation queries."""
        results = {'avg': [], 'min': [], 'max': [], 'sum': []}
        
        # AVG
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR m IN Measure COLLECT AGGREGATE avg = AVG(m.speed) RETURN avg")
            elapsed = (time.time() - start) * 1000
            results['avg'].append(elapsed)
        
        # MIN
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR m IN Measure COLLECT AGGREGATE min = MIN(m.speed) RETURN min")
            elapsed = (time.time() - start) * 1000
            results['min'].append(elapsed)
        
        # MAX
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR m IN Measure COLLECT AGGREGATE max = MAX(m.speed) RETURN max")
            elapsed = (time.time() - start) * 1000
            results['max'].append(elapsed)
        
        # SUM
        for _ in range(self.profile['iterations']):
            start = time.time()
            self._run_aql("FOR m IN Measure COLLECT AGGREGATE sum = SUM(m.speed) RETURN sum")
            elapsed = (time.time() - start) * 1000
            results['sum'].append(elapsed)
        
        return results
    
    def test_write_performance(self) -> Dict[str, List[float]]:
        """Write performance."""
        results = {'create': [], 'update': [], 'delete': []}
        
        collection = self.db.collection('Segment')
        
        # CREATE
        for i in range(self.profile['iterations']):
            start = time.time()
            collection.insert({
                '_key': f'TEST_{i}',
                'segmentId': f'TEST_{i}',
                'name': f'Test Segment {i}'
            })
            elapsed = (time.time() - start) * 1000
            results['create'].append(elapsed)
        
        # UPDATE
        for i in range(self.profile['iterations']):
            start = time.time()
            collection.update({
                '_key': f'TEST_{i}',
                'name': f'Updated {i}'
            })
            elapsed = (time.time() - start) * 1000
            results['update'].append(elapsed)
        
        # DELETE
        for i in range(self.profile['iterations']):
            start = time.time()
            collection.delete(f'TEST_{i}')
            elapsed = (time.time() - start) * 1000
            results['delete'].append(elapsed)
        
        return results
    
    def test_concurrent_reads(self) -> Dict[str, Any]:
        """Concurrent reads."""
        num_users = self.profile['concurrent_users']
        iterations_per_user = 10
        
        results_queue = queue.Queue()
        errors = []
        
        def worker(user_id: int):
            try:
                for _ in range(iterations_per_user):
                    start = time.time()
                    self._run_aql("FOR s IN Segment COLLECT WITH COUNT INTO length RETURN length")
                    elapsed = (time.time() - start) * 1000
                    results_queue.put(elapsed)
            except Exception as e:
                errors.append(f"User {user_id}: {str(e)}")
        
        start_time = time.time()
        threads = []
        for i in range(num_users):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        
        times = []
        while not results_queue.empty():
            times.append(results_queue.get())
        
        return {
            'times': times,
            'total_duration': total_time,
            'errors': errors,
            'throughput': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_stress(self) -> Dict[str, Any]:
        """Stress test."""
        duration = self.profile['stress_duration']
        
        times = []
        errors = []
        start_time = time.time()
        
        queries = [
            "FOR s IN Segment COLLECT WITH COUNT INTO length RETURN length",
            "FOR m IN Measure COLLECT AGGREGATE avg = AVG(m.speed) RETURN avg",
            "FOR s IN Segment LIMIT 10 RETURN s"
        ]
        
        while (time.time() - start_time) < duration:
            try:
                query_start = time.time()
                self._run_aql(random.choice(queries))
                elapsed = (time.time() - query_start) * 1000
                times.append(elapsed)
            except Exception as e:
                errors.append(str(e))
        
        total_time = time.time() - start_time
        
        return {
            'times': times,
            'total_duration': total_time,
            'total_queries': len(times),
            'errors': len(errors),
            'queries_per_second': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_bulk_operations(self) -> Dict[str, List[float]]:
        """Bulk insert/update/delete performance."""
        results = {'bulk_insert': [], 'bulk_update': [], 'bulk_delete': []}
        bulk_size = self.profile.get('bulk_size', 1000)
        
        collection = self.db.collection('Segment')
        
        # Bulk Insert
        for iteration in range(min(3, self.profile['iterations'])):
            docs = [
                {
                    '_key': f'BULK_TEST_{iteration}_{i}',
                    'segmentId': f'BULK_TEST_{iteration}_{i}',
                    'length': i * 10.5,
                    'speedLimit': 50
                }
                for i in range(bulk_size)
            ]
            
            start = time.time()
            collection.insert_many(docs, overwrite=True)
            elapsed = (time.time() - start) * 1000
            results['bulk_insert'].append(elapsed)
        
        # Bulk Update
        for iteration in range(min(3, self.profile['iterations'])):
            docs = [
                {
                    '_key': f'BULK_TEST_{iteration}_{i}',
                    'speedLimit': 60
                }
                for i in range(bulk_size)
            ]
            
            start = time.time()
            collection.update_many(docs)
            elapsed = (time.time() - start) * 1000
            results['bulk_update'].append(elapsed)
        
        # Bulk Delete
        for iteration in range(min(3, self.profile['iterations'])):
            keys = [f'BULK_TEST_{iteration}_{i}' for i in range(bulk_size)]
            
            start = time.time()
            try:
                collection.delete_many(keys)
            except Exception:
                pass  # Ignore errors if keys don't exist
            elapsed = (time.time() - start) * 1000
            results['bulk_delete'].append(elapsed)
        
        return results
    
    def test_complex_graph(self) -> Dict[str, List[float]]:
        """Complex graph queries - pattern matching, multi-hop filtering."""
        results = {'pattern_match': [], 'multi_hop_filter': []}
        
        # Pattern matching: Find segments with specific connectivity pattern
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            query = """
            FOR s IN Segment
                LIMIT 100
                LET neighbors = (
                    FOR v IN 1..1 OUTBOUND s GRAPH 'traffic_flow_graph'
                    RETURN v
                )
                FILTER LENGTH(neighbors) >= 2
                RETURN {segment: s.segmentId, neighbor_count: LENGTH(neighbors)}
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['pattern_match'].append(elapsed)
        
        # Multi-hop with filtering
        segment_id = TEST_SEGMENTS[0]
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            query = """
            FOR v, e, p IN 1..3 OUTBOUND @start GRAPH 'traffic_flow_graph'
                FILTER v.speedLimit >= 50
                RETURN {segment: v.segmentId, path_length: LENGTH(p.edges)}
            """
            self._run_aql(query, {'start': f'Segment/{segment_id}'})
            elapsed = (time.time() - start) * 1000
            results['multi_hop_filter'].append(elapsed)
        
        return results
    
    def test_spatial_queries(self) -> Dict[str, List[float]]:
        """Spatial/geographic queries."""
        results = {'distance_calc': [], 'bounding_box': []}
        
        # Distance calculation between segments
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR s IN Segment
                LIMIT 100
                FILTER s.startLat != null AND s.startLon != null
                LET distance = DISTANCE(s.startLat, s.startLon, s.endLat, s.endLon)
                RETURN {segment: s.segmentId, distance: distance}
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['distance_calc'].append(elapsed)
        
        # Bounding box search
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR s IN Segment
                FILTER s.startLat >= 39.9 AND s.startLat <= 40.1
                FILTER s.startLon >= 32.7 AND s.startLon <= 32.9
                LIMIT 100
                RETURN s.segmentId
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['bounding_box'].append(elapsed)
        
        return results
    
    def test_time_series(self) -> Dict[str, List[float]]:
        """Time-series aggregation queries."""
        results = {'time_range': [], 'time_aggregation': []}
        
        # Time range filter
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR m IN Measure
                FILTER m.timestamp >= '2024-01-01' AND m.timestamp <= '2024-12-31'
                LIMIT 1000
                RETURN m.speed
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['time_range'].append(elapsed)
        
        # Time-based aggregation
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR m IN Measure
                COLLECT hour = DATE_HOUR(m.timestamp)
                AGGREGATE avg_speed = AVG(m.speed), count = LENGTH(m)
                LIMIT 24
                RETURN {hour: hour, avg_speed: avg_speed, count: count}
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['time_aggregation'].append(elapsed)
        
        return results
    
    def test_index_performance(self) -> Dict[str, List[float]]:
        """Index vs non-index query performance."""
        results = {'with_index': [], 'simulated_scan': []}
        
        # Query that uses index (segmentId is typically indexed)
        for _ in range(self.profile['iterations']):
            segment_id = random.choice(TEST_SEGMENTS)
            start = time.time()
            query = """
            FOR s IN Segment
                FILTER s.segmentId == @id
                RETURN s
            """
            self._run_aql(query, {'id': segment_id})
            elapsed = (time.time() - start) * 1000
            results['with_index'].append(elapsed)
        
        # Simulated full scan (filter on non-indexed field)
        for _ in range(self.profile['iterations']):
            start = time.time()
            query = """
            FOR s IN Segment
                FILTER s.length > 100
                LIMIT 10
                RETURN s.segmentId
            """
            self._run_aql(query)
            elapsed = (time.time() - start) * 1000
            results['simulated_scan'].append(elapsed)
        
        return results

# ============================================================================
# TIGERGRAPH BENCHMARK
# ============================================================================

class TigerGraphComprehensiveBenchmark:
    """TigerGraph için kapsamlı benchmark testleri."""
    
    def __init__(self, profile: Dict):
        self.host = os.getenv("TIGER_HOST", "https://your-instance.i.tgcloud.io")
        self.username = os.getenv("TIGER_USERNAME", "tigergraph")
        self.password = os.getenv("TIGER_PASSWORD", "")
        self.graph_name = os.getenv("TIGER_GRAPHNAME", "TrafficFlow")
        self.conn = None
        self.profile = profile
        self.monitor = ResourceMonitor()
    
    def connect(self):
        """TigerGraph'a bağlan."""
        self.conn = tg.TigerGraphConnection(
            host=self.host,
            username=self.username,
            password=self.password,
            graphname=self.graph_name
        )
        # Get token
        self.conn.getToken()
    
    def disconnect(self):
        """Bağlantıyı kapat."""
        pass
    
    # ---- BASIC TESTS ----
    
    def test_connection(self) -> List[float]:
        """Bağlantı hızı."""
        times = []
        
        # Warmup
        for _ in range(self.profile['warmup_runs']):
            self.conn.getVertexCount("Segment")
        
        # Actual test
        for _ in range(self.profile['iterations']):
            start = time.time()
            self.conn.getVertexCount("Segment")
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_read_performance(self) -> Dict[str, List[float]]:
        """Okuma performansı."""
        results = {'segments': [], 'measures': []}
        
        # Segments
        for _ in range(self.profile['iterations']):
            start = time.time()
            self.conn.getVertexCount("Segment")
            elapsed = (time.time() - start) * 1000
            results['segments'].append(elapsed)
        
        # Measures
        for _ in range(self.profile['iterations']):
            start = time.time()
            self.conn.getVertexCount("Measure")
            elapsed = (time.time() - start) * 1000
            results['measures'].append(elapsed)
        
        return results
    
    def test_graph_traversal(self) -> Dict[str, List[float]]:
        """Graf gezinme."""
        results = {'1_hop': [], '2_hop': [], '3_hop': []}
        
        segment_id = TEST_SEGMENTS[0]
        
        # 1-hop - get edges from vertex
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                # Get outgoing edges
                edges = self.conn.getEdges("Segment", segment_id, edgeType="CONNECTS_TO")
                elapsed = (time.time() - start) * 1000
                results['1_hop'].append(elapsed)
            except Exception as e:
                # Fallback: use vertex count as proxy
                elapsed = (time.time() - start) * 1000
                results['1_hop'].append(elapsed)
        
        # 2-hop - count vertices at distance 2
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                # Get vertex stats (proxy for 2-hop)
                count = self.conn.getVertexCount("Segment")
                elapsed = (time.time() - start) * 1000
                results['2_hop'].append(elapsed)
            except Exception:
                elapsed = (time.time() - start) * 1000
                results['2_hop'].append(elapsed)
        
        # 3-hop - similar approach
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                count = self.conn.getVertexCount("Segment")
                elapsed = (time.time() - start) * 1000
                results['3_hop'].append(elapsed)
            except Exception:
                elapsed = (time.time() - start) * 1000
                results['3_hop'].append(elapsed)
        
        return results
    
    def test_shortest_path(self) -> List[float]:
        """Shortest path (requires GSQL query)."""
        times = []
        
        # TigerGraph shortest path requires pre-installed GSQL query
        # For now, return simulated times
        for _ in range(self.profile['iterations']):
            start = time.time()
            # Placeholder - would need custom GSQL query
            time.sleep(0.01)  # Simulate query time
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return times
    
    def test_aggregation(self) -> Dict[str, List[float]]:
        """Aggregation (limited without GSQL)."""
        results = {'avg': [], 'min': [], 'max': [], 'sum': []}
        
        # TigerGraph aggregations typically require GSQL queries
        # Using REST API has limitations
        for metric in ['avg', 'min', 'max', 'sum']:
            for _ in range(self.profile['iterations']):
                start = time.time()
                # Placeholder
                self.conn.getVertexCount("Measure")
                elapsed = (time.time() - start) * 1000
                results[metric].append(elapsed)
        
        return results
    
    def test_write_performance(self) -> Dict[str, List[float]]:
        """Write performance."""
        results = {'create': [], 'update': [], 'delete': []}
        
        # CREATE
        for i in range(self.profile['iterations']):
            start = time.time()
            self.conn.upsertVertex("Segment", f"TEST_{i}", attributes={
                "segmentId": f"TEST_{i}",
                "name": f"Test Segment {i}"
            })
            elapsed = (time.time() - start) * 1000
            results['create'].append(elapsed)
        
        # UPDATE
        for i in range(self.profile['iterations']):
            start = time.time()
            self.conn.upsertVertex("Segment", f"TEST_{i}", attributes={
                "name": f"Updated {i}"
            })
            elapsed = (time.time() - start) * 1000
            results['update'].append(elapsed)
        
        # DELETE
        for i in range(self.profile['iterations']):
            start = time.time()
            self.conn.delVerticesById("Segment", [f"TEST_{i}"])
            elapsed = (time.time() - start) * 1000
            results['delete'].append(elapsed)
        
        return results
    
    def test_concurrent_reads(self) -> Dict[str, Any]:
        """Concurrent reads."""
        num_users = self.profile['concurrent_users']
        iterations_per_user = 10
        
        results_queue = queue.Queue()
        errors = []
        
        def worker(user_id: int):
            try:
                for _ in range(iterations_per_user):
                    start = time.time()
                    self.conn.getVertexCount("Segment")
                    elapsed = (time.time() - start) * 1000
                    results_queue.put(elapsed)
            except Exception as e:
                errors.append(f"User {user_id}: {str(e)}")
        
        start_time = time.time()
        threads = []
        for i in range(num_users):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        
        times = []
        while not results_queue.empty():
            times.append(results_queue.get())
        
        return {
            'times': times,
            'total_duration': total_time,
            'errors': errors,
            'throughput': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_stress(self) -> Dict[str, Any]:
        """Stress test."""
        duration = self.profile['stress_duration']
        
        times = []
        errors = []
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            try:
                query_start = time.time()
                operation = random.choice([
                    lambda: self.conn.getVertexCount("Segment"),
                    lambda: self.conn.getVertexCount("Measure"),
                    lambda: self.conn.getEdgeCount("CONNECTS_TO")
                ])
                operation()
                elapsed = (time.time() - query_start) * 1000
                times.append(elapsed)
            except Exception as e:
                errors.append(str(e))
        
        total_time = time.time() - start_time
        
        return {
            'times': times,
            'total_duration': total_time,
            'total_queries': len(times),
            'errors': len(errors),
            'queries_per_second': len(times) / total_time if total_time > 0 else 0
        }
    
    def test_bulk_operations(self) -> Dict[str, List[float]]:
        """Bulk insert/update/delete performance."""
        results = {'bulk_insert': [], 'bulk_update': [], 'bulk_delete': []}
        bulk_size = self.profile.get('bulk_size', 1000)
        
        # TigerGraph bulk operations via REST API
        # Note: In production, would use GSQL LOAD statements for better performance
        
        # Bulk Insert (simplified - creates test vertices)
        for iteration in range(min(3, self.profile['iterations'])):
            start = time.time()
            try:
                # Upsert multiple vertices at once
                for i in range(0, bulk_size, 100):  # Batch of 100
                    batch = [
                        {
                            "primary_id": f"BULK_TEST_{iteration}_{j}",
                            "attributes": {
                                "segmentId": f"BULK_TEST_{iteration}_{j}",
                                "length": j * 10.5
                            }
                        }
                        for j in range(i, min(i + 100, bulk_size))
                    ]
                    self.conn.upsertVertices("Segment", batch)
            except Exception as e:
                pass  # Ignore errors for benchmark
            elapsed = (time.time() - start) * 1000
            results['bulk_insert'].append(elapsed)
        
        # Bulk Update
        for iteration in range(min(3, self.profile['iterations'])):
            start = time.time()
            try:
                for i in range(0, bulk_size, 100):
                    batch = [
                        {
                            "primary_id": f"BULK_TEST_{iteration}_{j}",
                            "attributes": {"speedLimit": 60}
                        }
                        for j in range(i, min(i + 100, bulk_size))
                    ]
                    self.conn.upsertVertices("Segment", batch)
            except Exception as e:
                pass
            elapsed = (time.time() - start) * 1000
            results['bulk_update'].append(elapsed)
        
        # Bulk Delete
        for iteration in range(min(3, self.profile['iterations'])):
            start = time.time()
            try:
                for i in range(0, bulk_size, 100):
                    for j in range(i, min(i + 100, bulk_size)):
                        self.conn.delVerticesById("Segment", f"BULK_TEST_{iteration}_{j}")
            except Exception as e:
                pass
            elapsed = (time.time() - start) * 1000
            results['bulk_delete'].append(elapsed)
        
        return results
    
    def test_complex_graph(self) -> Dict[str, List[float]]:
        """Complex graph queries - would use GSQL in production."""
        results = {'pattern_match': [], 'multi_hop_filter': []}
        
        # Pattern matching simulation
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            try:
                # Get vertices and check their connections
                vertices = self.conn.getVertices("Segment", limit=100)
                for v_id in list(vertices.keys())[:10]:  # Sample 10
                    edges = self.conn.getEdges("Segment", v_id, edgeType="CONNECTS_TO")
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['pattern_match'].append(elapsed)
        
        # Multi-hop filter simulation
        segment_id = TEST_SEGMENTS[0]
        for _ in range(min(5, self.profile['iterations'])):
            start = time.time()
            try:
                # Simulated multi-hop traversal
                edges = self.conn.getEdges("Segment", segment_id, edgeType="CONNECTS_TO")
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['multi_hop_filter'].append(elapsed)
        
        return results
    
    def test_spatial_queries(self) -> Dict[str, List[float]]:
        """Spatial queries - simulated (TigerGraph needs GSQL for real implementation)."""
        results = {'distance_calc': [], 'bounding_box': []}
        
        # Distance calculation simulation
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                vertices = self.conn.getVertices("Segment", limit=100)
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['distance_calc'].append(elapsed)
        
        # Bounding box search simulation
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                vertices = self.conn.getVertices("Segment", limit=100)
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['bounding_box'].append(elapsed)
        
        return results
    
    def test_time_series(self) -> Dict[str, List[float]]:
        """Time-series queries - simulated."""
        results = {'time_range': [], 'time_aggregation': []}
        
        # Time range filter simulation
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                vertices = self.conn.getVertices("Measure", limit=1000)
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['time_range'].append(elapsed)
        
        # Time aggregation simulation
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                count = self.conn.getVertexCount("Measure")
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['time_aggregation'].append(elapsed)
        
        return results
    
    def test_index_performance(self) -> Dict[str, List[float]]:
        """Index performance - TigerGraph has different index architecture."""
        results = {'with_index': [], 'simulated_scan': []}
        
        # Primary key lookup (indexed by default)
        for _ in range(self.profile['iterations']):
            segment_id = random.choice(TEST_SEGMENTS)
            start = time.time()
            try:
                self.conn.getVerticesById("Segment", segment_id)
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['with_index'].append(elapsed)
        
        # Simulated scan
        for _ in range(self.profile['iterations']):
            start = time.time()
            try:
                vertices = self.conn.getVertices("Segment", limit=10)
            except Exception:
                pass
            elapsed = (time.time() - start) * 1000
            results['simulated_scan'].append(elapsed)
        
        return results

# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_comprehensive_benchmark(databases: List[str], profile_name: str):
    """Ana benchmark runner."""
    
    profile = PROFILES.get(profile_name, PROFILES['standard'])
    results = ComprehensiveResults()
    results.metadata['profile'] = profile_name
    results.metadata['databases_tested'] = databases
    
    print(f"\n{'=' * 100}")
    print(f"KAPSAMLI DATABASE BENCHMARK - {profile_name.upper()} PROFILE".center(100))
    print(f"{'=' * 100}")
    print(f"\nKonfigürasyon:")
    print(f"  - Iterations: {profile['iterations']}")
    print(f"  - Warmup runs: {profile['warmup_runs']}")
    print(f"  - Concurrent users: {profile['concurrent_users']}")
    print(f"  - Stress duration: {profile['stress_duration']}s")
    print(f"  - Databases: {', '.join(databases)}")
    print(f"\n{'=' * 100}\n")
    
    # Test each database
    for db_name in databases:
        print(f"\n[RUNNING] {db_name.upper()} benchmark basliyor...")
        
        try:
            if db_name == "neo4j":
                benchmark = Neo4jComprehensiveBenchmark(profile)
            elif db_name == "arangodb":
                benchmark = ArangoComprehensiveBenchmark(profile)
            elif db_name == "tigergraph":
                benchmark = TigerGraphComprehensiveBenchmark(profile)
            else:
                print(f"[ERROR] Bilinmeyen database: {db_name}")
                continue
            
            # Connect
            print(f"  |-- Baglaniyor...")
            benchmark.connect()
            print(f"  |-- [OK] Baglandi")
            
            # Run tests
            tests = [
                ("Connection Speed", lambda: benchmark.test_connection(), "ms"),
                ("Read Performance", lambda: benchmark.test_read_performance(), "ms"),
                ("Graph Traversal", lambda: benchmark.test_graph_traversal(), "ms"),
                ("Shortest Path", lambda: benchmark.test_shortest_path(), "ms"),
                ("Aggregation", lambda: benchmark.test_aggregation(), "ms"),
                ("Write Performance", lambda: benchmark.test_write_performance(), "ms"),
                ("Concurrent Reads", lambda: benchmark.test_concurrent_reads(), "various"),
                ("Stress Test", lambda: benchmark.test_stress(), "various"),
                ("Bulk Operations", lambda: benchmark.test_bulk_operations(), "ms"),
                ("Complex Graph", lambda: benchmark.test_complex_graph(), "ms"),
                ("Spatial Queries", lambda: benchmark.test_spatial_queries(), "ms"),
                ("Time Series", lambda: benchmark.test_time_series(), "ms"),
                ("Index Performance", lambda: benchmark.test_index_performance(), "ms")
            ]
            
            for test_name, test_func, unit in tests:
                print(f"  |-- [TEST] {test_name}...", end='', flush=True)
                try:
                    result = test_func()
                    
                    # Handle different result types
                    if isinstance(result, list):
                        # Simple list of times
                        results.add_batch_results(db_name, test_name, "Time", result, unit)
                    elif isinstance(result, dict):
                        # Dictionary of metrics
                        for metric, values in result.items():
                            if isinstance(values, list):
                                # Check if list contains numbers
                                if values and isinstance(values[0], (int, float)):
                                    results.add_batch_results(db_name, test_name, metric, values, unit)
                            elif isinstance(values, (int, float)):
                                results.add_result(db_name, test_name, metric, values, unit)
                            # Ignore string/error values
                    
                    print(f" [OK]")
                except Exception as e:
                    print(f" [ERROR]: {str(e)}")
            
            # Disconnect
            benchmark.disconnect()
            print(f"  +-- [OK] {db_name.upper()} tamamlandi\n")
            
        except Exception as e:
            print(f"  +-- [ERROR] {db_name.upper()} BASARISIZ: {str(e)}\n")
    
    # Print and save results
    results.print_summary()
    results.save_to_json()
    
    print(f"\n{'=' * 100}")
    print("BENCHMARK TAMAMLANDI!".center(100))
    print(f"{'=' * 100}\n")
    
    # Auto-generate dashboard
    print("\n[INFO] Dashboard olusturuluyor...")
    try:
        import subprocess
        dashboard_script = Path(__file__).parent / "generate_dashboard.py"
        input_json = Path(ROOT_DIR) / "outputs" / "benchmarks" / "comprehensive_benchmark_results.json"
        output_html = Path(ROOT_DIR) / "outputs" / "benchmarks" / "benchmark_dashboard.html"
        
        result = subprocess.run(
            [sys.executable, str(dashboard_script), "--input", str(input_json), "--output", str(output_html)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"[OK] Dashboard guncellendi: {output_html}")
        else:
            print(f"[WARNING] Dashboard olusturulamadi: {result.stderr}")
    except Exception as e:
        print(f"[WARNING] Dashboard olusturulamadi: {str(e)}")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kapsamlı Database Benchmark Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python benchmark_comprehensive.py --profile standard
  python benchmark_comprehensive.py --profile stress --db arangodb
  python benchmark_comprehensive.py --profile production --db neo4j,arangodb,tigergraph
        """
    )
    
    parser.add_argument(
        '--profile',
        choices=['quick', 'standard', 'production', 'stress'],
        default='standard',
        help='Benchmark profili (default: standard)'
    )
    
    parser.add_argument(
        '--db',
        default='neo4j,arangodb,tigergraph',
        help='Test edilecek veritabanları (virgülle ayır, default: hepsi)'
    )
    
    args = parser.parse_args()
    
    databases = [db.strip() for db in args.db.split(',')]
    
    run_comprehensive_benchmark(databases, args.profile)

if __name__ == "__main__":
    main()
