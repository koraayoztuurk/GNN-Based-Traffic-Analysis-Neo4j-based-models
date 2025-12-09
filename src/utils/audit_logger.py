#!/usr/bin/env python3
"""
audit_logger.py
---------------
Neo4j Audit Logger - DR-26 & DR-27 Gereksinimleri

Gereksinimler:
- DR-26: segmentCount, measureCount, connectsToCount, atTimeCount
- DR-27: isolatedSegmentCount (izole segment tespiti)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from neo4j import GraphDatabase


class AuditLogger:
    """
    Neo4j veri yükleme audit'i için logger.
    
    DR-26: Temel sayılar (segment, measure, ilişkiler)
    DR-27: İzole segment tespiti
    """
    
    def __init__(self, log_dir: str = "outputs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_audit = {}
    
    def start_audit(self, source_file: str):
        """Yeni bir audit oturumu başlat"""
        self.current_audit = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_file": source_file,
                "database": "Neo4j"
            },
            "counts": {
                "segmentCount": 0,
                "measureCount": 0,
                "connectsToCount": 0,
                "atTimeCount": 0,
                "isolatedSegmentCount": 0  # DR-27
            },
            "details": {},
            "warnings": []
        }
    
    def query_neo4j_stats(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        neo4j_database: str = "neo4j"
    ):
        """
        Neo4j'den gerçek sayıları sorgula ve audit'e kaydet.
        
        DR-26: Tüm temel sayıları çek
        DR-27: İzole segment'leri tespit et
        """
        driver = None
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            
            with driver.session(database=neo4j_database) as session:
                # DR-26.1: Segment sayısı
                result = session.run("MATCH (s:Segment) RETURN count(s) AS cnt")
                self.current_audit["counts"]["segmentCount"] = result.single()["cnt"]
                
                # DR-26.2: Measure sayısı
                result = session.run("""
                    MATCH (m:Measure) 
                    RETURN count(m) AS cnt
                """)
                self.current_audit["counts"]["measureCount"] = result.single()["cnt"]
                
                # DR-26.3: CONNECTS_TO ilişki sayısı
                result = session.run("""
                    MATCH ()-[r:CONNECTS_TO]->() 
                    RETURN count(r) AS cnt
                """)
                self.current_audit["counts"]["connectsToCount"] = result.single()["cnt"]
                
                # DR-26.4: AT_TIME (atTime) ilişki sayısı
                result = session.run("""
                    MATCH ()-[r:AT_TIME]->() 
                    RETURN count(r) AS cnt
                """)
                self.current_audit["counts"]["atTimeCount"] = result.single()["cnt"]
                
                # DR-27: İzole segment sayısı (hiç bağlantısı olmayan)
                result = session.run("""
                    MATCH (s:Segment)
                    WHERE NOT (s)-[:CONNECTS_TO]-()
                    RETURN count(s) AS cnt
                """)
                isolated_count = result.single()["cnt"]
                self.current_audit["counts"]["isolatedSegmentCount"] = isolated_count
                
                # İzole segment'lerin detaylarını topla
                if isolated_count > 0:
                    result = session.run("""
                        MATCH (s:Segment)
                        WHERE NOT (s)-[:CONNECTS_TO]-()
                        RETURN s.segmentId AS segmentId
                        LIMIT 100
                    """)
                    
                    isolated_segments = [record["segmentId"] for record in result]
                    self.current_audit["details"]["isolated_segments"] = {
                        "count": isolated_count,
                        "sample": isolated_segments[:20],  # İlk 20 örnek
                        "note": "İzole segment'ler: Hiçbir CONNECTS_TO ilişkisi olmayan segment'ler"
                    }
                    
                    # Uyarı ekle
                    if isolated_count > 0:
                        self.current_audit["warnings"].append({
                            "code": "ISOLATED_SEGMENTS",
                            "message": f"{isolated_count} adet izole segment tespit edildi",
                            "severity": "WARNING" if isolated_count < 50 else "ERROR"
                        })
                
                # Ek detaylar: Graf yoğunluğu
                total_segments = self.current_audit["counts"]["segmentCount"]
                total_edges = self.current_audit["counts"]["connectsToCount"]
                
                if total_segments > 0:
                    # Ortalama derece (kaç komşusu var)
                    result = session.run("""
                        MATCH (s:Segment)
                        OPTIONAL MATCH (s)-[r:CONNECTS_TO]-()
                        WITH s, count(r) AS degree
                        RETURN avg(degree) AS avgDegree
                    """)
                    avg_degree = result.single()["avgDegree"] or 0
                    
                    # Graf yoğunluğu (0-1 arası)
                    max_edges = total_segments * (total_segments - 1)
                    density = total_edges / max_edges if max_edges > 0 else 0
                    
                    self.current_audit["details"]["topology"] = {
                        "avgDegree": round(avg_degree, 2),
                        "density": round(density, 6),
                        "maxPossibleEdges": max_edges,
                        "utilizationPercent": round((total_edges / max_edges) * 100, 4) if max_edges > 0 else 0
                    }
                
                # Measure dağılımı (AT_TIME ilişkisi üzerinden)
                result = session.run("""
                    MATCH (s:Segment)
                    OPTIONAL MATCH (m:Measure)-[:AT_TIME]->(s)
                    WITH s, count(m) AS measureCount
                    RETURN 
                        avg(measureCount) AS avgMeasures,
                        min(measureCount) AS minMeasures,
                        max(measureCount) AS maxMeasures
                """)
                record = result.single()
                
                self.current_audit["details"]["measures"] = {
                    "avgPerSegment": round(record["avgMeasures"] or 0, 2),
                    "minPerSegment": record["minMeasures"] or 0,
                    "maxPerSegment": record["maxMeasures"] or 0
                }
                
        finally:
            if driver:
                driver.close()
    
    def log_custom(self, key: str, value: Any):
        """Özel bir değer ekle"""
        if "custom" not in self.current_audit["details"]:
            self.current_audit["details"]["custom"] = {}
        self.current_audit["details"]["custom"][key] = value
    
    def add_warning(self, code: str, message: str, severity: str = "WARNING"):
        """Uyarı ekle"""
        self.current_audit["warnings"].append({
            "code": code,
            "message": message,
            "severity": severity
        })
    
    def finalize(self) -> Path:
        """Audit'i dosyaya kaydet ve döndür"""
        filename = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_audit, f, indent=2, ensure_ascii=False)
        
        # Konsol çıktısı (DR-26 & DR-27)
        self._print_audit_summary()
        
        return filepath
    
    def _print_audit_summary(self):
        """Audit özetini konsola yazdır"""
        print("\n" + "="*80)
        print("📊 NEO4J AUDIT RAPORU (DR-26 & DR-27)")
        print("="*80)
        print(f"⏰ Timestamp: {self.current_audit['metadata']['timestamp']}")
        print(f"📁 Kaynak: {self.current_audit['metadata'].get('source_file', 'N/A')}")
        print()
        
        # DR-26: Temel sayılar
        print("📈 TEMEL İSTATİSTİKLER (DR-26):")
        counts = self.current_audit['counts']
        print(f"   • segmentCount       : {counts['segmentCount']:,}")
        print(f"   • measureCount       : {counts['measureCount']:,}")
        print(f"   • connectsToCount    : {counts['connectsToCount']:,}")
        print(f"   • atTimeCount        : {counts['atTimeCount']:,}")
        
        # DR-27: İzole segment sayısı
        print(f"\n🔍 İZOLE SEGMENT ANALİZİ (DR-27):")
        print(f"   • isolatedSegmentCount: {counts['isolatedSegmentCount']:,}")
        
        if counts['isolatedSegmentCount'] > 0:
            print(f"   ⚠️  {counts['isolatedSegmentCount']} segment hiçbir komşuya bağlı değil!")
            if "isolated_segments" in self.current_audit["details"]:
                sample = self.current_audit["details"]["isolated_segments"]["sample"]
                print(f"   Örnek izole segment'ler: {', '.join(map(str, sample[:5]))}")
        else:
            print(f"   ✅ Tüm segment'ler bağlantılı")
        
        # Uyarılar
        if self.current_audit["warnings"]:
            print(f"\n⚠️  UYARILAR ({len(self.current_audit['warnings'])} adet):")
            for warning in self.current_audit["warnings"]:
                print(f"   [{warning['severity']}] {warning['code']}: {warning['message']}")
        
        # Topology detayları
        if "topology" in self.current_audit["details"]:
            topo = self.current_audit["details"]["topology"]
            print(f"\n🗺️  GRAF TOPOLOJİSİ:")
            print(f"   • Ortalama Derece    : {topo['avgDegree']}")
            print(f"   • Graf Yoğunluğu     : {topo['density']:.6f}")
            print(f"   • Kullanım Oranı     : {topo['utilizationPercent']:.4f}%")
        
        print("\n" + "="*80 + "\n")


# Standalone kullanım için
def run_audit(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str,
    neo4j_database: str = "neo4j",
    source_file: str = "manual_audit"
) -> Path:
    """
    Neo4j audit'ini çalıştır ve rapor oluştur.
    
    Usage:
        from src.utils.audit_logger import run_audit
        run_audit(
            "bolt://localhost:7687",
            "neo4j",
            "12345678"
        )
    """
    audit = AuditLogger()
    audit.start_audit(source_file)
    audit.query_neo4j_stats(neo4j_uri, neo4j_user, neo4j_pass, neo4j_database)
    return audit.finalize()


if __name__ == "__main__":
    # Doğrudan çalıştırma
    import os
    from dotenv import load_dotenv
    
    load_dotenv("config/.env")
    
    run_audit(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pass=os.getenv("NEO4J_PASS", "12345678"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        source_file="standalone_audit"
    )
