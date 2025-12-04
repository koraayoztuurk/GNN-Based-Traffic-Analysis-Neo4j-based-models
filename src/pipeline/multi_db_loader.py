#!/usr/bin/env python3
"""
multi_db_loader.py
------------------
Çoklu veritabanı yükleyici - Aynı anda 3 DB'ye veri yükler

KULLANIM:
    python src/pipeline/multi_db_loader.py --init-schema
    python src/pipeline/multi_db_loader.py --load-segments data/edges_static.geojson
    python src/pipeline/multi_db_loader.py --load-measure archive/flow_20251027_1244.geojson
    python src/pipeline/multi_db_loader.py --build-topology

Veya direkt import:
    from src.pipeline.multi_db_loader import MultiDBLoader
    
    loader = MultiDBLoader()
    loader.load_all("data/processed/edges_static.geojson", "archive/flow_20251027_1244.geojson")
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# .env yükle
ENV_PATH = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)

# Hangi DB'ler aktif?
ACTIVE_DBS = os.getenv("ACTIVE_DATABASES", "neo4j").split(",")
ACTIVE_DBS = [db.strip().lower() for db in ACTIVE_DBS]


class MultiDBLoader:
    """
    Çoklu veritabanı yükleyici
    
    Mevcut kodunuzu BOZMADAN 3 DB'ye birden veri yükler:
    - Neo4j (mevcut)
    - ArangoDB (yeni)
    - TigerGraph (yeni)
    """
    
    def __init__(self):
        """Aktif DB'lere bağlan"""
        self.loaders = {}
        
        print("=" * 70)
        print("🚀 MULTI-DB LOADER")
        print("=" * 70)
        print(f"ℹ️  Aktif veritabanları: {', '.join(ACTIVE_DBS)}")
        print()
        
        # Neo4j
        if "neo4j" in ACTIVE_DBS:
            try:
                from src.neo4j.neo4j_loader import cmd_init_schema, cmd_load_segments, cmd_load_measure, build_connects_to
                self.loaders["neo4j"] = {
                    "init": cmd_init_schema,
                    "segments": cmd_load_segments,
                    "measure": cmd_load_measure,
                    "topology": build_connects_to,
                    "name": "Neo4j"
                }
                print("✅ Neo4j loader hazır")
            except Exception as e:
                print(f"⚠️  Neo4j loader yüklenemedi: {e}")
        
        # ArangoDB
        if "arangodb" in ACTIVE_DBS:
            try:
                from src.arangodb.arango_loader import ArangoLoader
                arango = ArangoLoader()
                self.loaders["arangodb"] = {
                    "instance": arango,
                    "init": arango.init_schema,
                    "segments": arango.load_segments,
                    "measure": arango.load_measurements,
                    "topology": arango.build_connects_to,
                    "name": "ArangoDB"
                }
                print("✅ ArangoDB loader hazır")
            except Exception as e:
                print(f"⚠️  ArangoDB loader yüklenemedi: {e}")
                print(f"   Kurulum: pip install python-arango")
        
        # TigerGraph
        if "tigergraph" in ACTIVE_DBS:
            try:
                from src.tigergraph.tigergraph_loader import TigerGraphLoader
                tiger = TigerGraphLoader()
                self.loaders["tigergraph"] = {
                    "instance": tiger,
                    "init": tiger.init_schema,
                    "segments": tiger.load_segments,
                    "measure": tiger.load_measurements,
                    "topology": tiger.build_connects_to,
                    "name": "TigerGraph"
                }
                print("✅ TigerGraph loader hazır")
            except Exception as e:
                print(f"⚠️  TigerGraph loader yüklenemedi: {e}")
                print(f"   Kurulum: pip install pyTigerGraph")
        
        print()
        
        if not self.loaders:
            print("❌ Hiçbir veritabanı yüklenemedi!")
            print("ℹ️  config/.env dosyasındaki ACTIVE_DATABASES ayarını kontrol edin")
            sys.exit(1)
    
    def init_all_schemas(self):
        """Tüm veritabanlarında schema oluştur"""
        print("\n" + "=" * 70)
        print("🔧 TÜM VERİTABANLARINDA SCHEMA OLUŞTURULUYOR")
        print("=" * 70 + "\n")
        
        for db_name, loader in self.loaders.items():
            print(f"▶️  {loader['name']} schema oluşturuluyor...")
            try:
                if "init" in loader:
                    loader["init"]()
                print(f"✅ {loader['name']} schema hazır!\n")
            except Exception as e:
                print(f"❌ {loader['name']} hata: {e}\n")
    
    def load_segments_all(self, geojson_path):
        """Tüm veritabanlarına segment yükle"""
        print("\n" + "=" * 70)
        print(f"📦 TÜM VERİTABANLARINA SEGMENT YÜKLENİYOR")
        print(f"📂 Dosya: {geojson_path}")
        print("=" * 70 + "\n")
        
        for db_name, loader in self.loaders.items():
            print(f"▶️  {loader['name']} yükleniyor...")
            try:
                if "segments" in loader:
                    loader["segments"](geojson_path)
                print(f"✅ {loader['name']} segment yükleme tamam!\n")
            except Exception as e:
                print(f"❌ {loader['name']} hata: {e}\n")
    
    def load_measurements_all(self, geojson_path, timestamp=None):
        """Tüm veritabanlarına measure yükle"""
        print("\n" + "=" * 70)
        print(f"📊 TÜM VERİTABANLARINA MEASURE YÜKLENİYOR")
        print(f"📂 Dosya: {geojson_path}")
        print("=" * 70 + "\n")
        
        for db_name, loader in self.loaders.items():
            print(f"▶️  {loader['name']} yükleniyor...")
            try:
                if "measure" in loader:
                    loader["measure"](geojson_path, timestamp)
                print(f"✅ {loader['name']} measure yükleme tamam!\n")
            except Exception as e:
                print(f"❌ {loader['name']} hata: {e}\n")
    
    def build_topology_all(self, threshold=12.0):
        """Tüm veritabanlarında CONNECTS_TO ilişkileri oluştur"""
        print("\n" + "=" * 70)
        print(f"🔗 TÜM VERİTABANLARINDA TOPOLOJİ OLUŞTURULUYOR")
        print(f"⚙️  Threshold: {threshold}m")
        print("=" * 70 + "\n")
        
        for db_name, loader in self.loaders.items():
            print(f"▶️  {loader['name']} topoloji oluşturuluyor...")
            try:
                if "topology" in loader:
                    loader["topology"](threshold)
                else:
                    print(f"⚠️  {loader['name']} için topoloji metodu tanımlı değil")
                print(f"✅ {loader['name']} topoloji tamam!\n")
            except Exception as e:
                print(f"❌ {loader['name']} hata: {e}\n")
    
    def close_all(self):
        """Tüm bağlantıları kapat"""
        for db_name, loader in self.loaders.items():
            try:
                if "instance" in loader and hasattr(loader["instance"], "close"):
                    loader["instance"].close()
            except:
                pass


# CLI
def main():
    parser = argparse.ArgumentParser(description="Multi-Database Loader")
    
    parser.add_argument("--init-schema", action="store_true", help="Tüm DB'lerde schema oluştur")
    parser.add_argument("--load-segments", type=str, help="Segment GeoJSON yükle")
    parser.add_argument("--load-measure", type=str, help="Measure GeoJSON yükle")
    parser.add_argument("--timestamp", type=str, help="Measure timestamp (ISO format)")
    parser.add_argument("--build-topology", action="store_true", help="CONNECTS_TO ilişkileri oluştur")
    parser.add_argument("--threshold", type=float, default=12.0, help="Topoloji mesafe eşiği (metre)")
    parser.add_argument("--all", action="store_true", help="Tüm işlemleri yap (schema + segments + measure + topology)")
    
    args = parser.parse_args()
    
    # Loader oluştur
    loader = MultiDBLoader()
    
    try:
        # İşlemleri yap
        if args.init_schema or args.all:
            loader.init_all_schemas()
        
        if args.load_segments or args.all:
            segments_file = args.load_segments or "data/processed/edges_static.geojson"
            if Path(segments_file).exists():
                loader.load_segments_all(segments_file)
            else:
                print(f"⚠️  Dosya bulunamadı: {segments_file}")
        
        if args.load_measure or args.all:
            measure_file = args.load_measure
            # --all için en son arşiv dosyasını kullan
            if not measure_file and args.all:
                archive_dir = Path("archive")
                if archive_dir.exists():
                    flow_files = sorted(archive_dir.glob("flow_*.geojson"))
                    if flow_files:
                        measure_file = str(flow_files[-1])  # En son dosya
            
            if measure_file and Path(measure_file).exists():
                loader.load_measurements_all(measure_file, args.timestamp)
            else:
                print(f"⚠️  Dosya bulunamadı veya belirtilmedi: {measure_file}")
        
        if args.build_topology or args.all:
            loader.build_topology_all(args.threshold)
        
        if not any([args.init_schema, args.load_segments, args.load_measure, args.build_topology, args.all]):
            parser.print_help()
    
    finally:
        loader.close_all()
    
    print("\n" + "=" * 70)
    print("✅ İŞLEMLER TAMAMLANDI!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
