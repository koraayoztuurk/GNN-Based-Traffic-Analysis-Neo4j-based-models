#!/usr/bin/env python3
"""
reset_tigergraph_schema.py - TigerGraph Schema Sıfırlama
---------------------------------------------------------
Mevcut TrafficGraph'ı silip yeniden oluşturur.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pyTigerGraph as tg

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / "config" / ".env"
load_dotenv(ENV_PATH)

print("\n" + "=" * 80)
print("🟠 TIGERGRAPH SCHEMA SIFIRLA VE YENİDEN OLUŞTUR")
print("=" * 80)
print()

# TigerGraph bağlantı bilgileri
host = os.getenv("TIGER_HOST", "http://127.0.0.1")
rest_port = os.getenv("TIGER_REST_PORT", "9000")
gsql_port = os.getenv("TIGER_GSQL_PORT", "14240")
username = os.getenv("TIGER_USERNAME", "tigergraph")
password = os.getenv("TIGER_PASSWORD", "tigergraph")
graphname = os.getenv("TIGER_GRAPHNAME", "TrafficGraph")

print(f"ℹ️  Host: {host}:{rest_port}")
print(f"ℹ️  Graph: {graphname}")
print()

# Onay al
response = input("⚠️  Mevcut graph silinecek! Devam edilsin mi? (EVET yazın): ")
if response != "EVET":
    print("\n❌ İşlem iptal edildi")
    sys.exit(0)

print()

try:
    # Bağlan
    print("🔗 TigerGraph'a bağlanılıyor...")
    conn = tg.TigerGraphConnection(
        host=host,
        restppPort=rest_port,
        username=username,
        password=password,
        graphname=graphname
    )
    print("✅ Bağlantı başarılı!")
    print()
    
    # 1. Mevcut graph'ı sil
    print("=" * 80)
    print("🗑️  MEVCUT GRAPH SİLİNİYOR")
    print("=" * 80)
    
    try:
        # GSQL komutu ile drop
        drop_query = f"DROP GRAPH {graphname}"
        print(f"ℹ️  Komut: {drop_query}")
        
        # pyTigerGraph ile GSQL çalıştır
        result = conn.gsql(drop_query)
        print(f"✅ Graph silindi!")
        print(f"   Sonuç: {result}")
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "not found" in error_msg:
            print(f"ℹ️  Graph zaten yok, devam ediliyor...")
        else:
            print(f"⚠️  Silme hatası: {e}")
            print(f"   (Normal olabilir, devam ediliyor...)")
    
    print()
    
    # 2. Yeni schema oluştur
    print("=" * 80)
    print("🔧 YENİ SCHEMA OLUŞTURULUYOR")
    print("=" * 80)
    
    # GSQL schema definition
    schema_gsql = f"""
CREATE GRAPH {graphname}()

USE GRAPH {graphname}

CREATE SCHEMA_CHANGE JOB traffic_schema FOR GRAPH {graphname} {{
    
    // Vertex Types
    ADD VERTEX Segment (
        PRIMARY_ID segmentId STRING,
        hereSegmentId STRING,
        osmWayId STRING,
        frc INT,
        lengthM DOUBLE,
        name STRING,
        geom STRING,
        lat DOUBLE,
        lon DOUBLE,
        startLat DOUBLE,
        startLon DOUBLE,
        endLat DOUBLE,
        endLon DOUBLE
    ) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true";
    
    ADD VERTEX Measure (
        PRIMARY_ID measureId STRING,
        segmentId STRING,
        timestamp DATETIME,
        jamFactor DOUBLE,
        speed DOUBLE,
        freeFlow DOUBLE,
        confidence DOUBLE
    ) WITH PRIMARY_ID_AS_ATTRIBUTE="true";
    
    // Edge Types
    ADD UNDIRECTED EDGE CONNECTS_TO (
        FROM Segment,
        TO Segment,
        distance DOUBLE
    );
    
    ADD DIRECTED EDGE AT_TIME (
        FROM Segment,
        TO Measure
    );
}}

RUN SCHEMA_CHANGE JOB traffic_schema
DROP JOB traffic_schema
"""
    
    print("📝 GSQL Schema:")
    print(schema_gsql)
    print()
    
    print("🔧 Schema çalıştırılıyor...")
    result = conn.gsql(schema_gsql)
    
    print("✅ Schema oluşturuldu!")
    print()
    print("📋 Sonuç:")
    print(result)
    print()
    
    # 3. Doğrulama
    print("=" * 80)
    print("✅ DOĞRULAMA")
    print("=" * 80)
    
    try:
        # Graph bilgilerini al
        schema = conn.getSchema()
        print("📊 Graph Schema:")
        print(f"   Vertex Types: {list(schema.get('VertexTypes', {}).keys())}")
        print(f"   Edge Types: {list(schema.get('EdgeTypes', {}).keys())}")
        print()
    except Exception as e:
        print(f"ℹ️  Schema doğrulama atlandı: {e}")
    
    print("=" * 80)
    print("🎉 TİGERGRAPH SCHEMA HAZIR!")
    print("=" * 80)
    print()
    print("🚀 Şimdi pipeline'ı çalıştırabilirsiniz:")
    print("   python run_pipeline.py")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print("❌ HATA!")
    print("=" * 80)
    print(f"⚠️  {e}")
    print()
    print("💡 Olası çözümler:")
    print("   1. TigerGraph container'ını yeniden başlatın:")
    print("      docker restart tigergraph")
    print()
    print("   2. TigerGraph'ı devre dışı bırakın:")
    print("      .env dosyasında ACTIVE_DATABASES=neo4j,arangodb")
    print()
    sys.exit(1)
