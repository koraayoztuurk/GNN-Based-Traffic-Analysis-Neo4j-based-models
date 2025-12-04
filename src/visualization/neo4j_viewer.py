#!/usr/bin/env python3
"""
neo4j_viewer.py
---------------
Neo4j trafik haritası web server
"""
import os
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from neo4j import GraphDatabase
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

app = Flask(__name__, static_folder='.')

# Neo4j Bağlantı
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def get_driver():
    """Neo4j driver oluştur"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@app.route('/')
def index():
    return send_from_directory('.', 'neo4j_traffic_map.html')

@app.route('/api/timestamps')
def get_timestamps():
    """Tüm mevcut timestamp'leri döndür"""
    driver = None
    try:
        driver = get_driver()
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (m:Measure)
                RETURN DISTINCT m.timestamp AS ts
                ORDER BY ts DESC
            """)
            timestamps = [record["ts"] for record in result]
        
        return jsonify({"timestamps": timestamps})
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "timestamps": []}), 500
    finally:
        if driver:
            driver.close()

@app.route('/api/traffic')
def get_traffic():
    """Belirli bir timestamp için trafik verilerini döndür"""
    driver = None
    try:
        # URL'den timestamp parametresini al
        selected_ts = request.args.get('timestamp', None)
        print(f"[DEBUG] Requested timestamp: {selected_ts}")
        
        driver = get_driver()
        
        with driver.session(database=NEO4J_DATABASE) as session:
            # Eğer timestamp seçilmemişse en son timestamp'i al
            if not selected_ts:
                print("[DEBUG] No timestamp provided, getting latest...")
                result = session.run("""
                    MATCH (m:Measure)
                    RETURN m.timestamp AS ts
                    ORDER BY ts DESC
                    LIMIT 1
                """)
                latest_ts_record = result.single()
                
                if not latest_ts_record:
                    return jsonify({"error": "No data", "features": [], "timestamp": None})
                
                selected_ts = latest_ts_record["ts"]
            
            latest_ts = selected_ts
            print(f"[DEBUG] Using timestamp: {latest_ts}")
            
            # Verileri çek - SADECE o timestamp'te measure'i olan segmentleri getir
            result = session.run("""
                MATCH (s:Segment)
                MATCH (m:Measure {segmentId: s.segmentId, timestamp: $ts})
                RETURN 
                    s.segmentId AS segmentId,
                    s.geom AS geometry,
                    m.jamFactor AS jamFactor,
                    m.speed AS speed,
                    m.freeFlow AS freeFlow,
                    m.confidence AS confidence
            """, ts=latest_ts)
            
            print(f"[DEBUG] Query executed for timestamp: {latest_ts}")
            
            features = []
            record_count = 0
            for record in result:
                record_count += 1
                geom = record["geometry"]
                if not geom:
                    continue
                
                try:
                    # WKT'den GeoJSON'a çevir
                    # Hem "LINESTRING(" hem "LINESTRING (" formatını destekle
                    coords_str = geom.replace("LINESTRING (", "").replace("LINESTRING(", "").replace(")", "").strip()
                    if not coords_str or coords_str.startswith("LINESTRING"):
                        continue
                    coord_pairs = coords_str.split(", ")
                    coordinates = []
                    for pair in coord_pairs:
                        parts = pair.strip().split()
                        if len(parts) < 2:
                            continue
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coordinates.append([lon, lat])
                    
                    if len(coordinates) < 2:
                        continue
                    
                    # Renk belirle - jamFactor bazlı
                    jam_factor = record["jamFactor"]
                    if jam_factor is None:
                        color = "#9e9e9e"  # Gri - veri yok
                    elif jam_factor == 0 or jam_factor < 0.5:
                        color = "#1b5e20"  # Koyu yeşil - mükemmel akış
                    else:
                        jf = float(jam_factor)
                        if jf >= 9.5:
                            color = "#000000"  # Siyah - durmuş
                        elif jf >= 7.5:
                            color = "#b71c1c"  # Koyu kırmızı
                        elif jf >= 5.0:
                            color = "#f57c00"  # Turuncu
                        elif jf >= 2.5:
                            color = "#fbc02d"  # Sarı
                        else:
                            color = "#2e7d32"  # Yeşil
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coordinates
                        },
                        "properties": {
                            "segmentId": record["segmentId"],
                            "jamFactor": jam_factor,
                            "speed": record["speed"],
                            "freeFlow": record["freeFlow"],
                            "confidence": record["confidence"],
                            "color": color
                        }
                    }
                    features.append(feature)
                except Exception as e:
                    print(f"Error processing segment: {e}")
                    continue
        
        print(f"[DEBUG] Processed {record_count} records, created {len(features)} features")
        
        return jsonify({
            "type": "FeatureCollection",
            "features": features,
            "timestamp": latest_ts
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "features": [], "timestamp": None}), 500
    finally:
        if driver:
            driver.close()

if __name__ == '__main__':
    print("="*70)
    print("  NEO4J CANLI TRAFIK HARITASI")
    print("="*70)
    print()
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Database: {NEO4J_DATABASE}")
    print()
    print("Tarayıcıda açın: http://localhost:5000")
    print()
    print("Durdurmak için: Ctrl+C")
    print("="*70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
