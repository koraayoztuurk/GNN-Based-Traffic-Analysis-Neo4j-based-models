#!/usr/bin/env python3
"""
13_arangodb_web_server.py
--------------------------
ArangoDB trafik haritası web server
"""
import os
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from arango import ArangoClient
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

app = Flask(__name__, static_folder='.')

# ArangoDB Bağlantı
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://127.0.0.1:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "1234")
ARANGO_DATABASE = os.getenv("ARANGO_DATABASE", "traffic_db")

def get_db():
    """ArangoDB bağlantısı kur"""
    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(ARANGO_DATABASE, username=ARANGO_USER, password=ARANGO_PASS)
    return client, db

@app.route('/')
def index():
    return send_from_directory('.', 'arangodb_traffic_map.html')

@app.route('/api/status')
def get_status():
    """Basit status kontrolü - database_selector_web için"""
    return jsonify({"status": "running", "database": "arangodb", "port": 5001})

@app.route('/api/timestamps')
def get_timestamps():
    """Tüm mevcut timestamp'leri döndür"""
    try:
        client, db = get_db()
        
        # AQL sorgusu ile unique timestamp'leri al
        query = """
        FOR m IN Measure
        COLLECT ts = m.timestamp
        SORT ts DESC
        RETURN ts
        """
        cursor = db.aql.execute(query)
        timestamps = list(cursor)
        
        client.close()
        return jsonify({"timestamps": timestamps})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "timestamps": []}), 500

@app.route('/api/traffic')
def get_traffic():
    """Belirli bir timestamp için trafik verilerini döndür"""
    try:
        selected_ts = request.args.get('timestamp', None)
        print(f"[DEBUG] Requested timestamp: {selected_ts}")
        
        client, db = get_db()
        
        # Eğer timestamp seçilmemişse en son timestamp'i al
        if not selected_ts:
            print("[DEBUG] No timestamp provided, getting latest...")
            query = """
            FOR m IN Measure
            SORT m.timestamp DESC
            LIMIT 1
            RETURN m.timestamp
            """
            cursor = db.aql.execute(query)
            latest_ts_list = list(cursor)
            
            if not latest_ts_list:
                client.close()
                return jsonify({"error": "No data", "features": [], "timestamp": None})
            
            selected_ts = latest_ts_list[0]
        
        latest_ts = selected_ts
        print(f"[DEBUG] Using timestamp: {latest_ts}")
        
        # Verileri çek - JOIN ile Segment + Measure
        query = """
        FOR s IN Segment
            FOR m IN Measure
                FILTER m.segmentId == s.segmentId AND m.timestamp == @ts
                RETURN {
                    segmentId: s.segmentId,
                    geometry: s.geom,
                    jamFactor: m.jamFactor,
                    speed: m.speed,
                    freeFlow: m.freeFlow,
                    confidence: m.confidence
                }
        """
        cursor = db.aql.execute(query, bind_vars={'ts': latest_ts})
        
        print(f"[DEBUG] Query executed for timestamp: {latest_ts}")
        
        features = []
        record_count = 0
        for record in cursor:
            record_count += 1
            geom = record["geometry"]
            if not geom:
                continue
            
            try:
                # WKT'den GeoJSON'a çevir
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
        
        client.close()
        
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

if __name__ == '__main__':
    print("="*70)
    print("  ARANGODB CANLI TRAFIK HARITASI")
    print("="*70)
    print()
    print(f"ArangoDB: {ARANGO_HOST}")
    print(f"Database: {ARANGO_DATABASE}")
    print()
    print("Tarayıcıda açın: http://localhost:5001")
    print()
    print("Durdurmak için: Ctrl+C")
    print("="*70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
