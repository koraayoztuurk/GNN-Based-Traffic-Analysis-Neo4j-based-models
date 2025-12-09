#!/usr/bin/env python3
"""
14_tigergraph_web_server.py
----------------------------
TigerGraph trafik haritası web server
"""
import os
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
import pyTigerGraph as tg
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

app = Flask(__name__, static_folder='.')

# TigerGraph Bağlantı
TG_HOST = os.getenv("TG_HOST", "http://127.0.0.1")
TG_GRAPH = os.getenv("TG_GRAPH", "TrafficGraph")
TG_USER = os.getenv("TG_USER", "tigergraph")
TG_PASS = os.getenv("TG_PASS", "tigergraph")
TG_REST_PORT = int(os.getenv("TG_REST_PORT", "14240"))

def get_connection():
    """TigerGraph bağlantısı kur"""
    conn = tg.TigerGraphConnection(
        host=TG_HOST,
        graphname=TG_GRAPH,
        username=TG_USER,
        password=TG_PASS,
        restppPort=TG_REST_PORT
    )
    return conn

@app.route('/')
def index():
    return send_from_directory('.', 'tigergraph_traffic_map.html')

@app.route('/api/status')
def get_status():
    """Basit status kontrolü - database_selector_web için"""
    return jsonify({"status": "running", "database": "tigergraph", "port": 5002})

@app.route('/api/timestamps')
def get_timestamps():
    """Tüm mevcut timestamp'leri döndür"""
    try:
        conn = get_connection()
        
        # Tüm Measure vertex'lerini çek
        print("[DEBUG] Fetching all Measure vertices...")
        measures = conn.getVertices("Measure", limit=999999)
        print(f"[DEBUG] Got {len(measures)} measures")
        print(f"[DEBUG] Type of measures: {type(measures)}")
        
        # Unique timestamp'leri topla
        timestamps = set()
        
        # Check if it's a list or dict
        if isinstance(measures, list):
            for vertex_data in measures:
                ts = vertex_data.get("attributes", {}).get("timestamp")
                if ts:
                    timestamps.add(ts)
        else:
            for vertex_id, vertex_data in measures.items():
                ts = vertex_data.get("attributes", {}).get("timestamp")
                if ts:
                    timestamps.add(ts)
        
        timestamps_list = sorted(list(timestamps), reverse=True)
        print(f"[DEBUG] Found {len(timestamps_list)} unique timestamps")
        
        return jsonify({"timestamps": timestamps_list})
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "timestamps": []}), 500

@app.route('/api/traffic')
def get_traffic():
    """Belirli bir timestamp için trafik verilerini döndür"""
    try:
        selected_ts = request.args.get('timestamp', None)
        print(f"[DEBUG] Requested timestamp: {selected_ts}")
        
        conn = get_connection()
        
        # Eğer timestamp seçilmemişse en son timestamp'i al
        if not selected_ts:
            print("[DEBUG] No timestamp provided, getting latest...")
            measures = conn.getVertices("Measure", limit=999999)
            timestamps = set()
            
            if isinstance(measures, list):
                for vertex_data in measures:
                    ts = vertex_data.get("attributes", {}).get("timestamp")
                    if ts:
                        timestamps.add(ts)
            else:
                for vertex_id, vertex_data in measures.items():
                    ts = vertex_data.get("attributes", {}).get("timestamp")
                    if ts:
                        timestamps.add(ts)
            
            if not timestamps:
                return jsonify({"error": "No data", "features": [], "timestamp": None})
            
            selected_ts = sorted(list(timestamps), reverse=True)[0]
        
        latest_ts = selected_ts
        print(f"[DEBUG] Using timestamp: {latest_ts}")
        
        # Tüm Segment ve Measure vertex'lerini çek
        print("[DEBUG] Fetching segments and measures...")
        segments = conn.getVertices("Segment", limit=999999)
        measures = conn.getVertices("Measure", limit=999999)
        
        print(f"[DEBUG] Got {len(segments)} segments and {len(measures)} measures")
        print(f"[DEBUG] Segments type: {type(segments)}, Measures type: {type(measures)}")
        
        # Seçili timestamp için measure'ları filtrele
        measures_for_ts = {}
        
        if isinstance(measures, list):
            for vertex_data in measures:
                attrs = vertex_data.get("attributes", {})
                if attrs.get("timestamp") == latest_ts:
                    segment_id = attrs.get("segmentId")
                    if segment_id:
                        measures_for_ts[segment_id] = attrs
        else:
            for vertex_id, vertex_data in measures.items():
                attrs = vertex_data.get("attributes", {})
                if attrs.get("timestamp") == latest_ts:
                    segment_id = attrs.get("segmentId")
                    if segment_id:
                        measures_for_ts[segment_id] = attrs
        
        print(f"[DEBUG] Filtered to {len(measures_for_ts)} measures for timestamp: {latest_ts}")
        
        # GeoJSON features oluştur
        features = []
        
        # Handle both list and dict formats
        if isinstance(segments, list):
            for vertex_data in segments:
                attrs = vertex_data.get("attributes", {})
                segment_id = attrs.get("segmentId")
                
                # Bu segment'in measure'ını bul
                if segment_id not in measures_for_ts:
                    continue
                
                measure = measures_for_ts[segment_id]
                geom = attrs.get("geom")
                
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
                    jam_factor = measure.get("jamFactor")
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
                            "segmentId": segment_id,
                            "jamFactor": jam_factor,
                            "speed": measure.get("speed"),
                            "freeFlow": measure.get("freeFlow"),
                            "confidence": measure.get("confidence"),
                            "color": color
                        }
                    }
                    features.append(feature)
                except Exception as e:
                    print(f"Error processing segment {segment_id}: {e}")
                    continue
        else:
            for vertex_id, vertex_data in segments.items():
                attrs = vertex_data.get("attributes", {})
                segment_id = attrs.get("segmentId")
                
                # Bu segment'in measure'ını bul
                if segment_id not in measures_for_ts:
                    continue
                
                measure = measures_for_ts[segment_id]
                geom = attrs.get("geom")
                
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
                    jam_factor = measure.get("jamFactor")
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
                            "segmentId": segment_id,
                            "jamFactor": jam_factor,
                            "speed": measure.get("speed"),
                            "freeFlow": measure.get("freeFlow"),
                            "confidence": measure.get("confidence"),
                            "color": color
                        }
                    }
                    features.append(feature)
                except Exception as e:
                    print(f"Error processing segment {segment_id}: {e}")
                    continue
        
        print(f"[DEBUG] Created {len(features)} features")
        
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
    print("  TIGERGRAPH CANLI TRAFIK HARITASI")
    print("="*70)
    print()
    print(f"TigerGraph: {TG_HOST}:{TG_REST_PORT}")
    print(f"Graph: {TG_GRAPH}")
    print()
    print("Tarayıcıda açın: http://localhost:5002")
    print()
    print("Durdurmak için: Ctrl+C")
    print("="*70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5002)
