#!/usr/bin/env python3
"""
11_web_server.py
----------------
Neo4j'den canlı trafik verilerini çeken web server.
Flask ile API sağlar ve dinamik harita gösterir.

Kullanım:
  python 11_web_server.py

Sonra tarayıcıda:
  http://localhost:5000
"""
import os
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from neo4j import GraphDatabase

app = Flask(__name__)

# Neo4j Bağlantı Bilgileri
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")

def get_neo4j_driver():
    """Neo4j bağlantısı oluştur"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_latest_traffic_data():
    """En son trafik verilerini çek"""
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # En son timestamp'i bul
        result = session.run("""
            MATCH (m:Measure)
            RETURN m.timestamp AS ts
            ORDER BY ts DESC
            LIMIT 1
        """)
        latest_ts = result.single()
        if not latest_ts:
            driver.close()
            return None, None
        
        latest_timestamp = latest_ts["ts"]
        
        # O timestamp için tüm verileri çek
        result = session.run("""
            MATCH (s:Segment)
            OPTIONAL MATCH (m:Measure {segmentId: s.segmentId, timestamp: $ts})
            RETURN 
                s.segmentId AS segmentId,
                s.geom AS geometry,
                m.jamFactor AS jamFactor,
                m.speed AS speed,
                m.freeFlow AS freeFlow,
                m.confidence AS confidence
        """, ts=latest_timestamp)
        
        features = []
        for record in result:
            geom = record["geometry"]
            if not geom:
                continue
            
            try:
                # WKT'den GeoJSON'a çevir (WKT formatı: lon lat)
                coords_str = geom.replace("LINESTRING(", "").replace(")", "")
                coord_pairs = coords_str.split(", ")
                coordinates = []
                for pair in coord_pairs:
                    parts = pair.split()
                    lon = float(parts[0])
                    lat = float(parts[1])
                    coordinates.append([lon, lat])
                
                # Renk belirle
                jam_factor = record["jamFactor"]
                if jam_factor is None:
                    color = "#808080"
                else:
                    jf = float(jam_factor)
                    if jf <= 1.0:
                        color = "#00FF00"
                    elif jf <= 2.0:
                        color = "#7FFF00"
                    elif jf <= 3.0:
                        color = "#FFFF00"
                    elif jf <= 4.0:
                        color = "#FFA500"
                    elif jf <= 5.0:
                        color = "#FF6600"
                    elif jf <= 7.0:
                        color = "#FF0000"
                    elif jf <= 9.0:
                        color = "#CC0000"
                    else:
                        color = "#000000"
                
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
            except:
                continue
        
    driver.close()
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson, latest_timestamp

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Canlı Trafik Haritası - Neo4j</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }
        #map {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
        }
        .info-panel {
            position: fixed;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }
        .legend {
            position: fixed;
            bottom: 30px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 12px;
        }
        .legend-item {
            margin: 5px 0;
        }
        .legend-color {
            display: inline-block;
            width: 30px;
            height: 3px;
            margin-right: 5px;
            vertical-align: middle;
        }
        .refresh-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }
        .refresh-btn:hover {
            background: #45a049;
        }
        .loading {
            display: none;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-panel">
        <h3 style="margin: 0 0 10px 0;">🚦 Canlı Trafik</h3>
        <div id="timestamp">Yükleniyor...</div>
        <div id="stats">-</div>
        <button class="refresh-btn" onclick="refreshData()">🔄 Yenile</button>
        <div class="loading" id="loading">Yenileniyor...</div>
    </div>
    
    <div class="legend">
        <h4 style="margin: 0 0 10px 0;">Trafik Durumu</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #00FF00;"></span> Çok İyi (0-1)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #7FFF00;"></span> İyi (1-2)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #FFFF00;"></span> Normal (2-3)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #FFA500;"></span> Yoğun (3-4)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #FF6600;"></span> Çok Yoğun (4-5)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #FF0000;"></span> Tıkanık (5-7)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #CC0000;"></span> Çok Tıkanık (7-9)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #000000;"></span> Durmuş (9+)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #808080;"></span> Veri Yok
        </div>
    </div>

    <script>
        // Harita oluştur (Eskişehir merkez)
        var map = L.map('map').setView([39.7767, 30.5206], 12);
        
        // Tile layer ekle
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        var trafficLayer = null;
        
        function loadTrafficData() {
            document.getElementById('loading').style.display = 'block';
            
            fetch('/api/traffic')
                .then(response => response.json())
                .then(data => {
                    // Eski katmanı kaldır
                    if (trafficLayer) {
                        map.removeLayer(trafficLayer);
                    }
                    
                    // Yeni katman ekle
                    trafficLayer = L.geoJSON(data.geojson, {
                        style: function(feature) {
                            return {
                                color: feature.properties.color,
                                weight: 7,
                                opacity: 0.9
                            };
                        },
                        onEachFeature: function(feature, layer) {
                            var props = feature.properties;
                            var popupContent = `
                                <b>Segment:</b> ${props.segmentId}<br>
                                <b>Hız:</b> ${props.speed ? props.speed.toFixed(1) : 'N/A'} km/h<br>
                                <b>Serbest Akış:</b> ${props.freeFlow ? props.freeFlow.toFixed(1) : 'N/A'} km/h<br>
                                <b>Jam Factor:</b> ${props.jamFactor ? props.jamFactor.toFixed(1) : 'N/A'}<br>
                                <b>Güven:</b> ${props.confidence ? props.confidence.toFixed(2) : 'N/A'}
                            `;
                            layer.bindPopup(popupContent);
                        }
                    }).addTo(map);
                    
                    // İlk yüklemede haritayı veri bölgesine odakla
                    if (!window.mapCentered && data.geojson.features.length > 0) {
                        try {
                            map.fitBounds(trafficLayer.getBounds(), {padding: [50, 50]});
                            window.mapCentered = true;
                        } catch(e) {
                            console.log("Bounds hesaplanamadı");
                        }
                    }
                    
                    // Bilgileri güncelle
                    document.getElementById('timestamp').innerHTML = 
                        `<b>Son Güncelleme:</b><br>${data.timestamp || 'Bilinmiyor'}`;
                    document.getElementById('stats').innerHTML = 
                        `<b>Toplam Segment:</b> ${data.geojson.features.length}`;
                    document.getElementById('loading').style.display = 'none';
                })
                .catch(error => {
                    console.error('Veri yüklenirken hata:', error);
                    document.getElementById('loading').style.display = 'none';
                    alert('Veri yüklenirken hata oluştu!');
                });
        }
        
        function refreshData() {
            loadTrafficData();
        }
        
        // Sayfa yüklendiğinde verileri çek
        loadTrafficData();
        
        // Her 60 saniyede bir otomatik yenile
        setInterval(loadTrafficData, 60000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/traffic')
def api_traffic():
    """Trafik verilerini JSON olarak döndür"""
    try:
        geojson, timestamp = get_latest_traffic_data()
        if geojson is None:
            return jsonify({
                "error": "No data available",
                "geojson": {"type": "FeatureCollection", "features": []},
                "timestamp": None
            })
        
        return jsonify({
            "geojson": geojson,
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "geojson": {"type": "FeatureCollection", "features": []},
            "timestamp": None
        }), 500

if __name__ == '__main__':
    print("=" * 70)
    print("  CANLI TRAFİK HARİTASI WEB SERVER")
    print("=" * 70)
    print()
    print("🌐 Server başlatılıyor...")
    print(f"📍 Neo4j: {NEO4J_URI}")
    print()
    print("✅ Server hazır!")
    print()
    print("🔗 Tarayıcınızda açın:")
    print("   http://localhost:5000")
    print()
    print("💡 Özellikler:")
    print("   • Neo4j'den canlı veri çekme")
    print("   • Otomatik 60 saniyede bir yenileme")
    print("   • Manuel yenileme butonu")
    print("   • İnteraktif harita")
    print()
    print("⏹️  Durdurmak için: Ctrl+C")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
