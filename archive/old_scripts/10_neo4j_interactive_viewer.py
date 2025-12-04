#!/usr/bin/env python3
"""
10_neo4j_interactive_viewer.py
-------------------------------
Neo4j'den trafik verilerini çekip zaman kaydırıcılı (time slider) 
interaktif harita üzerinde gösterir.

Kullanım:
  python 10_neo4j_interactive_viewer.py

ENV değişkenleri:
  NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import json

from neo4j import GraphDatabase
import folium
from folium import plugins

# ---------- Neo4j Bağlantısı ----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")

def connect_neo4j():
    """Neo4j bağlantısı oluştur"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        return driver
    except Exception as e:
        print(f"❌ Neo4j bağlantı hatası: {e}")
        sys.exit(1)

# ---------- Veri Çekme Fonksiyonları ----------
def get_all_data_with_timeline(driver):
    """
    Neo4j'den tüm zaman dilimlerindeki segment ve trafik verilerini çek
    """
    query = """
    MATCH (s:Segment)
    OPTIONAL MATCH (m:Measure {segmentId: s.segmentId})
    WHERE m.timestamp IS NOT NULL
    RETURN 
        s.segmentId AS segmentId,
        s.geom AS geometry,
        m.jamFactor AS jamFactor,
        m.speed AS speed,
        m.freeFlow AS freeFlow,
        m.confidence AS confidence,
        m.timestamp AS timestamp
    ORDER BY m.timestamp
    """
    
    with driver.session() as session:
        result = session.run(query)
        data = []
        for record in result:
            data.append({
                "segmentId": record["segmentId"],
                "geometry": record["geometry"],
                "jamFactor": record["jamFactor"],
                "speed": record["speed"],
                "freeFlow": record["freeFlow"],
                "confidence": record["confidence"],
                "timestamp": record["timestamp"]
            })
    
    return data

def get_segments_only(driver):
    """Sadece segment geometrilerini çek (trafik verisi olmadan)"""
    query = """
    MATCH (s:Segment)
    RETURN 
        s.segmentId AS segmentId,
        s.geom AS geometry
    """
    
    with driver.session() as session:
        result = session.run(query)
        segments = {}
        for record in result:
            segments[record["segmentId"]] = record["geometry"]
    
    return segments

def organize_by_timestamp(data):
    """Verileri timestamp'e göre grupla"""
    timeline = {}
    for item in data:
        ts = item["timestamp"]
        if ts not in timeline:
            timeline[ts] = []
        timeline[ts].append(item)
    
    return timeline

# ---------- Renk Hesaplama ----------
def get_color_from_jam_factor(jam_factor):
    """
    Jam factor'a göre renk döndür
    0.0 = yeşil (serbest akış)
    5.0 = koyu kırmızı (tam tıkanıklık)
    10.0 = siyah (durma)
    """
    if jam_factor is None:
        return "#808080"  # Gri (veri yok)
    
    jf = float(jam_factor)
    
    if jf <= 1.0:
        return "#00FF00"  # Yeşil
    elif jf <= 2.0:
        return "#7FFF00"  # Açık yeşil
    elif jf <= 3.0:
        return "#FFFF00"  # Sarı
    elif jf <= 4.0:
        return "#FFA500"  # Turuncu
    elif jf <= 5.0:
        return "#FF6600"  # Koyu turuncu
    elif jf <= 7.0:
        return "#FF0000"  # Kırmızı
    elif jf <= 9.0:
        return "#CC0000"  # Koyu kırmızı
    else:
        return "#000000"  # Siyah

# ---------- Harita Oluşturma ----------
def create_timeline_map(timeline, segments):
    """Zaman kaydırıcılı (TimestampedGeoJson) haritası oluştur"""
    
    if not timeline:
        print("⚠️  Gösterilecek veri bulunamadı!")
        return None
    
    # Harita merkezi hesapla
    all_coords = []
    for geom in list(segments.values())[:100]:  # İlk 100 segment yeterli
        if geom:
            try:
                coords_str = geom.replace("LINESTRING(", "").replace(")", "")
                coord_pairs = coords_str.split(", ")
                for pair in coord_pairs:
                    lon, lat = map(float, pair.split())
                    all_coords.append([lat, lon])
            except:
                pass
    
    if not all_coords:
        center_lat, center_lon = 41.0082, 28.9784  # İstanbul default
    else:
        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)
    
    # Harita oluştur
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # GeoJSON features hazırla (TimestampedGeoJson için)
    features = []
    
    for timestamp, measures in sorted(timeline.items()):
        if timestamp is None:
            continue
            
        for measure in measures:
            seg_id = measure["segmentId"]
            geom = segments.get(seg_id)
            
            if not geom:
                continue
            
            try:
                # WKT'den GeoJSON'a çevir
                coords_str = geom.replace("LINESTRING(", "").replace(")", "")
                coord_pairs = coords_str.split(", ")
                coordinates = []
                for pair in coord_pairs:
                    lon, lat = map(float, pair.split())
                    coordinates.append([lon, lat])
                
                # Renk belirle
                color = get_color_from_jam_factor(measure.get("jamFactor"))
                
                # GeoJSON feature oluştur
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "time": str(timestamp),
                        "style": {
                            "color": color,
                            "weight": 7,
                            "opacity": 0.9
                        },
                        "popup": f"""
                        <b>Segment:</b> {seg_id}<br>
                        <b>Hız:</b> {measure.get('speed', 'N/A')} km/h<br>
                        <b>Serbest Akış:</b> {measure.get('freeFlow', 'N/A')} km/h<br>
                        <b>Jam Factor:</b> {measure.get('jamFactor', 'N/A')}<br>
                        <b>Güven:</b> {measure.get('confidence', 'N/A')}<br>
                        <b>Zaman:</b> {timestamp}
                        """
                    }
                }
                features.append(feature)
            except:
                continue
    
    # TimestampedGeoJson ekle
    if features:
        plugins.TimestampedGeoJson({
            "type": "FeatureCollection",
            "features": features
        },
        period="PT1M",  # 1 dakikalık periyotlar
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm:ss",
        time_slider_drag_update=True
        ).add_to(m)
    
    # Legend (açıklama) ekle
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 280px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">Trafik Durumu</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><span style="color: #00FF00;">━━━</span> Çok İyi (0-1)</p>
    <p style="margin: 5px 0;"><span style="color: #7FFF00;">━━━</span> İyi (1-2)</p>
    <p style="margin: 5px 0;"><span style="color: #FFFF00;">━━━</span> Normal (2-3)</p>
    <p style="margin: 5px 0;"><span style="color: #FFA500;">━━━</span> Yoğun (3-4)</p>
    <p style="margin: 5px 0;"><span style="color: #FF6600;">━━━</span> Çok Yoğun (4-5)</p>
    <p style="margin: 5px 0;"><span style="color: #FF0000;">━━━</span> Tıkanık (5-7)</p>
    <p style="margin: 5px 0;"><span style="color: #CC0000;">━━━</span> Çok Tıkanık (7-9)</p>
    <p style="margin: 5px 0;"><span style="color: #000000;">━━━</span> Durmuş (9+)</p>
    <p style="margin: 5px 0;"><span style="color: #808080;">━━━</span> Veri Yok</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0; font-size: 10px;"><i>Zaman kaydırıcısı ile<br>zamanda gezinin!</i></p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Tam ekran özelliği ekle
    plugins.Fullscreen().add_to(m)
    
    return m

# ---------- Ana Program ----------
def main():
    print("=" * 70)
    print("  NEO4J İNTERAKTİF TRAFİK HARİTASI (ZAMAN KAYDIRICI)")
    print("=" * 70)
    print()
    
    # Neo4j'ye bağlan
    print("🔗 Neo4j'ye bağlanılıyor...")
    driver = connect_neo4j()
    print(f"✅ Bağlantı başarılı: {NEO4J_URI}")
    print()
    
    # Tüm segment geometrilerini çek
    print("📍 Segment geometrileri çekiliyor...")
    segments = get_segments_only(driver)
    print(f"✅ {len(segments)} segment bulundu")
    print()
    
    # Tüm zaman dilimlerindeki verileri çek
    print("📊 Tüm trafik verileri çekiliyor...")
    all_data = get_all_data_with_timeline(driver)
    print(f"✅ {len(all_data)} veri noktası çekildi")
    print()
    
    # Verileri timestamp'e göre grupla
    print("🕒 Veriler zaman çizelgesine göre düzenleniyor...")
    timeline = organize_by_timestamp(all_data)
    print(f"✅ {len(timeline)} farklı zaman dilimi")
    
    # İstatistikler
    total_with_data = sum(len(measures) for measures in timeline.values())
    print(f"   └─ Toplam {total_with_data} segment-zaman kombinasyonu")
    print()
    
    # Harita oluştur
    print("🗺️  İnteraktif harita oluşturuluyor...")
    print("   (Bu işlem birkaç saniye sürebilir...)")
    m = create_timeline_map(timeline, segments)
    
    if m:
        # Haritayı kaydet
        output_file = "neo4j_interactive_map.html"
        m.save(output_file)
        print(f"✅ Harita kaydedildi: {output_file}")
        print()
        
        # Tarayıcıda aç
        import webbrowser
        import os
        abs_path = os.path.abspath(output_file)
        webbrowser.open('file://' + abs_path)
        print("🌐 Harita tarayıcıda açıldı!")
        print()
        print("💡 İPUCU:")
        print("   - Haritanın alt kısmındaki zaman kaydırıcısını kullanarak")
        print("     farklı zaman dilimlerindeki trafik durumunu görebilirsiniz")
        print("   - Play butonu ile animasyon başlatabilirsiniz")
    
    # Bağlantıyı kapat
    driver.close()
    print()
    print("=" * 70)
    print("✅ İşlem tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()
