#!/usr/bin/env python3
"""
09_neo4j_map_viewer.py
----------------------
Neo4j'den trafik verilerini çekip interaktif harita üzerinde gösterir.
Kullanıcı zaman seçimi yapabilir.

Kullanım:
  python 09_neo4j_map_viewer.py

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
from dotenv import load_dotenv

from neo4j import GraphDatabase
import folium
from folium import plugins

# .env dosyasını yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

# ---------- Neo4j Bağlantısı ----------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def connect_neo4j():
    """Neo4j bağlantısı oluştur"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        return driver
    except Exception as e:
        print(f"❌ Neo4j bağlantı hatası: {e}")
        sys.exit(1)

# ---------- Veri Çekme Fonksiyonları ----------
def get_available_timestamps(driver):
    """Neo4j'den mevcut tüm timestamp'leri çek"""
    query = """
    MATCH (m:Measure)
    RETURN DISTINCT m.timestamp AS timestamp
    ORDER BY timestamp DESC
    """
    
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        timestamps = [record["timestamp"] for record in result]
    
    return timestamps

def get_segments_with_flow(driver, timestamp=None):
    """
    Neo4j'den segment ve trafik verilerini çek
    timestamp verilirse o zamana ait verileri, verilmezse en son verileri çeker
    """
    if timestamp:
        query = """
        MATCH (s:Segment)
        OPTIONAL MATCH (m:Measure {segmentId: s.segmentId, timestamp: $timestamp})
        RETURN 
            s.segmentId AS segmentId,
            s.geom AS geometry,
            m.jamFactor AS jamFactor,
            m.speed AS speed,
            m.freeFlow AS freeFlow,
            m.confidence AS confidence,
            m.timestamp AS timestamp
        """
        params = {"timestamp": timestamp}
    else:
        # En son timestamp'i kullan
        query = """
        MATCH (s:Segment)
        OPTIONAL MATCH (m:Measure {segmentId: s.segmentId})
        WITH s, m
        ORDER BY m.timestamp DESC
        WITH s, COLLECT(m)[0] AS latest_m
        RETURN 
            s.segmentId AS segmentId,
            s.geom AS geometry,
            latest_m.jamFactor AS jamFactor,
            latest_m.speed AS speed,
            latest_m.freeFlow AS freeFlow,
            latest_m.confidence AS confidence,
            latest_m.timestamp AS timestamp
        """
        params = {}
    
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params)
        segments = []
        for record in result:
            segment = {
                "segmentId": record["segmentId"],
                "geometry": record["geometry"],
                "jamFactor": record["jamFactor"],
                "speed": record["speed"],
                "freeFlow": record["freeFlow"],
                "confidence": record["confidence"],
                "timestamp": record["timestamp"]
            }
            segments.append(segment)
    
    return segments

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
def create_map(segments, timestamp_str=None):
    """Folium haritası oluştur"""
    
    if not segments:
        print("⚠️  Gösterilecek segment bulunamadı!")
        return None
    
    # Harita merkezi hesapla (geometry'den)
    all_coords = []
    for seg in segments:
        geom = seg.get("geometry")
        if geom:
            try:
                # WKT formatından koordinatları çıkar
                # Format: LINESTRING(lon1 lat1, lon2 lat2, ...)
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
    
    # Segment'leri çiz
    for seg in segments:
        geom = seg.get("geometry")
        if not geom:
            continue
        
        # WKT LineString parse et
        try:
            # Format: LINESTRING(lon1 lat1, lon2 lat2, ...)
            coords_str = geom.replace("LINESTRING(", "").replace(")", "")
            coord_pairs = coords_str.split(", ")
            line_coords = []
            for pair in coord_pairs:
                lon, lat = map(float, pair.split())
                line_coords.append([lat, lon])
        except Exception as e:
            continue
        
        # Renk belirle
        color = get_color_from_jam_factor(seg.get("jamFactor"))
        
        # Popup bilgisi
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; min-width: 200px;">
            <b>Segment ID:</b> {seg.get('segmentId', 'N/A')}<br>
            <hr>
            <b>Hız:</b> {seg.get('speed', 'N/A')} km/h<br>
            <b>Serbest Akış:</b> {seg.get('freeFlow', 'N/A')} km/h<br>
            <b>Jam Factor:</b> {seg.get('jamFactor', 'N/A')}<br>
            <b>Güven:</b> {seg.get('confidence', 'N/A')}<br>
            <b>Zaman:</b> {seg.get('timestamp', 'N/A')}<br>
        </div>
        """
        
        # PolyLine ekle
        folium.PolyLine(
            line_coords,
            color=color,
            weight=7,
            opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
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
    '''
    
    if timestamp_str:
        legend_html += f'<hr style="margin: 5px 0;"><p style="margin: 5px 0; font-size: 11px;"><b>Zaman:</b><br>{timestamp_str}</p>'
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Tam ekran özelliği ekle
    plugins.Fullscreen().add_to(m)
    
    return m

# ---------- Ana Program ----------
def main():
    print("=" * 70)
    print("  NEO4J TRAFİK HARİTASI GÖRÜNTÜLEYİCİ")
    print("=" * 70)
    print()
    
    # Neo4j'ye bağlan
    print("🔗 Neo4j'ye bağlanılıyor...")
    driver = connect_neo4j()
    print(f"✅ Bağlantı başarılı: {NEO4J_URI}")
    print()
    
    # Mevcut timestamp'leri çek
    print("📅 Mevcut zaman verileri çekiliyor...")
    timestamps = get_available_timestamps(driver)
    
    if not timestamps:
        print("⚠️  Neo4j'de hiç zaman verisi bulunamadı!")
        print("   Önce verileri yükleyin: python 06_auto_load_to_neo4j.py")
        driver.close()
        return
    
    print(f"✅ {len(timestamps)} farklı zaman verisi bulundu")
    print()
    
    # Kullanıcıya timestamp seçtir
    print("=" * 70)
    print("MEVCUT ZAMAN VERİLERİ:")
    print("=" * 70)
    for i, ts in enumerate(timestamps, 1):
        print(f"{i}. {ts}")
    print()
    print("0. Tümünü göster (en son veri)")
    print("=" * 70)
    print()
    
    # Kullanıcı seçimi
    while True:
        try:
            choice = input("Seçiminiz (0-{}): ".format(len(timestamps)))
            choice = int(choice)
            if 0 <= choice <= len(timestamps):
                break
            else:
                print(f"⚠️  Lütfen 0 ile {len(timestamps)} arasında bir sayı girin!")
        except ValueError:
            print("⚠️  Lütfen geçerli bir sayı girin!")
        except KeyboardInterrupt:
            print("\n\n👋 İptal edildi.")
            driver.close()
            return
    
    # Seçime göre veri çek
    if choice == 0:
        print("\n📊 En son veriler çekiliyor...")
        selected_timestamp = None
        timestamp_str = "En Son Veri"
    else:
        selected_timestamp = timestamps[choice - 1]
        timestamp_str = selected_timestamp
        print(f"\n📊 {selected_timestamp} verileri çekiliyor...")
    
    segments = get_segments_with_flow(driver, selected_timestamp)
    print(f"✅ {len(segments)} segment çekildi")
    
    # Veri istatistikleri
    segments_with_data = [s for s in segments if s["jamFactor"] is not None]
    print(f"   └─ {len(segments_with_data)} segment'te trafik verisi var")
    print()
    
    # Harita oluştur
    print("🗺️  Harita oluşturuluyor...")
    m = create_map(segments, timestamp_str)
    
    if m:
        # Haritayı kaydet
        output_file = "neo4j_traffic_map.html"
        m.save(output_file)
        print(f"✅ Harita kaydedildi: {output_file}")
        print()
        
        # Tarayıcıda aç
        import webbrowser
        import os
        abs_path = os.path.abspath(output_file)
        webbrowser.open('file://' + abs_path)
        print("🌐 Harita tarayıcıda açıldı!")
    
    # Bağlantıyı kapat
    driver.close()
    print()
    print("=" * 70)
    print("✅ İşlem tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()
