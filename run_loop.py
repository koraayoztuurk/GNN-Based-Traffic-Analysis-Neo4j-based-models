#!/usr/bin/env python3
"""
run_loop.py
-----------
OTOMATİK DÖNGÜ PIPELINE

.env dosyasındaki PIPELINE_INTERVAL_MIN ayarına göre
sürekli çalışır ve her iterasyonda:
1. HERE API'den trafik verisi çeker
2. Harita oluşturur ve arşivler
3. Timeseries oluşturur
4. Neo4j'ye yükler
5. Koordinat çıkarır + CONNECTS_TO bağlantıları oluşturur

Kullanım:
    python run_loop.py
    
    # Özel interval (dakika):
    python run_loop.py --interval 5
    
Durdurmak için: Ctrl + C
"""
import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
PYTHON = sys.executable

# .env yükle
load_dotenv(ROOT_DIR / "config" / ".env")

def get_interval():
    """Interval'i .env'den al"""
    interval_str = os.getenv("PIPELINE_INTERVAL_MIN", "15")
    try:
        return int(interval_str)
    except ValueError:
        # Sadece sayıları al (örn: "15 min" -> 15)
        digits = "".join(c for c in interval_str if c.isdigit())
        return int(digits) if digits else 15

def run_pipeline():
    """Tek iterasyon pipeline çalıştır"""
    result = subprocess.run(
        [PYTHON, str(ROOT_DIR / "run_pipeline.py")],
        cwd=str(ROOT_DIR)
    )
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Otomatik döngü pipeline")
    parser.add_argument("--interval", type=int, help="Dakika cinsinden interval (varsayılan: .env'den)")
    args = parser.parse_args()
    
    interval = args.interval if args.interval else get_interval()
    
    print("\n" + "🔄 "*20)
    print(f"     OTOMATİK DÖNGÜ BAŞLIYOR")
    print(f"     Interval: {interval} dakika")
    print("     Durdurmak için: Ctrl + C")
    print("🔄 "*20 + "\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            print(f"\n{'#'*70}")
            print(f"# İTERASYON {iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print('#'*70 + "\n")
            
            # Pipeline çalıştır
            success = run_pipeline()
            
            if success:
                print(f"\n✅ İterasyon {iteration} başarılı!")
            else:
                print(f"\n⚠️  İterasyon {iteration} hatalarla tamamlandı")
            
            # Bekle
            print(f"\n⏸️  {interval} dakika bekleniyor...")
            print(f"   Sonraki çalışma: {datetime.now().strftime('%H:%M:%S')} + {interval} dk\n")
            
            time.sleep(interval * 60)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Pipeline kullanıcı tarafından durduruldu!")
        print(f"📊 Toplam {iteration} iterasyon tamamlandı\n")
        return 0

if __name__ == "__main__":
    sys.exit(main())
