#!/usr/bin/env python3
"""
run_pipeline.py
---------------
TEK SEFERLİK PIPELINE

Tüm işlemleri sırayla yapar:
1. HERE API'den trafik verisi çeker
2. Harita oluşturur ve arşivler
3. Timeseries oluşturur
4. TÜM AKTİF VERİTABANLARINA yükler (Neo4j, ArangoDB, TigerGraph)
5. Koordinat çıkarır + CONNECTS_TO bağlantıları oluşturur

Kullanım:
    python run_pipeline.py
    
Not: .env dosyasındaki ACTIVE_DATABASES değişkenine göre hangi veritabanlarına 
     yükleme yapılacağı belirlenir (örn: neo4j,tigergraph,arangodb)
"""
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent
PYTHON = sys.executable

# PYTHONPATH'i ayarla (src modüllerini bulabilmesi için)
import os
os.environ['PYTHONPATH'] = str(ROOT_DIR)

def run_step(name, script_path):
    """Bir adımı çalıştır"""
    print(f"\n{'='*70}")
    print(f"▶️  {name}")
    print('='*70)
    
    # Script + parametreleri ayır
    if isinstance(script_path, list):
        cmd = [PYTHON] + [str(p) for p in script_path]
    else:
        cmd = [PYTHON, str(script_path)]
    
    # Environment'ı kopyala ve PYTHONPATH ekle
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT_DIR)
    
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
    
    if result.returncode != 0:
        print(f"❌ HATA: {name} başarısız oldu!")
        return False
    
    print(f"✅ {name} tamamlandı!")
    return True

def main():
    print("\n" + "🚀 "*20)
    print("     TAM OTOMATİK MULTI-DB PIPELINE")
    print("🚀 "*20 + "\n")
    
    steps = [
        ("0. Schema Oluşturma (İlk Kez Gerekli)", [ROOT_DIR / "src/pipeline/multi_db_loader.py", "--init-schema"]),
        ("1. HERE API Veri Çekme", ROOT_DIR / "src/pipeline/fetch_here_flow.py"),
        ("2. Harita Render & Arşivleme", ROOT_DIR / "src/pipeline/render_flow_map.py"),
        ("3. Timeseries Oluşturma", ROOT_DIR / "src/pipeline/build_timeseries.py"),
        ("4. MULTI-DB Yükleme + Topoloji (Neo4j + TigerGraph + ArangoDB)", [ROOT_DIR / "src/pipeline/multi_db_loader.py", "--all"]),
    ]
    
    for name, script in steps:
        if not run_step(name, script):
            print("\n❌ Pipeline durduruldu!")
            return 1
    
    print("\n" + "🎉 "*20)
    print("     TÜM İŞLEMLER BAŞARILI!")
    print("🎉 "*20 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
