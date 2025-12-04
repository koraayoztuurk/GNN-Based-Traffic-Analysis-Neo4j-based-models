#!/usr/bin/env python3
"""
Test Pipeline - Tüm aşamaları adım adım test et
"""
import sys
import subprocess
from pathlib import Path
import time

ROOT = Path(__file__).parent

def run_step(step_num, name, script_path, args=None):
    """Bir adımı çalıştır ve sonucu göster"""
    print("\n" + "=" * 70)
    print(f"🔹 ADIM {step_num}: {name}")
    print("=" * 70)
    
    cmd = [sys.executable, str(ROOT / script_path)]
    if args:
        cmd.extend(args)
    
    print(f"▶️  Çalıştırılıyor: {' '.join(cmd)}")
    print()
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✅ BAŞARILI! (Süre: {duration:.1f}s)")
        return True
    else:
        print(f"\n❌ HATA! (Exit code: {result.returncode})")
        response = input("\nDevam etmek istiyor musunuz? (e/h): ")
        return response.lower() == 'e'

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     📋 FULL PIPELINE TEST                        ║
║                                                                  ║
║  Tüm aşamaları sırayla test edeceğiz:                          ║
║  1. HERE API veri çekme                                         ║
║  2. Harita render                                               ║
║  3. Timeseries oluşturma                                        ║
║  4. Neo4j'ye yükleme                                            ║
║  5. GNN hazırlık kontrolü                                       ║
║  6. Feature engineering                                          ║
║  7. PyTorch Geometric export                                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    response = input("Başlayalım mı? (e/h): ")
    if response.lower() != 'e':
        print("❌ İptal edildi")
        return 1
    
    steps = [
        (1, "HERE API Veri Çekme", "src/pipeline/01_fetch_here_flow.py"),
        (2, "Harita Render & Arşivleme", "src/pipeline/02_render_flow_map.py"),
        (3, "Timeseries Oluşturma", "src/pipeline/05_build_timeseries.py"),
        (4, "Neo4j'ye Yükleme", "src/neo4j/06_auto_load_to_neo4j.py"),
        (5, "GNN Hazırlık Kontrolü", "src/gnn/test_gnn_readiness.py"),
    ]
    
    completed = []
    
    for step_num, name, script in steps:
        if run_step(step_num, name, script):
            completed.append(name)
        else:
            print(f"\n⚠️  {name} adımında durdu")
            break
    
    # Özet
    print("\n" + "=" * 70)
    print("📊 TEST SONUÇLARI")
    print("=" * 70)
    print(f"\n✅ Tamamlanan: {len(completed)}/{len(steps)}")
    for i, name in enumerate(completed, 1):
        print(f"  {i}. {name}")
    
    if len(completed) == len(steps):
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("\nŞimdi GNN adımlarına geçebilirsiniz:")
        print("  • python src/gnn/run_step1_enhance_schema.py")
        print("  • python src/gnn/run_step2_build_connects_to.py")
        print("  • python src/gnn/04_generate_features.py")
        print("  • python src/gnn/05_export_pyg.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
