#!/usr/bin/env python3
"""
08_auto_pipeline.py
-------------------
HERE Traffic → Neo4j tam otomatik pipeline

Bu script şu adımları otomatik yapar:
1. HERE API'den trafik verisi çeker (01_fetch_here_flow.py)
2. Harita render eder ve GeoJSON arşivler (02_render_flow_map.py)
3. Arşivlerden timeseries oluşturur (05_build_timeseries.py)
4. Neo4j'ye yükler (07_silent_load_to_neo4j.py)

Kullanım:
  # Tek seferlik çalıştır:
  python 08_auto_pipeline.py

  # Belirli aralıklarla sürekli çalıştır (15 dk):
  python 08_auto_pipeline.py --loop --interval 15

  # Sadece mevcut arşivleri yükle (HERE çekme YOK):
  python 08_auto_pipeline.py --skip-fetch

  # Detaylı log:
  python 08_auto_pipeline.py --verbose
"""
import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

# ---------- .env Loader ----------
ROOT_DIR = Path(__file__).parent.parent.parent

def load_env():
    """Load .env file into environment variables"""
    env_file = ROOT_DIR / "config" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

def get_env_int(key, default):
    """Get integer value from environment"""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        # Extract digits only (e.g., "15 min" -> 15)
        digits = "".join(ch for ch in val if ch.isdigit())
        return int(digits) if digits else default

# Load .env at module level
load_env()

# ---------- Logging Ayarları ----------
def setup_logging(verbose=False):
    """Hem konsol hem dosya logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Dosya handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Konsol handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ---------- Script Çalıştırıcı ----------
def run_script(script_name, args=None, timeout=300, critical=True):
    """
    Python script çalıştır ve sonucu döndür
    
    Args:
        script_name: Çalıştırılacak script (örn: "01_fetch_here_flow.py")
        args: Opsiyonel argümanlar (list)
        timeout: Maksimum bekleme süresi (saniye)
        critical: Hata durumunda pipeline'ı durdur mu?
    
    Returns:
        (success: bool, output: str)
    """
    logger = logging.getLogger()
    
    # Script path'ini ROOT_DIR'e göre oluştur
    if script_name.startswith("07_"):
        script_path = ROOT_DIR / "src" / "neo4j" / script_name
    elif script_name == "ensure_topology.py":
        script_path = ROOT_DIR / "src" / "gnn" / script_name
    else:
        script_path = ROOT_DIR / "src" / "pipeline" / script_name
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    logger.info(f"▶️  Çalıştırılıyor: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        
        # Çıktıları logla
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.debug(f"   {line}")
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} başarılı (exit code: 0)")
            return True, result.stdout
        else:
            logger.error(f"❌ {script_name} hata verdi (exit code: {result.returncode})")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            
            if critical:
                logger.critical(f"Pipeline durduruluyor: {script_name} kritik hata!")
                sys.exit(1)
            
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️  {script_name} timeout! ({timeout}s)")
        if critical:
            sys.exit(1)
        return False, "TIMEOUT"
    
    except Exception as e:
        logger.error(f"💥 {script_name} beklenmedik hata: {e}")
        if critical:
            sys.exit(1)
        return False, str(e)

# ---------- Pipeline Adımları ----------
def step_fetch_here_flow():
    """1. HERE API'den veri çek"""
    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info("ADIM 1: HERE Traffic Flow veri çekme")
    logger.info("=" * 70)
    
    return run_script("01_fetch_here_flow.py", timeout=60, critical=True)

def step_render_map():
    """2. Harita render et ve arşivle"""
    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info("ADIM 2: Harita render ve GeoJSON arşivleme")
    logger.info("=" * 70)
    
    return run_script("02_render_flow_map.py", timeout=60, critical=True)

def step_build_timeseries():
    """3. Arşivlerden timeseries.parquet oluştur"""
    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info("ADIM 3: Timeseries (Parquet) oluşturma")
    logger.info("=" * 70)
    
    # Arşivde dosya var mı kontrol et
    archive_dir = Path("archive")
    if not archive_dir.exists():
        logger.warning("⚠️  archive/ klasörü yok, oluşturuluyor...")
        archive_dir.mkdir(parents=True, exist_ok=True)
    
    geojson_files = list(archive_dir.glob("flow_*.geojson"))
    if not geojson_files:
        logger.warning("⚠️  archive/ içinde GeoJSON dosyası yok!")
        logger.warning("   İlk fetch yaptıysan bu normal, bir sonraki çalışmada yüklenecek.")
        return False, "NO_ARCHIVE_FILES"
    
    logger.info(f"📁 {len(geojson_files)} adet arşiv dosyası bulundu")
    
    return run_script("05_build_timeseries.py", timeout=300, critical=False)

def step_ensure_topology():
    """3.5. Topoloji kontrolü ve oluşturma (akıllı)"""
    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info("ADIM 3.5: Topoloji Kontrolü (CONNECTS_TO)")
    logger.info("=" * 70)
    
    return run_script("ensure_topology.py", timeout=900, critical=False)

def step_load_neo4j():
    """4. Neo4j'ye yükle"""
    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info("ADIM 4: Neo4j'ye veri yükleme")
    logger.info("=" * 70)
    
    # Gerekli dosyaları kontrol et
    required = [
        Path("data/edges_static.geojson"),
        Path("data/timeseries.parquet")
    ]
    
    missing = [f for f in required if not f.exists()]
    if missing:
        logger.error(f"❌ Eksik dosya(lar): {[str(f) for f in missing]}")
        logger.error("   Önce timeseries oluştur!")
        return False, "MISSING_FILES"
    
    return run_script("07_silent_load_to_neo4j.py", timeout=600, critical=False)

# ---------- Ana Pipeline ----------
def run_full_pipeline(skip_fetch=False):
    """Tüm pipeline'ı çalıştır"""
    logger = logging.getLogger()
    
    start_time = datetime.now()
    logger.info("🚀 " * 20)
    logger.info("🚀 OTOMATIK PIPELINE BAŞLIYOR")
    logger.info(f"🚀 Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🚀 " * 20)
    
    results = {}
    
    # 1. HERE veri çekme (opsiyonel)
    if skip_fetch:
        logger.info("⏭️  ADIM 1 atlanıyor (--skip-fetch)")
        results['fetch'] = (True, "SKIPPED")
    else:
        results['fetch'] = step_fetch_here_flow()
    
    # 2. Harita render (sadece fetch yapıldıysa)
    if not skip_fetch:
        results['render'] = step_render_map()
    else:
        logger.info("⏭️  ADIM 2 atlanıyor (fetch yapılmadı)")
        results['render'] = (True, "SKIPPED")
    
    # 3. Timeseries oluştur
    results['timeseries'] = step_build_timeseries()
    
    # 3.5. Topoloji kontrolü (akıllı - yoksa oluştur)
    results['topology'] = step_ensure_topology()
    
    # 4. Neo4j'ye yükle
    results['neo4j'] = step_load_neo4j()
    
    # Özet
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("📊 PIPELINE SONUÇLARI")
    logger.info("=" * 70)
    logger.info(f"⏱️  Toplam süre: {duration:.1f} saniye")
    logger.info("")
    logger.info("Adım sonuçları:")
    
    all_success = True
    for step_name, (success, _) in results.items():
        status = "✅ BAŞARILI" if success else "❌ HATA"
        logger.info(f"  {step_name:12} → {status}")
        if not success:
            all_success = False
    
    logger.info("=" * 70)
    
    if all_success:
        logger.info("🎉 TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!")
    else:
        logger.warning("⚠️  Bazı adımlarda hata oluştu (yukarıdaki logları incele)")
    
    logger.info("=" * 70)
    
    return all_success

# ---------- Loop Modu ----------
def run_loop_mode(interval_minutes):
    """Belirli aralıklarla pipeline'ı sürekli çalıştır"""
    logger = logging.getLogger()
    
    logger.info("🔄 " * 20)
    logger.info(f"🔄 LOOP MODU AKTIF: Her {interval_minutes} dakikada bir çalışacak")
    logger.info("🔄 Durdurmak için: Ctrl + C")
    logger.info("🔄 " * 20)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# İTERASYON {iteration}")
            logger.info(f"{'#' * 70}\n")
            
            run_full_pipeline(skip_fetch=False)
            
            logger.info(f"\n⏸️  {interval_minutes} dakika bekleniyor...")
            logger.info(f"   Sonraki çalışma: {datetime.now().strftime('%H:%M:%S')} + {interval_minutes} dk\n")
            
            time.sleep(interval_minutes * 60)
    
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Pipeline kullanıcı tarafından durduruldu (Ctrl+C)")
        logger.info(f"📊 Toplam {iteration} iterasyon tamamlandı")
        sys.exit(0)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="HERE Traffic → Neo4j tam otomatik pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Tek seferlik çalıştır:
  python 08_auto_pipeline.py

  # Her 15 dakikada bir sürekli çalıştır:
  python 08_auto_pipeline.py --loop --interval 15

  # Sadece mevcut arşivleri yükle (HERE çekme):
  python 08_auto_pipeline.py --skip-fetch

  # Detaylı log:
  python 08_auto_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Sürekli çalışma modu (belirtilen aralıklarla tekrar eder)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=None,
        help='Loop modunda bekleme süresi (dakika). Belirtilmezse .env dosyasındaki PIPELINE_INTERVAL_MIN kullanılır'
    )
    
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='HERE veri çekmeyi atla (sadece mevcut arşivleri yükle)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Detaylı log çıktısı (DEBUG seviye)'
    )
    
    args = parser.parse_args()
    
    # Logging setup
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger()
    
    # Interval'i .env'den veya argümandan al
    if args.interval is not None:
        interval = args.interval
    else:
        interval = get_env_int("PIPELINE_INTERVAL_MIN", 15)
    
    logger.info(f"📝 Pipeline interval: {interval} dakika (.env: PIPELINE_INTERVAL_MIN)")
    
    # Pipeline çalıştır
    if args.loop:
        run_loop_mode(interval)
    else:
        success = run_full_pipeline(skip_fetch=args.skip_fetch)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
