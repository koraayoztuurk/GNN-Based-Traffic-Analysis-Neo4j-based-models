# ============================================
# GNN/GCN-Hazır Veri Hattı - Hızlı Başlangıç
# ============================================
# Bu script tüm adımları sırayla çalıştırır

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🧠 GNN/GCN Veri Hattı - Otomatik Kurulum" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Çalışma dizinine geç
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 1. Gerekli paketleri kontrol et
Write-Host "📦 Adım 1/7: Gerekli Python paketlerini kontrol ediliyor..." -ForegroundColor Yellow
$packages = @("neo4j", "pandas", "numpy", "python-dotenv")
$missing = @()

foreach ($pkg in $packages) {
    python -c "import $($pkg.Replace('-', '_'))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host "  ⚠️  Eksik paketler bulundu: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "  🔄 Yükleniyor..." -ForegroundColor Yellow
    pip install $missing
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Paket yükleme başarısız! Manuel olarak yükleyin: pip install $($missing -join ' ')" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  ✅ Tüm paketler hazır" -ForegroundColor Green
Write-Host ""

# 2. .env dosyasını kontrol et
Write-Host "⚙️  Adım 2/7: Yapılandırma dosyası kontrol ediliyor..." -ForegroundColor Yellow
if (-Not (Test-Path "mvp\.env")) {
    Write-Host "  ⚠️  .env dosyası bulunamadı, .env.example kopyalanıyor..." -ForegroundColor Yellow
    Copy-Item "mvp\.env.example" "mvp\.env"
    Write-Host "  ⚠️  UYARI: mvp\.env dosyasını düzenleyip Neo4j şifrenizi girin!" -ForegroundColor Red
    Write-Host "  ⏸️  Devam etmek için Enter'a basın (Neo4j şifresini güncellediyseniz)..." -ForegroundColor Yellow
    Read-Host
}
Write-Host "  ✅ Yapılandırma hazır" -ForegroundColor Green
Write-Host ""

# 3. Neo4j bağlantısını test et
Write-Host "🔌 Adım 3/7: Neo4j bağlantısı test ediliyor..." -ForegroundColor Yellow
python -c @"
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path('mvp/scripts').resolve()))
from dotenv import load_dotenv
load_dotenv('mvp/.env')
from neo4j import GraphDatabase
uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASS', '123456789')
try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run('RETURN 1 AS test')
        result.single()
    driver.close()
    print('  ✅ Neo4j bağlantısı başarılı')
except Exception as e:
    print(f'  ❌ Neo4j bağlantı hatası: {e}')
    sys.exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Neo4j bağlantısı kurulamadı! Lütfen kontrol edin:" -ForegroundColor Red
    Write-Host "     - Neo4j Desktop çalışıyor mu?" -ForegroundColor Yellow
    Write-Host "     - mvp\.env dosyasındaki şifre doğru mu?" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 4. Şema iyileştirmeleri (Manuel)
Write-Host "🔧 Adım 4/7: Neo4j şema iyileştirmeleri" -ForegroundColor Yellow
Write-Host "  📋 Lütfen Neo4j Browser'da aşağıdaki dosyayı çalıştırın:" -ForegroundColor Cyan
Write-Host "     mvp\cypher\01_enhance_schema.cql" -ForegroundColor White
Write-Host ""
Write-Host "  ℹ️  Dosya yolu kopyalandı (Ctrl+V ile yapıştırabilirsiniz)" -ForegroundColor Gray
Set-Clipboard -Value (Resolve-Path "mvp\cypher\01_enhance_schema.cql").Path
Write-Host "  ⏸️  Tamamladıktan sonra Enter'a basın..." -ForegroundColor Yellow
Read-Host
Write-Host "  ✅ Şema iyileştirmeleri tamamlandı (varsayılan)" -ForegroundColor Green
Write-Host ""

# 5. Timeseries import
Write-Host "📊 Adım 5/7: Timeseries verileri yükleniyor..." -ForegroundColor Yellow
python mvp\scripts\03_fix_timeseries_import.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Timeseries import başarısız!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 6. CONNECTS_TO ilişkileri (Manuel)
Write-Host "🔗 Adım 6/7: CONNECTS_TO ilişkileri" -ForegroundColor Yellow
Write-Host "  📋 Lütfen Neo4j Browser'da aşağıdaki dosyayı çalıştırın:" -ForegroundColor Cyan
Write-Host "     mvp\cypher\02_build_connects_to.cql" -ForegroundColor White
Write-Host ""
Write-Host "  ℹ️  Dosya yolu kopyalandı (Ctrl+V ile yapıştırabilirsiniz)" -ForegroundColor Gray
Set-Clipboard -Value (Resolve-Path "mvp\cypher\02_build_connects_to.cql").Path
Write-Host "  ⚠️  NOT: Bu işlem segment sayısına göre 5-30 dakika sürebilir" -ForegroundColor Yellow
Write-Host "  ⏸️  Tamamladıktan sonra Enter'a basın..." -ForegroundColor Yellow
Read-Host
Write-Host "  ✅ CONNECTS_TO ilişkileri tamamlandı (varsayılan)" -ForegroundColor Green
Write-Host ""

# 7. Feature engineering
Write-Host "🧮 Adım 7/7: Feature engineering..." -ForegroundColor Yellow
python mvp\scripts\04_generate_features.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Feature engineering başarısız!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 8. PyG export
Write-Host "📦 Adım 8/7: PyTorch Geometric export..." -ForegroundColor Yellow
python mvp\scripts\05_export_pyg.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ PyG export başarısız!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Özet
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✨ Tüm işlemler tamamlandı!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📂 Oluşturulan dosyalar:" -ForegroundColor Yellow
Write-Host "   ✅ data\features_window.csv" -ForegroundColor Green
Write-Host "   ✅ data\pyg_graph.npz" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Sonraki adımlar:" -ForegroundColor Yellow
Write-Host "   1. Test edin: python mvp\scripts\06_test_pyg_data.py" -ForegroundColor Cyan
Write-Host "   2. GNN modeli geliştirin (PyTorch Geometric / DGL)" -ForegroundColor Cyan
Write-Host "   3. Benchmark için farklı graph store'lar deneyin" -ForegroundColor Cyan
Write-Host ""
Write-Host "📚 Dökümantasyon:" -ForegroundColor Yellow
Write-Host "   - mvp\README.md - Detaylı kullanım kılavuzu" -ForegroundColor Cyan
Write-Host "   - GNN_STATUS_REPORT.md - Durum raporu" -ForegroundColor Cyan
Write-Host ""
