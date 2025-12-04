# ============================================================
# setup_windows_task.ps1
# Windows Task Scheduler'da otomatik görev oluşturur
# ============================================================

# PowerShell 5.1+ gerekir (Windows 10/11'de varsayılan)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Windows Task Scheduler Kurulum Script'i" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Yönetici kontrolü
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[HATA] Bu script'i yönetici (Administrator) olarak çalıştırmalısınız!" -ForegroundColor Red
    Write-Host "Sağ tık -> 'Run as Administrator' seçin" -ForegroundColor Yellow
    pause
    exit 1
}

# Proje dizini
$ProjectDir = $PSScriptRoot
$PythonScript = Join-Path $ProjectDir "08_auto_pipeline.py"
$BatchScript = Join-Path $ProjectDir "start_pipeline.bat"

# Dosya kontrolü
if (-not (Test-Path $PythonScript)) {
    Write-Host "[HATA] $PythonScript bulunamadı!" -ForegroundColor Red
    pause
    exit 1
}

# Python yolu
$PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonPath) {
    Write-Host "[HATA] Python bulunamadı! PATH'e ekleyin ve tekrar deneyin." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[BILGI] Python yolu: $PythonPath" -ForegroundColor Green
Write-Host "[BILGI] Proje dizini: $ProjectDir" -ForegroundColor Green
Write-Host ""

# Kullanıcı seçimi
Write-Host "Hangi çalışma modunu tercih edersiniz?" -ForegroundColor Yellow
Write-Host "1) Her 15 dakikada bir otomatik çalışsın (önerilen)" -ForegroundColor White
Write-Host "2) Günde bir kez çalışsın (belirli saatte)" -ForegroundColor White
Write-Host "3) Bilgisayar başladığında bir kez çalışsın" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Seçiminiz (1/2/3)"

# Görev adı
$TaskName = "HERE_Traffic_Pipeline"

# Mevcut görevi sil (varsa)
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "[UYARI] '$TaskName' adlı görev zaten var, siliniyor..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Trigger oluştur
switch ($choice) {
    "1" {
        # Her 15 dakikada bir
        Write-Host "[BILGI] Her 15 dakikada bir çalışacak şekilde ayarlanıyor..." -ForegroundColor Cyan
        
        $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration ([TimeSpan]::MaxValue)
        
        $ActionArg = "-File `"$PythonScript`""
        $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ActionArg -WorkingDirectory $ProjectDir
        
        Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Description "HERE Traffic verilerini 15 dakikada bir çeker ve Neo4j'ye yükler" -RunLevel Highest
        
        Write-Host "[BAŞARILI] Görev oluşturuldu: Her 15 dakikada bir çalışacak" -ForegroundColor Green
    }
    
    "2" {
        # Günde bir kez
        Write-Host "Saat kaçta çalışsın? (örn: 09:00)" -ForegroundColor Yellow
        $timeInput = Read-Host "Saat"
        
        try {
            $time = [DateTime]::ParseExact($timeInput, "HH:mm", $null)
        } catch {
            Write-Host "[HATA] Geçersiz saat formatı! Örnek: 09:00" -ForegroundColor Red
            pause
            exit 1
        }
        
        $Trigger = New-ScheduledTaskTrigger -Daily -At $time
        $ActionArg = "-File `"$PythonScript`""
        $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ActionArg -WorkingDirectory $ProjectDir
        
        Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Description "HERE Traffic verilerini günde bir kez ($timeInput) çeker ve Neo4j'ye yükler" -RunLevel Highest
        
        Write-Host "[BAŞARILI] Görev oluşturuldu: Her gün saat $timeInput'de çalışacak" -ForegroundColor Green
    }
    
    "3" {
        # Bilgisayar başlangıcında
        $Trigger = New-ScheduledTaskTrigger -AtStartup
        $ActionArg = "-File `"$PythonScript`""
        $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ActionArg -WorkingDirectory $ProjectDir
        
        Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Description "HERE Traffic verilerini bilgisayar başladığında çeker ve Neo4j'ye yükler" -RunLevel Highest
        
        Write-Host "[BAŞARILI] Görev oluşturuldu: Bilgisayar her başladığında çalışacak" -ForegroundColor Green
    }
    
    default {
        Write-Host "[HATA] Geçersiz seçim!" -ForegroundColor Red
        pause
        exit 1
    }
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Görev başarıyla oluşturuldu!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Görevi yönetmek için:" -ForegroundColor Yellow
Write-Host "  1) Windows'ta 'Task Scheduler' uygulamasını açın" -ForegroundColor White
Write-Host "  2) 'Task Scheduler Library' altında '$TaskName' görevini bulun" -ForegroundColor White
Write-Host "  3) Sağ tık -> Run ile manuel test edebilirsiniz" -ForegroundColor White
Write-Host ""
Write-Host "Görevi silmek için:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor White
Write-Host ""

pause
