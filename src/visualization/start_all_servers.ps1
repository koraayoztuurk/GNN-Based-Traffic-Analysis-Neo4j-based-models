# Multi-Database Visualization Launcher
# Bu script 3 ayrı PowerShell penceresinde sunucuları başlatır

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  MULTI-DATABASE TRAFFIC VISUALIZATION LAUNCHER" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting 3 Flask servers in separate windows..." -ForegroundColor Yellow
Write-Host ""

# Neo4j Server (Port 5000)
Write-Host "[1/3] Starting Neo4j visualization server..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath'; python neo4j_viewer.py"
Start-Sleep -Seconds 2

# ArangoDB Server (Port 5001)
Write-Host "[2/3] Starting ArangoDB visualization server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath'; python arangodb_viewer.py"
Start-Sleep -Seconds 2

# TigerGraph Server (Port 5002)
Write-Host "[3/3] Starting TigerGraph visualization server..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath'; python tigergraph_viewer.py"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  ALL SERVERS STARTED!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Open these URLs in your browser:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [Neo4j]      -> http://localhost:5000" -ForegroundColor Blue
Write-Host "  [ArangoDB]   -> http://localhost:5001" -ForegroundColor Green
Write-Host "  [TigerGraph] -> http://localhost:5002" -ForegroundColor Magenta
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop all servers: Close the PowerShell windows or press Ctrl+C in each" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to open browsers..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open browsers
Write-Host ""
Write-Host "Opening browsers..." -ForegroundColor Yellow
Start-Process "http://localhost:5000"
Start-Sleep -Seconds 1
Start-Process "http://localhost:5001"
Start-Sleep -Seconds 1
Start-Process "http://localhost:5002"

Write-Host ""
Write-Host "Done! All servers are running and browsers opened." -ForegroundColor Green
Write-Host "Press any key to exit this launcher window..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
