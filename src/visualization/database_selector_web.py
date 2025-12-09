#!/usr/bin/env python3
"""
database_selector_web.py
------------------------
Web tabanlı veritabanı seçim arayüzü
Ana sayfa: http://localhost:3000
"""
import os
import sys
import json
import subprocess
import psutil
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from dotenv import load_dotenv

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

app = Flask(__name__)

# Çalışan viewer process'leri
running_viewers = {}

# Viewer path (aynı klasörde)
VIEWER_DIR = Path(__file__).parent

# PID dosyaları için klasör
PID_DIR = VIEWER_DIR / ".pids"
PID_DIR.mkdir(exist_ok=True)

# Database konfigürasyonları
DATABASES = {
    "neo4j": {
        "name": "Neo4j",
        "script": "neo4j_viewer.py",
        "port": 5000,
        "color": "#008cc1",
        "icon": "🔵",
        "description": "Graph database - Trafik ağı grafiği"
    },
    "arangodb": {
        "name": "ArangoDB",
        "script": "arangodb_viewer.py",
        "port": 5001,
        "color": "#00a86b",
        "icon": "🟢",
        "description": "Multi-model database - Esnek veri modeli"
    },
    "tigergraph": {
        "name": "TigerGraph",
        "script": "tigergraph_viewer.py",
        "port": 5002,
        "color": "#ff6b35",
        "icon": "🟠",
        "description": "Scalable graph database - Yüksek performans"
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Veritabanı Görselleştirme Seçici</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            min-height: 100vh;
            padding: 40px 20px;
            transition: background-color 0.3s ease;
        }
        
        body.light-mode {
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            margin-bottom: 50px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .header-content {
            flex: 1;
        }
        
        .header h1 {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
            transition: color 0.3s ease;
        }
        
        body.light-mode .header h1 {
            color: #0a0a0a;
        }
        
        .header p {
            color: #888;
            font-size: 0.95rem;
            font-weight: 400;
            transition: color 0.3s ease;
        }
        
        body.light-mode .header p {
            color: #666;
        }
        
        .theme-toggle {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        
        .theme-toggle:hover {
            background: #2a2a2a;
            border-color: #3a3a3a;
            transform: scale(1.05);
        }
        
        body.light-mode .theme-toggle {
            background: #ffffff;
            border-color: #e0e0e0;
        }
        
        body.light-mode .theme-toggle:hover {
            background: #f5f5f5;
            border-color: #d0d0d0;
        }
        
        .content {
            /* no padding needed */
        }
        
        .db-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }
        
        .db-card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 28px;
            transition: all 0.2s ease;
            position: relative;
            opacity: 0;
            animation: fadeInUp 0.5s ease forwards;
        }
        
        .db-card:nth-child(1) { animation-delay: 0.1s; }
        .db-card:nth-child(2) { animation-delay: 0.2s; }
        .db-card:nth-child(3) { animation-delay: 0.3s; }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .db-card:hover {
            border-color: #3a3a3a;
            background: #1e1e1e;
            transform: translateY(-2px);
        }
        
        body.light-mode .db-card {
            background: #ffffff;
            border-color: #e0e0e0;
        }
        
        body.light-mode .db-card:hover {
            border-color: #d0d0d0;
            background: #fafafa;
        }
        
        .db-icon {
            width: 48px;
            height: 48px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            margin-bottom: 20px;
        }
        
        .db-card.neo4j .db-icon {
            background: rgba(0, 136, 204, 0.15);
            color: #0088cc;
        }
        
        .db-card.arangodb .db-icon {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
        }
        
        .db-card.tigergraph .db-icon {
            background: rgba(249, 115, 22, 0.15);
            color: #f97316;
        }
        
        .db-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
            transition: color 0.3s ease;
        }
        
        body.light-mode .db-name {
            color: #0a0a0a;
        }
        
        .db-description {
            font-size: 0.875rem;
            color: #666;
            margin-bottom: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
            transition: color 0.3s ease;
        }
        
        body.light-mode .db-description {
            color: #888;
        }
        
        .db-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 16px;
        }
        
        .db-status::before {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }
        
        .db-status.running {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
        }
        
        .db-status.running::before {
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }
        
        .db-status.stopped {
            background: rgba(100, 100, 100, 0.1);
            color: #888;
        }
        
        .db-status.stopped::before {
            background: #555;
        }
        
        .db-button {
            width: 100%;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 500;
            background: #ffffff;
            color: #0f0f0f;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
            position: relative;
        }
        
        .db-button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(0, 0, 0, 0.1);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .db-button:active::after {
            width: 300px;
            height: 300px;
        }
        
        .db-button:hover:not(:disabled) {
            background: #f5f5f5;
            transform: translateY(-1px);
        }
        
        .db-button:active:not(:disabled) {
            transform: translateY(0);
        }
        
        .db-button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        
        .db-button.loading {
            pointer-events: none;
            position: relative;
        }
        
        .db-button.loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 0;
            background: rgba(0, 0, 0, 0.08);
            animation: progressFill 2s ease-out infinite;
            border-radius: 8px;
        }
        
        @keyframes progressFill {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        .db-button.loading span {
            position: relative;
            z-index: 1;
            opacity: 0.8;
        }
        
        body.light-mode .db-button {
            background: #0a0a0a;
            color: #ffffff;
        }
        
        body.light-mode .db-button:hover:not(:disabled) {
            background: #1a1a1a;
        }
        
        body.light-mode .db-button.loading::after {
            background: rgba(255, 255, 255, 0.15);
        }
        
        body.light-mode .db-button.loading {
            background: #1a1a1a;
        }
        
        .db-button-stop {
            width: 100%;
            padding: 12px 20px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 500;
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
            cursor: pointer;
            transition: all 0.2s ease;
            margin-top: 10px;
            font-family: inherit;
        }
        
        .db-button-stop:hover:not(:disabled) {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.3);
            transform: translateY(-1px);
        }
        
        .db-button-stop:active:not(:disabled) {
            transform: translateY(0);
        }
        
        .db-button-stop:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        
        .db-button-stop.loading {
            pointer-events: none;
            position: relative;
        }
        
        .db-button-stop.loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 0;
            background: rgba(239, 68, 68, 0.2);
            animation: progressFill 2s ease-out infinite;
            border-radius: 8px;
        }
        
        .db-button-stop.loading span {
            position: relative;
            z-index: 1;
            opacity: 0.8;
        }
        
        .status-bar {
            background: rgba(26, 26, 26, 0.5);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #2a2a2a;
        }
        
        body.light-mode .status-bar {
            background: rgba(255, 255, 255, 0.8);
            border-color: #e0e0e0;
        }
        
        .status-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #ffffff;
        }
        
        body.light-mode .status-title {
            color: #0a0a0a;
        }
        
        .status-message {
            color: #888;
            font-size: 0.95em;
        }
        
        body.light-mode .status-message {
            color: #666;
        }
        
        .running-services {
            margin-top: 10px;
            padding: 10px;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 5px;
            font-size: 0.9em;
            color: #10b981;
            display: none;
        }
        
        body.light-mode .running-services {
            background: rgba(16, 185, 129, 0.15);
            color: #059669;
        }
        
        .footer {
            text-align: center;
            color: #555;
            font-size: 0.875rem;
            margin-top: 40px;
            transition: color 0.3s ease;
        }
        
        body.light-mode .footer {
            color: #999;
        }
        
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                gap: 16px;
            }
            
            .db-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>Trafik Veritabanı Görüntüleyici</h1>
                <p>Gerçek zamanlı trafik verilerini görselleştirmek için bir veritabanı seçin</p>
            </div>
            <div class="theme-toggle" onclick="toggleTheme()" title="Tema değiştir">
                <span id="theme-icon">☀️</span>
            </div>
        </div>
        
        <div class="content">
            <div class="db-grid">
                {% for db_id, db in databases.items() %}
                <div class="db-card {{ db_id }}" id="card-{{ db_id }}">
                    <div class="db-icon">{{ db.icon }}</div>
                    <div class="db-name">{{ db.name }}</div>
                    <div class="db-description">Port {{ db.port }}</div>
                    <div class="db-status stopped" id="status-{{ db_id }}">Durduruldu</div>
                    <button class="db-button" onclick="startViewer('{{ db_id }}', {{ db.port }})" id="btn-{{ db_id }}">
                        <span>Görüntüleyiciyi Başlat</span>
                    </button>
                    <button class="db-button-stop" onclick="stopViewer('{{ db_id }}', {{ db.port }})" id="btn-stop-{{ db_id }}" style="display: none;">
                        Durdur
                    </button>
                </div>
                {% endfor %}
            </div>
            
            <div class="status-bar">
                <div class="status-title">Durum</div>
                <div class="status-message" id="status-message">Hazır - Bir veritabanı seçin</div>
                <div class="running-services" id="running-services" style="display: none;"></div>
            </div>
        </div>
        
        <div class="footer">
            Veritabanı Trafik Görselleştirme Sistemi v1.0
        </div>
    </div>
    
    <script>
        function toggleTheme() {
            const body = document.body;
            const icon = document.getElementById('theme-icon');
            const isLight = body.classList.toggle('light-mode');
            icon.textContent = isLight ? '🌙' : '☀️';
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        }
        
        // Load saved theme
        if (localStorage.getItem('theme') === 'dark') {
            document.body.classList.add('dark-mode');
            document.getElementById('theme-icon').textContent = '☀️';
        }
        
        // Sayfa yüklendiğinde durumu kontrol et
        window.onload = function() {
            checkStatus();
            // Her 3 saniyede bir durumu kontrol et
            setInterval(checkStatus, 3000);
        };
        
        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateUI(data);
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                });
        }
        
        function updateUI(data) {
            const runningServices = [];
            
            for (const [dbId, info] of Object.entries(data.running)) {
                const statusEl = document.getElementById('status-' + dbId);
                const btnEl = document.getElementById('btn-' + dbId);
                const btnStopEl = document.getElementById('btn-stop-' + dbId);
                
                if (info.running) {
                    statusEl.textContent = 'Çalışıyor (:' + info.port + ')';
                    statusEl.className = 'db-status running';
                    btnEl.innerHTML = '<span>Tarayıcıda Aç</span>';
                    btnStopEl.style.display = 'block';
                    runningServices.push(data.databases[dbId].name + ' (:' + info.port + ')');
                } else {
                    statusEl.textContent = 'Durduruldu';
                    statusEl.className = 'db-status stopped';
                    btnEl.innerHTML = '<span>Görüntüleyiciyi Başlat</span>';
                    btnStopEl.style.display = 'none';
                }
            }
            
            const runningServicesEl = document.getElementById('running-services');
            if (runningServices.length > 0) {
                runningServicesEl.style.display = 'block';
                runningServicesEl.innerHTML = '<strong>Çalışan Servisler:</strong> ' + runningServices.join(', ');
            } else {
                runningServicesEl.style.display = 'none';
            }
        }
        
        function startViewer(dbId, port) {
            const btnEl = document.getElementById('btn-' + dbId);
            const statusMsgEl = document.getElementById('status-message');
            
            // Add loading state
            btnEl.classList.add('loading');
            btnEl.disabled = true;
            
            // Check if already running
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.running[dbId].running) {
                        // Already running, just open
                        btnEl.classList.remove('loading');
                        btnEl.disabled = false;
                        statusMsgEl.textContent = 'Tarayıcıda açılıyor...';
                        window.open('http://localhost:' + port, '_blank');
                        return;
                    }
                    
                    // Start
                    statusMsgEl.textContent = data.databases[dbId].name + ' başlatılıyor...';
                    
                    fetch('/api/start', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ db: dbId })
                    })
                    .then(response => response.json())
                    .then(result => {
                        btnEl.classList.remove('loading');
                        btnEl.disabled = false;
                        
                        if (result.success) {
                            statusMsgEl.textContent = result.message;
                            
                            // 2 saniye sonra tarayıcıda aç
                            setTimeout(() => {
                                window.open('http://localhost:' + port, '_blank');
                            }, 2000);
                            
                            // Durumu güncelle
                            setTimeout(checkStatus, 1000);
                        } else {
                            statusMsgEl.textContent = 'Hata: ' + result.message;
                        }
                    })
                    .catch(error => {
                        btnEl.classList.remove('loading');
                        btnEl.disabled = false;
                        statusMsgEl.textContent = 'Hata: ' + error.message;
                    });
                });
        }
        
        function stopViewer(dbId, port) {
            const btnStopEl = document.getElementById('btn-stop-' + dbId);
            const statusMsgEl = document.getElementById('status-message');
            
            if (!confirm('Bu servisi durdurmak istediğinizden emin misiniz?')) {
                return;
            }
            
            btnStopEl.disabled = true;
            btnStopEl.classList.add('loading');
            btnStopEl.innerHTML = '<span>Durduruluyor...</span>';
            statusMsgEl.textContent = 'Servis durduruluyor...';
            
            fetch('/api/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ db: dbId })
            })
            .then(response => response.json())
            .then(result => {
                btnStopEl.disabled = false;
                btnStopEl.classList.remove('loading');
                btnStopEl.innerHTML = '<span>Durdur</span>';
                
                if (result.success) {
                    statusMsgEl.textContent = result.message;
                    // Birkaç kez status kontrolü yap - port kapanması zaman alabilir
                    setTimeout(checkStatus, 500);
                    setTimeout(checkStatus, 1500);
                    setTimeout(checkStatus, 3000);
                } else {
                    statusMsgEl.textContent = 'Hata: ' + result.message;
                }
            })
            .catch(error => {
                btnStopEl.disabled = false;
                btnStopEl.classList.remove('loading');
                btnStopEl.innerHTML = '<span>Durdur</span>';
                statusMsgEl.textContent = 'Hata: ' + error.message;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template_string(HTML_TEMPLATE, databases=DATABASES)

@app.route('/api/status')
def get_status():
    """Çalışan viewer'ların durumunu döndür"""
    status = {
        "databases": DATABASES,
        "running": {}
    }
    
    for db_id, db_info in DATABASES.items():
        port = db_info["port"]
        is_running = is_port_in_use(port)
        
        status["running"][db_id] = {
            "running": is_running,
            "port": port
        }
    
    return jsonify(status)

@app.route('/api/start', methods=['POST'])
def start_viewer():
    """Viewer'ı başlat"""
    try:
        data = request.json
        db_id = data.get('db')
        
        if db_id not in DATABASES:
            return jsonify({
                "success": False,
                "message": f"Geçersiz veritabanı: {db_id}"
            })
        
        db_info = DATABASES[db_id]
        script_path = VIEWER_DIR / db_info["script"]
        
        if not script_path.exists():
            return jsonify({
                "success": False,
                "message": f"Script bulunamadı: {script_path}"
            })
        
        # Already running?
        if is_port_in_use(db_info['port']):
            return jsonify({
                "success": True,
                "message": f"{db_info['name']} is already running"
            })
        
        # Process başlat
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(script_path.parent),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        running_viewers[db_id] = process
        
        # PID'yi dosyaya kaydet
        pid_file = PID_DIR / f"{db_id}.pid"
        try:
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            print(f"[START] Process PID {process.pid} saved to {pid_file}")
        except Exception as e:
            print(f"[START] Warning: Could not save PID: {e}")
        
        return jsonify({
            "success": True,
            "message": f"{db_info['name']} başarıyla başlatıldı! Port: {db_info['port']}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_viewer():
    """Viewer'ı durdur - %100 garantili versiyon"""
    try:
        data = request.json
        db_id = data.get('db')
        
        if db_id not in DATABASES:
            return jsonify({
                "success": False,
                "message": f"Geçersiz veritabanı: {db_id}"
            })
        
        db_info = DATABASES[db_id]
        port = db_info["port"]
        
        print(f"\n{'='*60}")
        print(f"[STOP] Stopping {db_info['name']} on port {port}")
        print(f"{'='*60}")
        
        # PID listesi - tüm PID'leri topla
        pids_to_kill = set()
        
        # 1. PID dosyasından PID'yi oku
        pid_file = PID_DIR / f"{db_id}.pid"
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    file_pid = int(f.read().strip())
                pids_to_kill.add(file_pid)
                print(f"[STOP] [OK] Found PID {file_pid} from PID file")
            except Exception as e:
                print(f"[STOP] [ERROR] Error reading PID file: {e}")
        
        # 2. Tracked process'ten PID al
        if db_id in running_viewers:
            try:
                tracked_pid = running_viewers[db_id].pid
                pids_to_kill.add(tracked_pid)
                print(f"[STOP] [OK] Found PID {tracked_pid} from tracked process")
            except Exception as e:
                print(f"[STOP] [ERROR] Error getting tracked PID: {e}")
        
        # 3. Port'u kullanan tüm PID'leri bul
        try:
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        port_pid = int(parts[-1])
                        pids_to_kill.add(port_pid)
                        print(f"[STOP] [OK] Found PID {port_pid} using port {port}")
        except Exception as e:
            print(f"[STOP] [ERROR] Error finding PIDs via netstat: {e}")
        
        # 4. Tüm PID'leri KILL ET - En agresif yöntem
        print(f"\n[STOP] Killing {len(pids_to_kill)} process(es)...")
        for pid in pids_to_kill:
            try:
                # /T ile child process'leri de kapat, /F ile force kill
                result = subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"[STOP]   [OK] PID {pid} killed successfully")
                else:
                    # Hata mesajını kontrol et
                    if "not found" not in result.stderr.lower():
                        print(f"[STOP]   [WARNING] PID {pid}: {result.stderr.strip()}")
                    else:
                        print(f"[STOP]   [INFO] PID {pid} already terminated")
            except Exception as e:
                print(f"[STOP]   [ERROR] PID {pid} kill error: {e}")
        
        # 5. Tracked process'i temizle
        if db_id in running_viewers:
            try:
                del running_viewers[db_id]
                print(f"[STOP] [OK] Removed from tracked processes")
            except Exception as e:
                print(f"[STOP] [ERROR] Error removing tracked process: {e}")
        
        # 6. PID dosyasını sil
        if pid_file.exists():
            try:
                pid_file.unlink()
                print(f"[STOP] [OK] PID file removed")
            except Exception as e:
                print(f"[STOP] [ERROR] Error removing PID file: {e}")
        
        # 7. Port'un kapanmasını bekle - MAKSIMUM 10 SANİYE
        import time
        print(f"\n[STOP] Waiting for port {port} to be freed...")
        
        port_freed = False
        max_attempts = 20  # 20 * 0.5 = 10 saniye
        
        for attempt in range(max_attempts):
            time.sleep(0.5)
            
            # Port kontrol et
            if not is_port_in_use(port):
                port_freed = True
                print(f"[STOP] [OK] Port {port} is FREE (after {(attempt+1)*0.5:.1f}s)")
                break
            else:
                if attempt % 2 == 0:  # Her 1 saniyede bir log
                    print(f"[STOP]   [WAIT] Still waiting... ({(attempt+1)*0.5:.1f}s)")
        
        # 8. Hala port kullanılıyorsa SON BİR DENEME
        if not port_freed:
            print(f"\n[STOP] [WARNING] Port {port} still in use after 10s. Final attempt...")
            
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        stubborn_pid = int(parts[-1])
                        print(f"[STOP] [KILL] Found stubborn PID {stubborn_pid}, force killing...")
                        try:
                            # ÇOK AGRESIF - tüm child process'lerle birlikte
                            subprocess.run(
                                ['taskkill', '/F', '/T', '/PID', str(stubborn_pid)],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                timeout=5
                            )
                            time.sleep(2)
                            
                            # Son kontrol
                            if not is_port_in_use(port):
                                port_freed = True
                                print(f"[STOP] [OK] Stubborn process killed, port is FREE")
                            else:
                                print(f"[STOP] [ERROR] Port STILL in use even after aggressive kill")
                        except Exception as e:
                            print(f"[STOP] [ERROR] Final kill attempt failed: {e}")
                        break
        
        # 9. Sonuç
        print(f"\n{'='*60}")
        print(f"[STOP] RESULT: Port {port} is {'FREE [OK]' if port_freed else 'STILL IN USE [ERROR]'}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "success": True,
            "message": f"{db_info['name']} {'stopped successfully' if port_freed else 'stopped (port still in use)'}",
            "port_free": port_freed
        })
        
    except Exception as e:
        print(f"\n[STOP] [ERROR][ERROR][ERROR] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

def is_port_in_use(port):
    """Portun kullanımda olup olmadığını kontrol et"""
    try:
        # Windows için netstat kullan - daha güvenilir
        import subprocess as sp
        result = sp.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                return True
        return False
    except Exception as e:
        print(f"[DEBUG] Error checking port {port}: {e}")
        return False

if __name__ == '__main__':
    print("="*70)
    print("  VERITABANI SECIM ARAYUZU")
    print("="*70)
    print()
    print("Tarayicinizda acin: http://localhost:3001")
    print()
    print("Mevcut veritabanlari:")
    for db_id, db_info in DATABASES.items():
        print(f"  {db_info['name']:12} -> Port {db_info['port']}")
    print()
    print("Not: Bu sayfa icin once login_page.py ile giris yapmalisiniz!")
    print("     Login sayfasi: http://localhost:3000")
    print()
    print("Durdurmak icin: Ctrl+C")
    print("="*70)
    print()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=3001)
    except KeyboardInterrupt:
        print("\n\nDurduruluyor...")
        # Çalışan viewer'ları durdur
        for db_id, process in running_viewers.items():
            try:
                process.terminate()
                process.wait(timeout=3)
            except:
                try:
                    process.kill()
                except:
                    pass
        print("All services stopped.")
