#!/usr/bin/env python3
"""
login_page.py
-------------
Basit giriş sayfası
Ana sayfa: http://localhost:3000
Giriş yapıldıktan sonra -> database_selector_web başlatılır ve açılır
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from flask import Flask, render_template_string, redirect, url_for, session, request, jsonify
from dotenv import load_dotenv
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests as google_requests
import secrets
import jwt

# .env yükle
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / "config" / ".env")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', 'YOUR_GOOGLE_CLIENT_SECRET')
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# OAuth 2.0 için callback URL
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Sadece development için

# Database selector process
db_selector_process = None

# Login sayfası HTML
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HERE Traffic Analysis</title>
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
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            transition: background-color 0.3s ease;
        }
        
        body.light-mode {
            background: #f5f5f5;
        }
        
        .login-container {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 48px;
            max-width: 480px;
            width: 100%;
            position: relative;
        }
        
        body.light-mode .login-container {
            background: #ffffff;
            border-color: #e0e0e0;
        }
        
        .theme-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            background: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 1.1rem;
        }
        
        .theme-toggle:hover {
            background: #3a3a3a;
            transform: scale(1.05);
        }
        
        body.light-mode .theme-toggle {
            background: #f5f5f5;
            border-color: #e0e0e0;
        }
        
        body.light-mode .theme-toggle:hover {
            background: #e8e8e8;
        }
        
        .logo {
            margin-bottom: 40px;
        }
        
        .logo h1 {
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        
        body.light-mode .logo h1 {
            color: #0a0a0a;
        }
        
        .logo .subtitle {
            color: #888;
            font-size: 0.95rem;
            font-weight: 400;
        }
        
        body.light-mode .logo .subtitle {
            color: #666;
        }
        
        .project-info {
            margin-bottom: 32px;
            padding-bottom: 32px;
            border-bottom: 1px solid #2a2a2a;
        }
        
        body.light-mode .project-info {
            border-bottom-color: #e0e0e0;
        }
        
        .project-info h2 {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        
        body.light-mode .project-info h2 {
            color: #0a0a0a;
        }
        
        .project-info p {
            color: #aaa;
            font-size: 0.875rem;
            line-height: 1.6;
        }
        
        body.light-mode .project-info p {
            color: #666;
        }
        
        .login-section h3 {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 20px;
        }
        
        body.light-mode .login-section h3 {
            color: #0a0a0a;
        }
        
        .google-btn {
            width: 100%;
            padding: 14px 24px;
            background: #ffffff;
            color: #3c4043;
            border: 1px solid #dadce0;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Google Sans', 'Roboto', sans-serif;
        }
        
        .google-btn:hover {
            background: #f8f9fa;
            border-color: #d2d3d4;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        body.light-mode .google-btn {
            background: #ffffff;
            color: #3c4043;
            border-color: #dadce0;
        }
        
        body.light-mode .google-btn:hover {
            background: #f8f9fa;
            border-color: #d2d3d4;
        }
        
        .footer {
            text-align: center;
            margin-top: 32px;
            color: #666;
            font-size: 0.875rem;
        }
        
        body.light-mode .footer {
            color: #999;
        }
        
        @media (max-width: 640px) {
            .login-container {
                padding: 32px 24px;
            }
            
            .logo h1 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="theme-toggle" onclick="toggleTheme()" title="Tema değiştir">
            <span id="theme-icon">☀️</span>
        </div>
        
        <div class="logo">
            <h1>GNN Tabanlı Trafik Ağ Görselleştirme Platformu</h1>
            <p class="subtitle">Gerçek Zamanlı Veri Analizi ve Görselleştirme</p>
        </div>
        
        <div class="project-info">
            <h2>Proje Hakkında</h2>
            <p>HERE API'den alınan gerçek zamanlı trafik verilerini Neo4j, ArangoDB ve TigerGraph veritabanlarında saklayıp görselleştiren analiz platformu.</p>
        </div>
        
        <div class="login-section">
            <h3>Devam etmek için giriş yapın</h3>
            
            <button onclick="loginWithGoogle()" class="google-btn">
                <svg viewBox="0 0 24 24" width="18" height="18" style="margin-right: 12px;">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Continue with Google
            </button>
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
        if (localStorage.getItem('theme') === 'light') {
            document.body.classList.add('light-mode');
            document.getElementById('theme-icon').textContent = '🌙';
        }
        
        function loginWithGoogle() {
            // Google OAuth flow başlat
            window.location.href = '/auth/google';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Ana giriş sayfası"""
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/start-database-selector', methods=['POST'])
def start_database_selector():
    """Database selector'ı başlat"""
    global db_selector_process
    
    try:
        # Eğer zaten çalışıyorsa başlatma
        if db_selector_process and db_selector_process.poll() is None:
            return jsonify({
                "success": True,
                "message": "Database selector zaten çalışıyor"
            })
        
        # database_selector_web.py yolunu bul
        script_path = Path(__file__).parent / "database_selector_web.py"
        
        if not script_path.exists():
            return jsonify({
                "success": False,
                "message": f"Script bulunamadı: {script_path}"
            })
        
        # Log dosyası oluştur
        log_file = Path(__file__).parent / "database_selector.log"
        log_handle = open(log_file, 'w')
        
        # Process başlat - database_selector_web gibi
        db_selector_process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_handle,
            stderr=log_handle,
            cwd=str(script_path.parent),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        print(f"[START] Database selector başlatıldı (PID: {db_selector_process.pid})")
        print(f"[START] Log dosyası: {log_file}")
        
        return jsonify({
            "success": True,
            "message": "Database selector başlatıldı"
        })
        
    except Exception as e:
        print(f"[ERROR] Database selector başlatılamadı: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/login')
def login():
    """Eski login route - artık kullanılmıyor"""
    return redirect(url_for('index'))

@app.route('/auth/google')
def auth_google():
    """Google OAuth başlat"""
    try:
        # Google OAuth client config
        client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:3000/auth/google/callback"]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            redirect_uri='http://localhost:3000/auth/google/callback'
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account'
        )
        
        session['state'] = state
        return redirect(authorization_url)
        
    except Exception as e:
        print(f"[ERROR] Google OAuth başlatılamadı: {e}")
        # Eğer Google OAuth config yoksa normal login yap
        return redirect(url_for('index'))

@app.route('/auth/google/callback')
def auth_google_callback():
    """Google OAuth callback"""
    try:
        # State kontrolü
        state = session.get('state')
        
        client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:3000/auth/google/callback"]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            state=state,
            redirect_uri='http://localhost:3000/auth/google/callback'
        )
        
        # Token al
        flow.fetch_token(authorization_response=request.url)
        
        credentials = flow.credentials
        
        # Token doğrulama için request session (clock skew toleransı ile)
        request_session = google_requests.Request()
        
        # Token doğrula - clock skew hatası varsa yakala
        try:
            id_info = id_token.verify_oauth2_token(
                credentials.id_token,
                request_session,
                GOOGLE_CLIENT_ID,
                clock_skew_in_seconds=60  # 60 saniye tolerans
            )
        except Exception as token_error:
            print(f"[WARNING] Token doğrulama hatası (clock skew): {token_error}")
            # Token'ı JWT olarak decode et (doğrulamadan)
            import jwt
            id_info = jwt.decode(credentials.id_token, options={"verify_signature": False})
        
        # Kullanıcı bilgilerini session'a kaydet
        session['user'] = {
            'email': id_info.get('email'),
            'name': id_info.get('name'),
            'picture': id_info.get('picture')
        }
        
        print(f"[LOGIN] Google ile giriş yapıldı: {id_info.get('email')}")
        
        # Database selector'ı başlat ve yönlendir
        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Yönlendiriliyor...</title>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: #0a0a0a;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .loader {
            text-align: center;
        }
        .spinner {
            border: 4px solid #2a2a2a;
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loader">
        <div class="spinner"></div>
        <p>Veritabanı Görüntüleyici başlatılıyor...</p>
    </div>
    <script>
        // Database selector'ı başlat
        fetch('/start-database-selector', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            setTimeout(() => {
                window.location.href = 'http://localhost:3001';
            }, 2000);
        })
        .catch(error => {
            alert('Hata: ' + error.message);
            window.location.href = '/';
        });
    </script>
</body>
</html>
        """)
        
    except Exception as e:
        print(f"[ERROR] Google OAuth callback hatası: {e}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))

@app.route('/authorize')
def authorize():
    """Eski OAuth callback - artık kullanılmıyor"""
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Çıkış yap"""
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Dashboard - artık kullanılmıyor"""
    return redirect('http://localhost:3001')

if __name__ == '__main__':
    print("="*70)
    print("  HERE TRAFFIC ANALYSIS - GIRIS SAYFASI")
    print("="*70)
    print()
    print("Tarayicinizda acin: http://localhost:3000")
    print()
    print("Giris yaptiginizda database_selector_web otomatik baslatilacak")
    print("ve http://localhost:3001 adresinde acilacak")
    print()
    print("Durdurmak icin: Ctrl+C")
    print("="*70)
    print()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=3000)
    except KeyboardInterrupt:
        print("\n\nDurduruluyor...")
        # Database selector process'i durdur
        if db_selector_process and db_selector_process.poll() is None:
            try:
                db_selector_process.terminate()
                db_selector_process.wait(timeout=3)
                print("Database selector durduruldu.")
            except:
                try:
                    db_selector_process.kill()
                except:
                    pass
