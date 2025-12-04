# 04_run_loop.py — 15 dk'da bir fetch + render + arşiv (sağlam .env okuma)
# Kullanım:
#   python 04_run_loop.py
# Durdurmak için: Ctrl + C
import os, subprocess, time, shlex
from pathlib import Path
from datetime import datetime

def load_env():
    p = Path(".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Satır içi yorumları (# ...) kaldır
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def getenv_clean(key, default=None):
    """Env değeri varsa al; satır içi # yorumlarını ve dış boşlukları at."""
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip()
    if "#" in val:
        val = val.split("#", 1)[0].strip()
    return val if val != "" else default

def get_int_env(key, default):
    val = getenv_clean(key, None)
    if val is None:
        return int(default)
    try:
        return int(val)
    except ValueError:
        # sadece sayısal kısmı almaya çalış (örn: "15 min")
        try:
            return int("".join(ch for ch in val if ch.isdigit()) or default)
        except Exception:
            print(f"WARN: {key}='{val}' int'e çevrilemedi; {default} kullanılıyor.")
            return int(default)

def pybin():
    return getenv_clean("PYTHON_BIN", "python")

def run(cmd):
    print(">", " ".join(shlex.quote(c) for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"Command failed (exit {r.returncode}):", " ".join(cmd))
    return r.returncode

def main():
    load_env()
    interval_min = get_int_env("SNAPSHOT_INTERVAL_MIN", 15)
    out_json = getenv_clean("OUTPUT_JSON", "here_flow_raw.json")
    html_out = "map.html"

    print(f"Loop interval: {interval_min} min")
    while True:
        print("\n=== SNAPSHOT START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "===")
        rc1 = run([pybin(), "01_fetch_here_flow.py"])
        rc2 = run([pybin(), "02_render_flow_map.py", out_json, html_out])
        print("=== SNAPSHOT END ===")

        # Hata olsa da devam etsin
        time.sleep(max(1, interval_min) * 60)

if __name__ == "__main__":
    main()
