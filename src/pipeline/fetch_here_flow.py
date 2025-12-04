# 01_fetch_here_flow.py — HERE Traffic v7 Flow fetcher (sağlam .env loader)
# Usage:
#   python 01_fetch_here_flow.py

import os, json, sys
from pathlib import Path
import requests

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.parent

def load_env():
    p = ROOT_DIR / "config" / ".env"
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

def call_flow(api_key, bbox, locref="shape", timeout=30):
    base = "https://data.traffic.hereapi.com/v7/flow"
    params = {
        "in": f"bbox:{bbox}",               # lon_min,lat_min,lon_max,lat_max
        "locationReferencing": locref,      # shape | olr | topology
        "apiKey": api_key,
    }
    # Opsiyonel parametreler (.env)
    units = os.environ.get("UNITS")
    if units: params["units"] = units
    tm = os.environ.get("TRANSPORT_MODE")
    if tm: params["transportMode"] = tm
    fc = os.environ.get("FUNCTIONAL_CLASS")
    if fc: params["functionalClass"] = fc
    adv = os.environ.get("ADVANCED_FEATURES")
    if adv: params["advancedFeatures"] = adv

    r = requests.get(base, params=params, timeout=timeout)
    # Hata durumunu net göster
    if r.status_code != 200:
        print(f"HTTP {r.status_code} ERROR\nURL: {r.url}\nBODY: {r.text[:800]}")
        # yine de JSON parse etmeyi deneriz (incelemek için)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text, "status": r.status_code}
    return r.status_code, data, r.url

def main():
    load_env()
    API_KEY = os.environ.get("HERE_API_KEY")
    if not API_KEY:
        print("ERROR: HERE_API_KEY missing")
        sys.exit(1)

    bbox = os.environ.get("BBOX", "").strip()
    if not bbox:
        print("ERROR: BBOX missing (lon_min,lat_min,lon_max,lat_max)")
        sys.exit(1)

    locref = os.environ.get("LOCATION_REF", "shape").strip()
    timeout = int(os.environ.get("TIMEOUT", "30"))
    out_json = ROOT_DIR / "data" / "raw" / os.environ.get("OUTPUT_JSON", "here_flow_raw.json")

    print("Requesting HERE Traffic Flow v7 …")
    code, data, url = call_flow(API_KEY, bbox, locref=locref, timeout=timeout)
    print(f"HTTP {code} | {url}")

    # Sonuç sayısını yaz
    results = data.get("results") if isinstance(data, dict) else None
    n = len(results) if isinstance(results, list) else 0
    print(f"results: {n}")

    # Dosyaya yaz
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_json}")

    if code != 200 or n == 0:
        print("\nHints:")
        print("- .env satır içi yorumları loader artık siliyor; yine de değerlerin temiz olduğuna bak")
        print("- BBOX şehir merkezini kapsıyor mu?")
        print("- HERE projesinde Traffic yetkisi açık mı?")
        print("- Gerekirse ADVANCED_FEATURES=deepCoverage,lanes")

if __name__ == "__main__":
    main()
