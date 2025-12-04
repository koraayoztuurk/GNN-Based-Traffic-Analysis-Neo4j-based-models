# 02_render_flow_map.py — HERE Flow JSON -> Leaflet + GeoJSON arşivi (edge_id + length_m + t_utc)
# Kullanım:
#   python 02_render_flow_map.py here_flow_raw.json map.html

import os, json, sys, math, glob, hashlib
from pathlib import Path
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.parent

# ---------- ENV ----------
def load_env():
    p = ROOT_DIR / "config" / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if "#" in line:
                line=line.split("#",1)[0].strip()
            k,v=line.split("=",1)
            os.environ.setdefault(k.strip(), v.strip())

def now_local_str():
    tzname = os.environ.get("TIMEZONE", "Europe/Istanbul")
    try:
        tz = ZoneInfo(tzname) if ZoneInfo else None
    except Exception:
        tz = None
    dt = datetime.now(tz or None)
    return dt.strftime("%Y-%m-%d %H:%M")

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# ---------- HELPERS ----------
def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# ---------- GEO UTILS ----------
R_EARTH = 6371000.0
def haversine(p1, p2):
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_EARTH * c

def polyline_length_m(coords_latlon):
    if len(coords_latlon) < 2: return 0.0
    total = 0.0
    for i in range(1, len(coords_latlon)):
        total += haversine(coords_latlon[i-1], coords_latlon[i])
    return total

def cumulative_lengths(coords):
    L = [0.0]
    for i in range(1, len(coords)):
        L.append(L[-1] + haversine(coords[i-1], coords[i]))
    return L

def interpolate_point(p1, p2, target_dist, seg_start_dist):
    total = haversine(p1, p2)
    if total == 0: return p1
    t = (target_dist - seg_start_dist) / total
    t = max(0.0, min(1.0, t))
    lat = p1[0] + (p2[0]-p1[0]) * t
    lon = p1[1] + (p2[1]-p1[1]) * t
    return (lat, lon)

def cut_polyline_by_lengths(coords, lengths):
    if len(coords) < 2:
        return [coords]
    cum = cumulative_lengths(coords)
    total = cum[-1]
    Ls = list(lengths)
    if total > 0 and abs(sum(Ls) - total) > 0.05 * total:
        scale = total / max(1e-9, sum(Ls))
        Ls = [l * scale for l in Ls]

    pieces = []
    curr_start = 0.0
    idx = 0
    for seg_len in Ls:
        target_end = curr_start + seg_len
        piece = []
        while idx < len(coords)-1 and cum[idx+1] < curr_start:
            idx += 1
        if idx < len(coords)-1:
            piece.append(interpolate_point(coords[idx], coords[idx+1], curr_start, cum[idx]))
        else:
            piece.append(coords[-1])
        while idx < len(coords)-1 and cum[idx+1] <= target_end:
            idx += 1
            piece.append(coords[idx])
        if idx < len(coords)-1:
            piece.append(interpolate_point(coords[idx], coords[idx+1], target_end, cum[idx]))
        else:
            piece.append(coords[-1])
        pieces.append(piece)
        curr_start = target_end
    return pieces

# ---------- EXTRACT ----------
def extract_line_coords(res):
    coords = []
    loc = res.get("location", {})
    shape = loc.get("shape", {})
    links = shape.get("links") or []
    for ln in links:
        pts = ln.get("points") or []
        for pt in pts:
            lat = pt.get("lat"); lon = pt.get("lng")
            if isinstance(lat,(int,float)) and isinstance(lon,(int,float)):
                coords.append((lat, lon))
    compact = []
    for p in coords:
        if not compact or (abs(compact[-1][0]-p[0])>1e-12 or abs(compact[-1][1]-p[1])>1e-12):
            compact.append(p)
    return compact

def feature_line(line_coords, props):
    return {
        "type": "Feature",
        "geometry": {"type":"LineString",
                     "coordinates": [[c[1], c[0]] for c in line_coords]},
        "properties": props
    }

def make_edge_id_from_coords(line_coords):
    # line_coords: [(lat,lon), ...] -> stabilize by rounding, then sha1
    pts = ["{:.5f},{:.5f}".format(lat, lon) for (lat,lon) in line_coords]
    h = hashlib.sha1(("|".join(pts)).encode("utf-8")).hexdigest()
    return f"edge:{h[:16]}"

# ---------- HTML TEMPLATE ----------
TEMPLATE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>HERE Traffic — Eskişehir</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html, body, #map { height: 100%; margin: 0; }
  .hdr { position:absolute; top:10px; left:10px; background:rgba(255,255,255,.9); padding:10px 12px; font:14px/1.3 system-ui; border-radius:8px; z-index:999; }
  #pane { position:absolute; top:10px; right:10px; background:rgba(255,255,255,.9); padding:10px 12px; font:13px system-ui; border-radius:8px; z-index:999; }
  .legend { font:12px system-ui }
  .legend .box { background:#fff; padding:8px 10px; border-radius:6px }
  .legend i { display:inline-block; width:12px; height:12px; margin-right:6px }
  code { background:#f5f5f5; padding:1px 4px; border-radius:4px }
</style>
</head>
<body>
<div id="map"></div>
<div class="hdr">HERE Traffic<br/><small>Son güncelleme: __LAST_UPDATED__</small></div>
<div id="pane"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const BBOX = __BBOX__; // [latMin, lonMin, latMax, lonMax]
  const bounds = L.latLngBounds([[BBOX[0], BBOX[1]],[BBOX[2], BBOX[3]]]);
  const map = L.map('map', { preferCanvas:true }).fitBounds(bounds);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution:'&copy; OpenStreetMap' }).addTo(map);

  const COLOR_MODE = "__COLOR_MODE__"; // jam | speed
  const data = {"type":"FeatureCollection","features":__FEATURES__};

  function jfToColor(jf){
    if (jf == null) return "#9e9e9e";
    if (jf >= 9.5) return "#000000";
    if (jf >= 7.5) return "#b71c1c";
    if (jf >= 5.0) return "#f57c00";
    if (jf >= 2.5) return "#fbc02d";
    return "#2e7d32";
  }
  function spToColor(sp, ff){
    if (sp == null || ff == null || ff <= 0) return "#9e9e9e";
    const r = sp / ff;
    if (r >= 0.85) return "#2e7d32";
    if (r >= 0.70) return "#fbc02d";
    if (r >= 0.50) return "#f57c00";
    return "#b71c1c";
  }
  const pickColor = p => (COLOR_MODE === "speed") ? spToColor(p.speed, p.freeFlow) : jfToColor(p.jamFactor);

  L.geoJSON(data, {
    style: f => ({ color: pickColor(f.properties), weight: 6, opacity: 0.9 }),
    onEachFeature: (f, layer) => {
      const p = f.properties || {};
      const sp = (p.speed!=null)? p.speed.toFixed(1): "—";
      const ff = (p.freeFlow!=null)? p.freeFlow.toFixed(1): "—";
      const jf = (p.jamFactor!=null)? p.jamFactor.toFixed(1): "—";
      const html = `<b>${p.desc || "—"}</b><br/>
        edge_id: <code>${p.edge_id || "—"}</code><br/>
        length: <code>${(p.length_m!=null)? p.length_m.toFixed(1): "—"} m</code><br/>
        speed: <code>${sp} km/h</code> · freeFlow: <code>${ff} km/h</code> · jamFactor: <code>${jf}</code><br/>
        confidence: <code>${p.confidence ?? "—"}</code> · traversability: <code>${p.traversability ?? "—"}</code>`;
      layer.bindPopup(html);
    }
  }).addTo(map);

  const pane = document.getElementById('pane');
  pane.innerHTML = `<b>Color mode:</b> <code>${COLOR_MODE.toUpperCase()}</code>
    <div class='legend box' style="margin-top:6px">
      <div><b>JamFactor</b></div>
      <div><i style="background:#2e7d32"></i> 0–2.5</div>
      <div><i style="background:#fbc02d"></i> 2.5–5</div>
      <div><i style="background:#f57c00"></i> 5–7.5</div>
      <div><i style="background:#b71c1c"></i> 7.5–9.5</div>
      <div><i style="background:#000"></i> 9.5–10</div>
    </div>`;
</script>
</body>
</html>
"""

def main():
    load_env()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / "data" / "raw" / os.environ.get("OUTPUT_JSON","here_flow_raw.json")
    dst = ROOT_DIR / (Path(sys.argv[2]) if len(sys.argv) > 2 else Path("src/visualization/map.html"))

    data = json.loads(src.read_text(encoding="utf-8"))
    results = data.get("results", []) if isinstance(data, dict) else []

    feats=[]
    for res in results:
        loc = res.get("location", {})
        cf  = res.get("currentFlow", {}) or {}

        # Hız seçimi (sağlam sıra) ve clamp
        sp  = first_not_none(cf.get("speedUncapped"), cf.get("speed"), cf.get("SP"))
        if sp is not None:
            sp = max(1.0, min(float(sp), 140.0))
        ffs = first_not_none(cf.get("freeFlow"), cf.get("FF"))
        jf  = first_not_none(cf.get("jamFactor"), cf.get("JF"))
        desc = (loc.get("description") or "").strip()

        line = extract_line_coords(res)
        if len(line) < 2:
            continue

        # Kenarın uzunluğu ve stabil kimlik
        length_m = polyline_length_m(line)
        edge_id = make_edge_id_from_coords(line)

        props_common = {
            "edge_id": edge_id,
            "length_m": length_m,
            "desc": desc,
            "speed": sp,
            "freeFlow": ffs,
            "jamFactor": jf,
            "confidence": cf.get("confidence"),
            "traversability": cf.get("traversability"),
        }

        # Alt-segmentler (varsa) – daha granüler renk
        sub = cf.get("subSegments")
        if isinstance(sub, list) and len(sub) >= 2:
            lens = [max(0.0, float(s.get("length", 0.0))) for s in sub]
            pieces = cut_polyline_by_lengths(line, lens)
            for piece, sseg in zip(pieces, sub):
                sub_props = props_common.copy()
                sub_props["speed"]     = first_not_none(sseg.get("speedUncapped"), sseg.get("speed"), sub_props.get("speed"))
                sub_props["freeFlow"]  = first_not_none(sseg.get("freeFlow"), sub_props.get("freeFlow"))
                sub_props["jamFactor"] = first_not_none(sseg.get("jamFactor"), sub_props.get("jamFactor"))
                # Alt parça için alt edge-id üret (ana + index)
                sub_props["edge_id"]   = f"{edge_id}:{len(feats)%1000:03d}"
                sub_props["length_m"]  = polyline_length_m(piece)
                feats.append(feature_line(piece, sub_props))
        else:
            feats.append(feature_line(line, props_common))

    # BBOX → Leaflet sırası
    bbox_env = os.environ.get("BBOX", "30.4000,39.7000,30.7500,39.8600")
    lon_min, lat_min, lon_max, lat_max = [float(x) for x in bbox_env.split(",")]
    bbox_js = [lat_min, lon_min, lat_max, lon_max]

    color_mode = os.environ.get("COLOR_MODE", "jam").lower()
    last_updated_local = now_local_str()

    # HTML yaz
    html = (TEMPLATE
        .replace("__FEATURES__", json.dumps(feats))
        .replace("__BBOX__", json.dumps(bbox_js))
        .replace("__COLOR_MODE__", color_mode)
        .replace("__LAST_UPDATED__", last_updated_local)
    )
    dst.write_text(html, encoding="utf-8")
    print(f"Wrote {dst.resolve()} with {len(feats)} feature(s). COLOR_MODE={color_mode}")

    # --- GeoJSON arşivi yaz (her feature'a t_utc ekleyerek) ---
    arch_dir = ROOT_DIR / os.environ.get("ARCHIVE_DIR","archive")
    arch_dir.mkdir(parents=True, exist_ok=True)
    ts_utc = now_utc_iso()
    fc = {"type":"FeatureCollection","features":[]}
    for f in feats:
        p = dict(f["properties"])
        p["t_utc"] = ts_utc
        fc["features"].append({"type":"Feature","geometry":f["geometry"],"properties":p})
    ts_name = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = arch_dir / f"flow_{ts_name}.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"Archived -> {out_path}")

    # Eski arşivleri buda
    try:
        keep = int(os.environ.get("MAX_ARCHIVES","500"))
        files = sorted(glob.glob(str(arch_dir / "flow_*.geojson")), key=os.path.getmtime, reverse=True)
        for old in files[keep:]:
            try: os.remove(old)
            except: pass
    except Exception as e:
        print("Prune warning:", e)

if __name__ == "__main__":
    main()
