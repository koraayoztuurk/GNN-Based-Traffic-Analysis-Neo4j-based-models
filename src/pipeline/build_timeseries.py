import os, json, glob, math
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.parent

OUT_DIR = ROOT_DIR / "data"
ARCHIVE_DIR = ROOT_DIR / os.environ.get("ARCHIVE_DIR","archive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_segment_id(seg_id):
    """
    Normalize segment ID - sub-segment'leri ana yol ID'sine dönüştür
    edge:xxx:001 -> edge:xxx
    edge:xxx:002 -> edge:xxx
    edge:xxx -> edge:xxx (değişmez)
    """
    if not seg_id:
        return seg_id
    # Sub-segment varsa (ikiden fazla : varsa), son kısmı kaldır
    parts = seg_id.split(':')
    if len(parts) > 2:
        # edge:xxx:001 -> edge:xxx
        return ':'.join(parts[:2])
    return seg_id

def load_geojson(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print("read error:", path, e)
        return None

def main():
    files = sorted(glob.glob(str(ARCHIVE_DIR/"flow_*.geojson")))
    if not files:
        print("No archive files found in", ARCHIVE_DIR)
        return

    rows = []
    geom_by_edge = {}  # edge_id -> geometry (GeoJSON LineString)
    props_by_edge = {}  # edge_id -> en son properties (desc, length_m, etc.)

    for fp in files:
        gj = load_geojson(fp)
        if not gj or gj.get("type") != "FeatureCollection":
            continue
        for f in gj.get("features", []):
            if f.get("type") != "Feature": continue
            geom = f.get("geometry") or {}
            props = f.get("properties") or {}
            if geom.get("type") != "LineString": continue

            raw_edge_id = props.get("edge_id")
            # NORMALIZE: edge:xxx:001 -> edge:xxx (sub-segment'leri birleştir)
            edge_id = normalize_segment_id(raw_edge_id)
            t_utc   = props.get("t_utc")
            speed   = props.get("speed")
            ff      = props.get("freeFlow")
            jf      = props.get("jamFactor")
            conf    = props.get("confidence")
            trav    = props.get("traversability")
            length  = props.get("length_m")
            name    = props.get("desc")

            if not edge_id or not t_utc:
                continue

            rows.append({
                "edge_id": edge_id,
                "t_utc": t_utc,
                "speed_kmh": speed,
                "freeflow_kmh": ff,
                "jam_factor": jf,
                "confidence": conf,
                "traversability": trav,
                "length_m": length,
                "name": name
            })

            # edge geometri sözlüğü (ilk gördüğümüzü al)
            if edge_id not in geom_by_edge:
                geom_by_edge[edge_id] = geom
            
            # edge properties'i sakla (en son gördüğümüzü tut)
            props_by_edge[edge_id] = {
                "desc": name,
                "length_m": length,
                "frc": props.get("frc"),
                "confidence": conf,
                "traversability": trav
            }

    # --- Çıktılar ---
    # 1) timeseries CSV/Parquet
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        # t_utc zaman sütununu datetime olarak tut
        with pd.option_context('mode.chained_assignment', None):
            df["t_utc"] = pd.to_datetime(df["t_utc"], utc=True, errors="coerce")
        # Parquet tercih
        pq_path = OUT_DIR / "timeseries.parquet"
        try:
            import pyarrow  # noqa
            df.to_parquet(pq_path, index=False)
            print("Wrote", pq_path)
        except Exception:
            csv_path = OUT_DIR / "timeseries.csv"
            df.to_csv(csv_path, index=False)
            print("PyArrow yok; CSV yazıldı ->", csv_path)
    except Exception as e:
        # pandas yoksa düz JSON satırları yaz
        js_path = OUT_DIR / "timeseries.jsonl"
        with js_path.open("w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("pandas bulunamadı; JSONL yazıldı ->", js_path)

    # 2) tekil geometri (edges_static.geojson) - tüm properties ile
    static = {"type":"FeatureCollection","features":[]}
    for edge_id, geom in geom_by_edge.items():
        # Properties'i birleştir
        full_props = {"edge_id": edge_id}
        if edge_id in props_by_edge:
            full_props.update(props_by_edge[edge_id])
        
        static["features"].append({
            "type":"Feature",
            "geometry": geom,
            "properties": full_props
        })
    out_geo = OUT_DIR / "processed" / "edges_static.geojson"
    out_geo.write_text(json.dumps(static, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out_geo, f"({len(static['features'])} edges)")

if __name__ == "__main__":
    main()
