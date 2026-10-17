import json
import urllib.request
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NETWORK_FILE = BASE_DIR / "data" / "trishuli_flood_network.json"
BRIDGE_FILE = BASE_DIR / "data" / "hot_flood_npl_bridge_damage.geojson"

def interpolate_waypoints(pts, steps_between=12):
    """Linearly interpolates points between waypoints to create smooth paths."""
    dense = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        for s in range(steps_between):
            t = s / steps_between
            dense.append([round(x1 + t * (x2 - x1), 6), round(y1 + t * (y2 - y1), 6)])
    dense.append([round(pts[-1][0], 6), round(pts[-1][1], 6)])
    return dense

def fetch_osrm_route(x1, y1, x2, y2):
    url = f"https://router.project-osrm.org/route/v1/driving/{x1},{y1};{x2},{y2}?overview=full&geometries=geojson"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BadiNepalDisasterApp/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read())
            if d.get("routes"):
                return d["routes"][0]["geometry"]["coordinates"]
    except Exception as e:
        print(f"OSRM fallback for ({x1},{y1})->({x2},{y2}): {e}")
    return None

def main():
    print(f"Loading network from {NETWORK_FILE}...")
    with open(NETWORK_FILE, "r", encoding="utf-8") as f:
        network = json.load(f)

    print(f"Loading bridges from {BRIDGE_FILE}...")
    with open(BRIDGE_FILE, "r", encoding="utf-8") as f:
        bridges_fc = json.load(f)

    bridges_by_name = {}
    for feat in bridges_fc["features"]:
        p = feat["properties"]
        geom = feat["geometry"]["coordinates"]
        bridges_by_name[p["name"]] = {
            "name": p["name"],
            "status": p.get("status", "Unknown"),
            "location": p.get("location", ""),
            "district": p.get("adm2_name", ""),
            "lon": geom[0],
            "lat": geom[1]
        }

    locs = {l["id"]: l for l in network["locations"]}

    # Define bridge lists for corridor roads based on real ground survey
    road_bridge_mapping = {
        "road_timure_syabru": [
            "Timure Bridge 1",
            "Timure Bridge 2",
            "Ghatte Khola",
            "Bhotekoshi Bridge at Chilime",
            "Chilime Bridge, Chilime Khola",
            "Hydropower Bailey Bridge at Bhotekoshi, Syaphrubeshi",
            "Syafrubesi Bridge"
        ],
        "road_timure_briddhim": [
            "Ghatte Khola"
        ],
        "road_briddhim_syabru": [
            "Hydropower Bailey Bridge at Bhotekoshi, Syaphrubeshi"
        ],
        "road_syabru_dhunche": [
            "Sano bharkhu",
            "Trisuli Khola Bridge"
        ],
        "road_syabru_mailung": [
            "Trisuli 216 Audit 1 Bailey Bridge",
            "Haku1 Bailey Bridge at Hydropower, Trishuli",
            "Haku khola",
            "Bimali Khola",
            "Haku2 Bailey Bridge at Hydropower, Trishuli",
            "Thade Khola",
            "Thado Khola (Crossing 0+750",
            "Crossing 0+700",
            "Upper Trishuli Hydrower Bailey Bridge",
            "Mailung Bailey Bridge at Mailung Bazaar"
        ],
        "road_syabru_goljung": [
            "Chilime Bridge, Chilime Khola"
        ],
        "road_goljung_gatlang": [],
        "road_gatlang_dhunche": [
            "Trisuli 216 Audit 1 Bailey Bridge"
        ],
        "road_dhunche_kalika": [],
        "road_kalika_betrawati": [
            "Sukaure Khola"
        ],
        "road_mailung_betrawati": [
            "Trishuli Bridge Mailung Chipleti",
            "Trishuli 3A Dam-3",
            "Trishuli 3A Dam-2",
            "Bahubali Jharana Pul",
            "Sukaure Khola",
            "Naya Pul Trishuli, Khalte CH 37+810",
            "Betrawati Old Bridge",
            "Betrawati New Bridge"
        ],
        "road_gatlang_kispang": [],
        "road_kispang_betrawati": [
            "Trisuli River Bridge Tupche"
        ],
        "road_betrawati_bidur": [
            "Trisuli River Bridge Tupche",
            "Dhobi Khola",
            "Gundre Khola",
            "Bire Khola",
            "Gerkhu khola",
            "Trisuli River Bridge ,Trisuli Bazzar(Old)",
            "Trisuli River Bridge ,Trisuli Bazzar(New)",
            "Trisuli River Bridge ,Simchaur"
        ],
        "road_devighat_bidur": [
            "Suspension Bridge at Devighat near Jalshrot tatha Sinchai Division",
            "Trisuli River Bridge, Near Devighat, Midhill Highway",
            "Tandi Khola Bridge, Devighat (Old)",
            "Tandi Khola Bridge, Devighat (New)",
            "Trisuli River Bridge,Khadgabhanjyang Srikhali Sadak"
        ]
    }

    # High precision road waypoints for winding mountainous corridors
    corridor_waypoints = {
        "road_timure_syabru": [
            [85.3667, 28.2528],
            [85.358278, 28.239978], # Timure Bridge 2
            [85.361803, 28.226711], # Ghatte Khola
            [85.3550, 28.2050],
            [85.343281, 28.182803], # Bhotekoshi Bridge at Chilime
            [85.342219, 28.181653], # Chilime Bridge
            [85.339928, 28.164386], # Hydropower Bailey Bridge
            [85.332492, 28.157478], # Syafrubesi Bridge
            [85.3427, 28.1646]       # Syabrubesi Bazaar
        ],
        "road_timure_briddhim": [
            [85.3667, 28.2528],
            [85.361803, 28.226711],
            [85.3610, 28.2140],
            [85.3577, 28.2007]
        ],
        "road_briddhim_syabru": [
            [85.3577, 28.2007],
            [85.3520, 28.1880],
            [85.3460, 28.1750],
            [85.3427, 28.1646]
        ],
        "road_syabru_mailung": [
            [85.3427, 28.1646],
            [85.332492, 28.157478],
            [85.318714, 28.145564], # Sano Bharkhu
            [85.299472, 28.127264], # Trisuli Khola Bridge
            [85.288422, 28.120283], # Trisuli 216 Audit 1
            [85.281031, 28.112953], # Haku1 Bailey Bridge
            [85.278964, 28.110156], # Haku Khola
            [85.277147, 28.108542], # Bimali Khola
            [85.275139, 28.108367], # Haku2 Bailey Bridge
            [85.251936, 28.091556], # Thade Khola
            [85.225556, 28.082778], # Thado Khola
            [85.222594, 28.081897], # Crossing 0+700
            [85.212844, 28.075978], # Upper Trishuli Bailey
            [85.207533, 28.071894], # Mailung Bailey Bridge
            [85.2070, 28.0718]       # Mailung Confluence
        ],
        "road_syabru_goljung": [
            [85.3427, 28.1646],
            [85.3350, 28.1680],
            [85.3250, 28.1720],
            [85.3159, 28.1700]
        ],
        "road_mailung_betrawati": [
            [85.2070, 28.0718],
            [85.205833, 28.065278], # Mailung Chipleti
            [85.206325, 28.062578], # Trishuli 3A Dam-3
            [85.206686, 28.062053], # Trishuli 3A Dam-2
            [85.192778, 28.038056], # Bahubali Jharana Pul
            [85.179778, 28.014339], # Sukaure Khola
            [85.184889, 27.995342], # Naya Pul Trishuli, Khalte
            [85.185811, 27.974339], # Betrawati Bridge
            [85.1860, 27.9731]       # Betrawati Hub
        ],
        "road_gatlang_kispang": [
            [85.2684, 28.1628],
            [85.2400, 28.1300],
            [85.2000, 28.1000],
            [85.1500, 28.0650],
            [85.1034, 28.0354]
        ],
        "road_kispang_betrawati": [
            [85.1034, 28.0354],
            [85.1300, 28.0100],
            [85.1550, 27.9900],
            [85.1728, 27.9638], # Tupche Bridge
            [85.1860, 27.9731]
        ]
    }

    # Highways with confirmed OSRM routes
    osrm_roads = [
        "road_syabru_dhunche",
        "road_dhunche_kalika",
        "road_kalika_betrawati",
        "road_betrawati_bidur",
        "road_devighat_bidur",
        "road_goljung_gatlang",
        "road_gatlang_dhunche"
    ]

    for road in network["road_segments"]:
        rid = road["id"]
        u = locs[road["u"]]
        v = locs[road["v"]]

        # 1. Attach real surveyed bridges
        bridge_names = road_bridge_mapping.get(rid, [])
        road_bridges = []
        for bname in bridge_names:
            if bname in bridges_by_name:
                road_bridges.append(bridges_by_name[bname])
            else:
                print(f"Warning: bridge '{bname}' not found in GeoJSON")
        road["bridges"] = road_bridges
        road["bridge_count"] = len(road_bridges)

        # 2. Assign high-precision curved coordinates
        coords = None
        if rid in osrm_roads:
            print(f"Fetching/validating OSRM route for {rid}...")
            coords = fetch_osrm_route(u["lon"], u["lat"], v["lon"], v["lat"])
            time.sleep(0.3)

        if not coords or len(coords) < 5:
            if rid in corridor_waypoints:
                print(f"Using surveyed river/corridor waypoints for {rid}...")
                coords = interpolate_waypoints(corridor_waypoints[rid], steps_between=12)
            else:
                print(f"Interpolating endpoints for {rid}...")
                coords = interpolate_waypoints([[u["lon"], u["lat"]], [v["lon"], v["lat"]]], steps_between=15)

        # Ensure starting and ending points strictly anchor to settlement coords
        coords[0] = [u["lon"], u["lat"]]
        coords[-1] = [v["lon"], v["lat"]]
        road["coordinates"] = coords
        print(f"  -> {rid}: {len(road_bridges)} bridges, {len(coords)} coordinate points.")

    # Save enriched network
    print(f"Writing enriched data back to {NETWORK_FILE}...")
    with open(NETWORK_FILE, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2)

    # Also copy bridge GeoJSON to frontend for fast client-side loading
    frontend_bridge_file = BASE_DIR / "frontend" / "hot_flood_npl_bridge_damage.geojson"
    print(f"Copying bridge damage dataset to {frontend_bridge_file}...")
    with open(frontend_bridge_file, "w", encoding="utf-8") as f:
        json.dump(bridges_fc, f, indent=2)

    print("Data integration complete!")

if __name__ == "__main__":
    main()
