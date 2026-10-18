import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NETWORK_FILE = BASE_DIR / "data" / "trishuli_flood_network.json"
BRIDGE_FILE = BASE_DIR / "data" / "hot_flood_npl_bridge_damage.geojson"


def interpolate_waypoints(waypoints: list[list[float]], steps_between: int = 12) -> list[list[float]]:
    """Linearly interpolates points between waypoints to create smooth paths."""
    dense_path: list[list[float]] = []
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        for step in range(steps_between):
            ratio = step / steps_between
            dense_path.append([round(x1 + ratio * (x2 - x1), 6), round(y1 + ratio * (y2 - y1), 6)])
    dense_path.append([round(waypoints[-1][0], 6), round(waypoints[-1][1], 6)])
    return dense_path


def fetch_osrm_route(x1: float, y1: float, x2: float, y2: float) -> list[list[float]] | None:
    """Queries public OSRM service for road driving coordinates with graceful fallback."""
    url = f"https://router.project-osrm.org/route/v1/driving/{x1},{y1};{x2},{y2}?overview=full&geometries=geojson"
    try:
        request_obj = urllib.request.Request(url, headers={"User-Agent": "BadiNepalDisasterApp/1.0"})
        with urllib.request.urlopen(request_obj, timeout=5) as http_response:
            payload = json.loads(http_response.read().decode("utf-8"))
            if payload.get("routes"):
                return payload["routes"][0]["geometry"]["coordinates"]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as err:
        print(f"OSRM fallback for ({x1},{y1})->({x2},{y2}): {err}")
    return None


def main():
    print(f"Loading network from {NETWORK_FILE}...")
    with open(NETWORK_FILE, "r", encoding="utf-8") as file_handle:
        network = json.load(file_handle)

    print(f"Loading bridges from {BRIDGE_FILE}...")
    with open(BRIDGE_FILE, "r", encoding="utf-8") as file_handle:
        bridges_fc = json.load(file_handle)

    bridges_by_name = {}
    for feature in bridges_fc["features"]:
        props = feature["properties"]
        geom_coords = feature["geometry"]["coordinates"]
        bridges_by_name[props["name"]] = {
            "name": props["name"],
            "status": props.get("status", "Unknown"),
            "location": props.get("location", ""),
            "district": props.get("adm2_name", ""),
            "lon": geom_coords[0],
            "lat": geom_coords[1],
        }

    locations_by_id = {loc["id"]: loc for loc in network["locations"]}

    road_bridge_mapping = {
        "road_timure_syabru": [
            "Timure Bridge 1",
            "Timure Bridge 2",
            "Ghatte Khola",
            "Bhotekoshi Bridge at Chilime",
            "Chilime Bridge, Chilime Khola",
            "Hydropower Bailey Bridge at Bhotekoshi, Syaphrubeshi",
            "Syafrubesi Bridge",
        ],
        "road_timure_briddhim": [
            "Ghatte Khola",
        ],
        "road_briddhim_syabru": [
            "Hydropower Bailey Bridge at Bhotekoshi, Syaphrubeshi",
        ],
        "road_syabru_dhunche": [
            "Sano bharkhu",
            "Trisuli Khola Bridge",
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
            "Mailung Bailey Bridge at Mailung Bazaar",
        ],
        "road_syabru_goljung": [
            "Chilime Bridge, Chilime Khola",
        ],
        "road_goljung_gatlang": [],
        "road_gatlang_dhunche": [
            "Trisuli 216 Audit 1 Bailey Bridge",
        ],
        "road_dhunche_kalika": [],
        "road_kalika_betrawati": [
            "Sukaure Khola",
        ],
        "road_mailung_betrawati": [
            "Trishuli Bridge Mailung Chipleti",
            "Trishuli 3A Dam-3",
            "Trishuli 3A Dam-2",
            "Bahubali Jharana Pul",
            "Sukaure Khola",
            "Naya Pul Trishuli, Khalte CH 37+810",
            "Betrawati Old Bridge",
            "Betrawati New Bridge",
        ],
        "road_gatlang_kispang": [],
        "road_kispang_betrawati": [
            "Trisuli River Bridge Tupche",
        ],
        "road_betrawati_bidur": [
            "Trisuli River Bridge Tupche",
            "Dhobi Khola",
            "Gundre Khola",
            "Bire Khola",
            "Gerkhu khola",
            "Trisuli River Bridge ,Trisuli Bazzar(Old)",
            "Trisuli River Bridge ,Trisuli Bazzar(New)",
            "Trisuli River Bridge ,Simchaur",
        ],
        "road_devighat_bidur": [
            "Suspension Bridge at Devighat near Jalshrot tatha Sinchai Division",
            "Trisuli River Bridge, Near Devighat, Midhill Highway",
            "Tandi Khola Bridge, Devighat (Old)",
            "Tandi Khola Bridge, Devighat (New)",
            "Trisuli River Bridge,Khadgabhanjyang Srikhali Sadak",
        ],
    }

    # Surveyed waypoints for mountainous corridors
    corridor_waypoints = {
        "road_timure_syabru": [
            [85.3667, 28.2528],
            [85.358278, 28.239978],
            [85.361803, 28.226711],
            [85.3550, 28.2050],
            [85.343281, 28.182803],
            [85.342219, 28.181653],
            [85.339928, 28.164386],
            [85.332492, 28.157478],
            [85.3427, 28.1646],
        ],
        "road_timure_briddhim": [
            [85.3667, 28.2528],
            [85.361803, 28.226711],
            [85.3610, 28.2140],
            [85.3577, 28.2007],
        ],
        "road_briddhim_syabru": [
            [85.3577, 28.2007],
            [85.3520, 28.1880],
            [85.3460, 28.1750],
            [85.3427, 28.1646],
        ],
        "road_syabru_mailung": [
            [85.3427, 28.1646],
            [85.332492, 28.157478],
            [85.318714, 28.145564],
            [85.299472, 28.127264],
            [85.288422, 28.120283],
            [85.281031, 28.112953],
            [85.278964, 28.110156],
            [85.277147, 28.108542],
            [85.275139, 28.108367],
            [85.251936, 28.091556],
            [85.225556, 28.082778],
            [85.222594, 28.081897],
            [85.212844, 28.075978],
            [85.207533, 28.071894],
            [85.2070, 28.0718],
        ],
        "road_syabru_goljung": [
            [85.3427, 28.1646],
            [85.3350, 28.1680],
            [85.3250, 28.1720],
            [85.3159, 28.1700],
        ],
        "road_mailung_betrawati": [
            [85.2070, 28.0718],
            [85.205833, 28.065278],
            [85.206325, 28.062578],
            [85.206686, 28.062053],
            [85.192778, 28.038056],
            [85.179778, 28.014339],
            [85.184889, 27.995342],
            [85.185811, 27.974339],
            [85.1860, 27.9731],
        ],
        "road_gatlang_kispang": [
            [85.2684, 28.1628],
            [85.2400, 28.1300],
            [85.2000, 28.1000],
            [85.1500, 28.0650],
            [85.1034, 28.0354],
        ],
        "road_kispang_betrawati": [
            [85.1034, 28.0354],
            [85.1300, 28.0100],
            [85.1550, 27.9900],
            [85.1728, 27.9638],
            [85.1860, 27.9731],
        ],
    }

    osrm_roads = [
        "road_syabru_dhunche",
        "road_dhunche_kalika",
        "road_kalika_betrawati",
        "road_betrawati_bidur",
        "road_devighat_bidur",
        "road_goljung_gatlang",
        "road_gatlang_dhunche",
    ]

    for road in network["road_segments"]:
        road_id = road["id"]
        origin_loc = locations_by_id[road["u"]]
        dest_loc = locations_by_id[road["v"]]

        # Attach surveyed bridges
        bridge_names = road_bridge_mapping.get(road_id, [])
        road_bridges = []
        for bridge_name in bridge_names:
            if bridge_name in bridges_by_name:
                road_bridges.append(bridges_by_name[bridge_name])
            else:
                print(f"Warning: bridge '{bridge_name}' not found in GeoJSON")
        road["bridges"] = road_bridges
        road["bridge_count"] = len(road_bridges)

        # Assign curved geometry coordinates
        coordinates = None
        if road_id in osrm_roads:
            print(f"Fetching/validating OSRM route for {road_id}...")
            coordinates = fetch_osrm_route(origin_loc["lon"], origin_loc["lat"], dest_loc["lon"], dest_loc["lat"])
            time.sleep(0.3)

        if not coordinates or len(coordinates) < 5:
            if road_id in corridor_waypoints:
                print(f"Using surveyed river/corridor waypoints for {road_id}...")
                coordinates = interpolate_waypoints(corridor_waypoints[road_id], steps_between=12)
            else:
                print(f"Interpolating endpoints for {road_id}...")
                coordinates = interpolate_waypoints(
                    [[origin_loc["lon"], origin_loc["lat"]], [dest_loc["lon"], dest_loc["lat"]]],
                    steps_between=15,
                )

        # Anchor endpoints to settlement coordinates
        coordinates[0] = [origin_loc["lon"], origin_loc["lat"]]
        coordinates[-1] = [dest_loc["lon"], dest_loc["lat"]]
        road["coordinates"] = coordinates
        print(f"  -> {road_id}: {len(road_bridges)} bridges, {len(coordinates)} coordinate points.")

    print(f"Writing enriched data back to {NETWORK_FILE}...")
    with open(NETWORK_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(network, file_handle, indent=2)

    frontend_bridge_file = BASE_DIR / "frontend" / "hot_flood_npl_bridge_damage.geojson"
    print(f"Copying bridge damage dataset to {frontend_bridge_file}...")
    with open(frontend_bridge_file, "w", encoding="utf-8") as file_handle:
        json.dump(bridges_fc, file_handle, indent=2)

    print("Data integration complete.")


if __name__ == "__main__":
    main()
