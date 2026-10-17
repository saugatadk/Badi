import json
from pathlib import Path
from backend.graph import get_falkor_client
from backend.config import FALKORDB_GRAPH
from backend.missions import recompute_missions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "trishuli_flood_dataset.json"
if not DATA_FILE.exists():
    DATA_FILE = DATA_DIR / "trishuli_flood_network.json"

def seed_falkordb():
    print(f"Connecting to FalkorDB and selecting graph '{FALKORDB_GRAPH}'...")
    client = get_falkor_client()
    graph = client.select_graph(FALKORDB_GRAPH)

    print("Clearing existing graph data...")
    graph.query("MATCH (n) DETACH DELETE n")

    print(f"Loading seed data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Create Sources
    print(f"Creating {len(data['sources'])} Source nodes...")
    for s in data["sources"]:
        graph.query(
            "CREATE (s:Source {id: $id, name: $name, url: $url, license: $license})",
            s
        )

    # 2. Create Locations
    print(f"Creating {len(data['locations'])} Location nodes...")
    for loc in data["locations"]:
        graph.query(
            "CREATE (l:Location { "
            "  id: $id, "
            "  name: $name, "
            "  type: $type, "
            "  district: $district, "
            "  lat: $lat, "
            "  lon: $lon, "
            "  households: $households "
            "})",
            loc
        )

    # 3. Create Facilities and link to Locations
    print(f"Creating {len(data['facilities'])} Facility nodes...")
    for fac in data["facilities"]:
        graph.query(
            "MATCH (l:Location {id: $location_id}) "
            "CREATE (f:Facility { "
            "  id: $id, "
            "  name: $name, "
            "  location_id: $location_id, "
            "  type: $type, "
            "  importance_weight: $importance_weight "
            "}) "
            "CREATE (f)-[:AT_LOCATION]->(l)",
            fac
        )

    # 4. Create RoadSegments, bidirectional ROAD_TO edges, CONNECTS edges, and initial Observations
    print(f"Creating {len(data['road_segments'])} RoadSegment nodes and topology edges...")
    for road in data["road_segments"]:
        r_params = {
            "id": road["id"],
            "name": road["name"],
            "u": road["u"],
            "v": road["v"],
            "length_km": road["length_km"],
            "is_bridge": road["is_bridge"],
            "status": road["status"]
        }
        # Create RoadSegment node & connects relationships
        r_params["coordinates_json"] = json.dumps(road.get("coordinates", []))
        graph.query(
            "MATCH (u:Location {id: $u}) "
            "MATCH (v:Location {id: $v}) "
            "CREATE (r:RoadSegment { "
            "  id: $id, "
            "  name: $name, "
            "  u: $u, "
            "  v: $v, "
            "  length_km: $length_km, "
            "  is_bridge: $is_bridge, "
            "  status: $status, "
            "  coordinates_json: $coordinates_json "
            "}) "
            "CREATE (r)-[:CONNECTS]->(u) "
            "CREATE (r)-[:CONNECTS]->(v) "
            "CREATE (u)-[:ROAD_TO { "
            "  road_id: $id, "
            "  road_name: $name, "
            "  status: $status, "
            "  length_km: $length_km, "
            "  is_bridge: $is_bridge "
            "}]->(v) "
            "CREATE (v)-[:ROAD_TO { "
            "  road_id: $id, "
            "  road_name: $name, "
            "  status: $status, "
            "  length_km: $length_km, "
            "  is_bridge: $is_bridge "
            "}]->(u)",
            r_params
        )

        # Create Bridge nodes and link to RoadSegment
        for bridge in road.get("bridges", []):
            b_id = f"bridge_{bridge['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MATCH (r:RoadSegment {id: $road_id}) "
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon "
                "MERGE (r)-[:INCLUDES_BRIDGE]->(b)",
                {
                    "road_id": road["id"],
                    "id": b_id,
                    "name": bridge["name"],
                    "status": bridge["status"],
                    "location": bridge.get("location", ""),
                    "district": bridge.get("district", ""),
                    "lat": bridge["lat"],
                    "lon": bridge["lon"]
                }
            )

        # Create initial observation if present
        if "initial_observation" in road and road["initial_observation"]:
            obs = road["initial_observation"]
            obs_id = f"obs_init_{road['id']}"
            graph.query(
                "MATCH (r:RoadSegment {id: $road_id}) "
                "MATCH (s:Source {id: $source_id}) "
                "CREATE (o:Observation { "
                "  id: $obs_id, "
                "  road_id: $road_id, "
                "  status: $status, "
                "  note: $note, "
                "  timestamp: $timestamp, "
                "  is_simulated: false "
                "}) "
                "CREATE (r)-[:HAS_OBSERVATION]->(o) "
                "CREATE (o)-[:FROM_SOURCE]->(s)",
                {
                    "road_id": road["id"],
                    "source_id": obs["source_id"],
                    "obs_id": obs_id,
                    "status": road["status"],
                    "note": obs["note"],
                    "timestamp": obs["timestamp"]
                }
            )

    # 4b. Seed all surveyed bridges from GeoJSON
    bridge_geojson = DATA_FILE.parent / "hot_flood_npl_bridge_damage.geojson"
    if bridge_geojson.exists():
        with open(bridge_geojson, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        print(f"Indexing all {len(b_data['features'])} bridge survey points in FalkorDB...")
        for feat in b_data["features"]:
            bp = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            b_id = f"bridge_{bp['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon",
                {
                    "id": b_id,
                    "name": bp["name"],
                    "status": bp.get("status", "Unknown"),
                    "location": bp.get("location", ""),
                    "district": bp.get("adm2_name", ""),
                    "lat": coords[1],
                    "lon": coords[0]
                }
            )

    # 5. Create Seed Episodic Memory Event
    print("Recording initial episodic memory in FalkorDB (Track 02)...")
    graph.query(
        "CREATE (e:EpisodicMemory { "
        "  id: 'mem_seed_init', "
        "  session_id: 'init_session', "
        "  agent_role: 'SystemSeeder', "
        "  event_type: 'network_initialized', "
        "  summary: 'Seeded Bhote Koshi–Trishuli flood road network (12 locations, 3 facilities, 15 road segments).', "
        "  road_id: '', "
        "  timestamp: '2026-10-16T07:30:00Z' "
        "})"
    )

    # 6. Compute initial prioritized missions
    print("Computing initial prioritized recon missions...")
    missions = recompute_missions()
    print(f"Computed {len(missions)} initial reconnaissance missions.")

    print("FalkorDB graph seeding completed successfully.")

if __name__ == "__main__":
    seed_falkordb()
