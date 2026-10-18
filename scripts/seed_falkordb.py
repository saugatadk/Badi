import json
from pathlib import Path

from backend.config import FALKORDB_GRAPH
from backend.graph import get_falkor_client
from backend.missions import recompute_missions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "trishuli_flood_dataset.json"
if not DATA_FILE.exists():
    DATA_FILE = DATA_DIR / "trishuli_flood_network.json"


def seed_falkordb():
    """Populates FalkorDB with baseline network topology and survey bridges."""
    print(f"Connecting to FalkorDB and selecting graph '{FALKORDB_GRAPH}'...")
    client = get_falkor_client()
    graph = client.select_graph(FALKORDB_GRAPH)

    print("Clearing existing graph data...")
    graph.query("MATCH (n) DETACH DELETE n")

    print(f"Loading seed data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    # Ingest observational sources
    sources = data.get("sources", [])
    print(f"Creating {len(sources)} Source nodes...")
    for source in sources:
        graph.query(
            "CREATE (s:Source {id: $id, name: $name, url: $url, license: $license})",
            source,
        )

    # Ingest locations
    locations = data.get("locations", [])
    print(f"Creating {len(locations)} Location nodes...")
    for location in locations:
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
            location,
        )

    # Ingest healthcare facilities and attach to locations
    facilities = data.get("facilities", [])
    print(f"Creating {len(facilities)} Facility nodes...")
    for facility in facilities:
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
            facility,
        )

    # Ingest road segments, bidirectional topological edges, and initial observations
    road_segments = data.get("road_segments", [])
    print(f"Creating {len(road_segments)} RoadSegment nodes and topology edges...")
    for road in road_segments:
        road_params = {
            "id": road["id"],
            "name": road["name"],
            "u": road["u"],
            "v": road["v"],
            "length_km": road["length_km"],
            "is_bridge": road["is_bridge"],
            "status": road["status"],
            "coordinates_json": json.dumps(road.get("coordinates", [])),
        }
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
            road_params,
        )

        for bridge in road.get("bridges", []):
            bridge_id = f"bridge_{bridge['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MATCH (r:RoadSegment {id: $road_id}) "
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon "
                "MERGE (r)-[:INCLUDES_BRIDGE]->(b)",
                {
                    "road_id": road["id"],
                    "id": bridge_id,
                    "name": bridge["name"],
                    "status": bridge["status"],
                    "location": bridge.get("location", ""),
                    "district": bridge.get("district", ""),
                    "lat": bridge["lat"],
                    "lon": bridge["lon"],
                },
            )

        if "initial_observation" in road and road["initial_observation"]:
            observation = road["initial_observation"]
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
                    "source_id": observation["source_id"],
                    "obs_id": obs_id,
                    "status": road["status"],
                    "note": observation["note"],
                    "timestamp": observation["timestamp"],
                },
            )

    # Ingest surveyed bridge points from GeoJSON
    bridge_geojson = DATA_FILE.parent / "hot_flood_npl_bridge_damage.geojson"
    if bridge_geojson.exists():
        with open(bridge_geojson, "r", encoding="utf-8") as file_handle:
            bridge_data = json.load(file_handle)
        print(f"Indexing all {len(bridge_data['features'])} bridge survey points in FalkorDB...")
        for feature in bridge_data["features"]:
            bridge_props = feature["properties"]
            coordinates = feature["geometry"]["coordinates"]
            bridge_id = f"bridge_{bridge_props['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon",
                {
                    "id": bridge_id,
                    "name": bridge_props["name"],
                    "status": bridge_props.get("status", "Unknown"),
                    "location": bridge_props.get("location", ""),
                    "district": bridge_props.get("adm2_name", ""),
                    "lat": coordinates[1],
                    "lon": coordinates[0],
                },
            )

    # Record baseline episodic memory event
    print("Recording initial episodic memory in FalkorDB...")
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

    print("Computing initial prioritized recon missions...")
    missions = recompute_missions()
    print(f"Computed {len(missions)} initial reconnaissance missions.")
    print("FalkorDB graph seeding completed successfully.")


if __name__ == "__main__":
    seed_falkordb()
