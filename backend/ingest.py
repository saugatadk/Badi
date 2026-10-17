"""
Ingestion module for ReliefWeave / Badi disaster response system.
Loads the complete Trishuli flood corridor dataset into FalkorDB,
establishing topological relationships, facility links, bridge associations,
observational sources, and computing initial recon missions.
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union
from backend.graph import get_falkor_client
from backend.config import FALKORDB_GRAPH
from backend.missions import recompute_missions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_FILE = DATA_DIR / "trishuli_flood_dataset.json"

def resolve_default_file() -> Path:
    for candidate in [
        DATA_DIR / "trishuli_flood_dataset.json",
        DATA_DIR / "badi_trishuli_complete_dataset.json",
        DATA_DIR / "trishuli_flood_network.json"
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No baseline flood dataset found in {DATA_DIR}")

def normalize_topology_dataset(raw: Any) -> Dict[str, Any]:
    """
    Universally normalizes arbitrary disaster JSON files into the standard Badi topology format.
    Handles:
    - Badi schema: locations, facilities, road_segments / roads
    - Graph Node-Link schemas: nodes / vertices and links / edges / relations
    - GeoJSON: FeatureCollection with Point (locations) and LineString (roads)
    - Loose property keys: source/target -> u/v, latitude/longitude -> lat/lon, population -> households
    - Auto-generates missing endpoints, facilities, and sources.
    """
    if isinstance(raw, list):
        raw_dict = {"nodes": raw}
    elif isinstance(raw, dict):
        raw_dict = raw
    else:
        return {"sources": [], "locations": [], "facilities": [], "road_segments": [], "bridges": []}

    locations_dict: Dict[str, Dict[str, Any]] = {}
    roads_list: list[Dict[str, Any]] = []
    facilities_list: list[Dict[str, Any]] = []
    bridges_list: list[Dict[str, Any]] = []
    sources_list: list[Dict[str, Any]] = []

    # 1. Parse Sources if present
    for s in raw_dict.get("sources", []):
        if isinstance(s, dict):
            s_id = str(s.get("id") or s.get("source_id") or "src_custom")
            sources_list.append({
                "id": s_id,
                "name": str(s.get("name") or s.get("label") or s_id),
                "url": str(s.get("url") or "local://import"),
                "license": str(s.get("license") or "Open Access")
            })

    # Ensure default sources exist
    known_src_ids = {s["id"] for s in sources_list}
    for def_s in [
        {"id": "src_ndrrma", "name": "NDRRMA Nepal", "url": "https://ndrrma.gov.np", "license": "Open Data Nepal"},
        {"id": "src_scout", "name": "Field Recon Scout", "url": "local://scout", "license": "Operational"},
        {"id": "src_user", "name": "Field Community Report", "url": "local://user", "license": "Community"},
        {"id": "src_operator", "name": "Emergency Operations Center", "url": "local://eoc", "license": "Operational"}
    ]:
        if def_s["id"] not in known_src_ids:
            sources_list.append(def_s)
            known_src_ids.add(def_s["id"])

    # 2. Extract raw node/location candidates
    raw_nodes = (
        raw_dict.get("locations") or
        raw_dict.get("nodes") or
        raw_dict.get("vertices") or
        raw_dict.get("settlements") or
        raw_dict.get("points") or
        []
    )

    # 3. Handle GeoJSON features if present
    features = raw_dict.get("features", [])
    if isinstance(features, list) and features:
        for f in features:
            if not isinstance(f, dict):
                continue
            geom = f.get("geometry", {}) or {}
            props = f.get("properties", {}) or {}
            g_type = geom.get("type", "")
            if g_type == "Point":
                coords = geom.get("coordinates", [85.3, 28.1])
                f_id = str(props.get("id") or props.get("name") or f"loc_{len(locations_dict) + 1}")
                locations_dict[f_id] = {
                    "id": f_id,
                    "name": str(props.get("name") or props.get("label") or f_id),
                    "type": str(props.get("type") or "settlement"),
                    "district": str(props.get("district") or props.get("adm2_name") or "Corridor"),
                    "lat": float(coords[1]) if len(coords) >= 2 else 28.1,
                    "lon": float(coords[0]) if len(coords) >= 2 else 85.3,
                    "households": int(props.get("households") or props.get("population") or 0)
                }
            elif g_type in ["LineString", "MultiLineString"]:
                coords = geom.get("coordinates", [])
                if g_type == "MultiLineString" and coords:
                    coords = coords[0]
                r_id = str(props.get("id") or f"road_{len(roads_list) + 1}")
                u = str(props.get("u") or props.get("source") or props.get("from") or f"pt_{r_id}_u")
                v = str(props.get("v") or props.get("target") or props.get("to") or f"pt_{r_id}_v")
                roads_list.append({
                    "id": r_id,
                    "name": str(props.get("name") or r_id),
                    "u": u,
                    "v": v,
                    "length_km": float(props.get("length_km") or props.get("length") or 2.0),
                    "is_bridge": bool(props.get("is_bridge", False)),
                    "status": str(props.get("status") or "unknown"),
                    "coordinates": coords
                })

    # Process standard/node-link locations
    for idx, n in enumerate(raw_nodes):
        if not isinstance(n, dict):
            continue
        n_id = str(n.get("id") or n.get("_id") or n.get("node_id") or n.get("key") or n.get("name") or f"loc_{idx+1}")
        name = str(n.get("name") or n.get("label") or n.get("title") or n_id)

        # Coordinates extraction with fallbacks
        lat = 28.1 + (idx * 0.01)
        lon = 85.3 + (idx * 0.01)
        if "lat" in n:
            lat = float(n["lat"])
        elif "latitude" in n:
            lat = float(n["latitude"])
        elif "y" in n:
            lat = float(n["y"])

        if "lon" in n:
            lon = float(n["lon"])
        elif "lng" in n:
            lon = float(n["lng"])
        elif "longitude" in n:
            lon = float(n["longitude"])
        elif "x" in n:
            lon = float(n["x"])

        if "coordinates" in n and isinstance(n["coordinates"], (list, tuple)) and len(n["coordinates"]) >= 2:
            lon = float(n["coordinates"][0])
            lat = float(n["coordinates"][1])

        households = int(n.get("households") or n.get("household_count") or n.get("population") or n.get("pop") or 0)
        district = str(n.get("district") or n.get("admin") or n.get("region") or n.get("municipality") or "Disaster Area")
        n_type = str(n.get("type") or n.get("category") or n.get("role") or "settlement")

        locations_dict[n_id] = {
            "id": n_id,
            "name": name,
            "type": n_type,
            "district": district,
            "lat": lat,
            "lon": lon,
            "households": households
        }

    # 4. Extract raw road/edge candidates
    raw_edges = (
        raw_dict.get("road_segments") or
        raw_dict.get("roads") or
        raw_dict.get("edges") or
        raw_dict.get("links") or
        raw_dict.get("lines") or
        raw_dict.get("connections") or
        []
    )

    for idx, r in enumerate(raw_edges):
        if not isinstance(r, dict):
            continue
        u = str(r.get("u") or r.get("source") or r.get("from") or r.get("start") or r.get("from_id") or r.get("node1") or "")
        v = str(r.get("v") or r.get("target") or r.get("to") or r.get("end") or r.get("to_id") or r.get("node2") or "")
        if not u or not v:
            continue

        r_id = str(r.get("id") or r.get("road_id") or r.get("edge_id") or f"road_{u}_{v}")
        name = str(r.get("name") or r.get("label") or r.get("title") or f"{u} to {v}")

        # Ensure endpoints exist in locations_dict so edges are never dropped
        if u not in locations_dict:
            locations_dict[u] = {
                "id": u,
                "name": u.replace("_", " ").title(),
                "type": "settlement",
                "district": "Disaster Area",
                "lat": 28.1 + (len(locations_dict) * 0.01),
                "lon": 85.3 + (len(locations_dict) * 0.01),
                "households": 0
            }
        if v not in locations_dict:
            locations_dict[v] = {
                "id": v,
                "name": v.replace("_", " ").title(),
                "type": "settlement",
                "district": "Disaster Area",
                "lat": 28.1 + (len(locations_dict) * 0.01),
                "lon": 85.3 + (len(locations_dict) * 0.01),
                "households": 0
            }

        coords = r.get("coordinates") or r.get("coords")
        if not coords or not isinstance(coords, list):
            coords = [
                [locations_dict[u]["lon"], locations_dict[u]["lat"]],
                [locations_dict[v]["lon"], locations_dict[v]["lat"]]
            ]

        roads_list.append({
            "id": r_id,
            "name": name,
            "u": u,
            "v": v,
            "length_km": float(r.get("length_km") or r.get("length") or r.get("distance") or r.get("weight") or 2.5),
            "is_bridge": bool(r.get("is_bridge", False)),
            "status": str(r.get("status") or "unknown"),
            "coordinates": coords,
            "bridges": r.get("bridges", []),
            "initial_observation": r.get("initial_observation")
        })

    # 5. Extract/synthesize facilities (for hospital pathfinding)
    raw_facilities = raw_dict.get("facilities", [])
    fac_ids = set()
    for f in raw_facilities:
        if isinstance(f, dict):
            f_id = str(f.get("id") or f"fac_{len(facilities_list) + 1}")
            loc_id = str(f.get("location_id") or f.get("location") or "")
            if loc_id and loc_id in locations_dict:
                facilities_list.append({
                    "id": f_id,
                    "name": str(f.get("name") or locations_dict[loc_id]["name"]),
                    "location_id": loc_id,
                    "type": str(f.get("type") or "hospital"),
                    "importance_weight": float(f.get("importance_weight") or 1.5)
                })
                fac_ids.add(loc_id)

    # Auto-detect medical facilities from locations if none provided
    if not facilities_list:
        for loc in locations_dict.values():
            l_name = loc["name"].lower()
            l_type = loc["type"].lower()
            if any(k in l_name for k in ["hospital", "clinic", "health", "aid", "medical"]) or \
               l_type in ["hospital", "clinic", "health_post", "facility_hub"]:
                facilities_list.append({
                    "id": f"fac_{loc['id']}",
                    "name": loc["name"],
                    "location_id": loc["id"],
                    "type": "hospital",
                    "importance_weight": 1.5
                })
                fac_ids.add(loc["id"])

    # If still no facilities, designate the first location as an emergency hub so path calculations succeed
    if not facilities_list and locations_dict:
        first_loc = next(iter(locations_dict.values()))
        facilities_list.append({
            "id": f"fac_{first_loc['id']}",
            "name": f"{first_loc['name']} Emergency Hub",
            "location_id": first_loc["id"],
            "type": "hospital",
            "importance_weight": 1.0
        })

    # 6. Extract bridges
    raw_bridges = raw_dict.get("bridges", [])
    for b in raw_bridges:
        if isinstance(b, dict):
            b_id = str(b.get("id") or f"bridge_{len(bridges_list) + 1}")
            bridges_list.append({
                "id": b_id,
                "name": str(b.get("name") or b_id),
                "status": str(b.get("status") or "unknown"),
                "location": str(b.get("location") or ""),
                "district": str(b.get("district") or ""),
                "lat": float(b.get("lat") or 28.1),
                "lon": float(b.get("lon") or 85.3)
            })

    return {
        "sources": sources_list,
        "locations": list(locations_dict.values()),
        "facilities": facilities_list,
        "road_segments": roads_list,
        "bridges": bridges_list,
        "initial_episodic_memory": raw_dict.get("initial_episodic_memory")
    }

def import_dataset(
    file_path: Optional[Union[Path, str]] = None,
    data_dict: Optional[Dict[str, Any]] = None,
    clear_existing: bool = True
) -> Dict[str, Any]:
    """
    Imports a topology dataset (from file or in-memory dict) into FalkorDB.
    Normalizes arbitrary JSON schemas, creates all graph nodes/edges, and calculates initial mission rankings.
    """
    target_name = "in-memory-json"
    if data_dict is not None:
        raw_data = data_dict
    else:
        target_file = Path(file_path) if file_path else resolve_default_file()
        if not target_file.exists():
            target_file = resolve_default_file()
        target_name = target_file.name
        print(f"Loading unified dataset from {target_file.name}...")
        with open(target_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    # Universally normalize dataset structure
    data = normalize_topology_dataset(raw_data)

    print(f"Connecting to FalkorDB and selecting graph '{FALKORDB_GRAPH}'...")
    client = get_falkor_client()
    graph = client.select_graph(FALKORDB_GRAPH)
    cypher_count = 0

    if clear_existing:
        print("Clearing existing graph data...")
        graph.query("MATCH (n) DETACH DELETE n")
        cypher_count += 1

    # 1. Import Sources
    sources = data.get("sources", [])
    print(f"Creating {len(sources)} Source nodes...")
    for s in sources:
        graph.query(
            "MERGE (s:Source {id: $id}) "
            "ON CREATE SET s.name = $name, s.url = $url, s.license = $license "
            "ON MATCH SET s.name = $name",
            s
        )

    # 2. Import Locations
    locations = data.get("locations", [])
    print(f"Creating {len(locations)} Location nodes...")
    for loc in locations:
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

    # 3. Import Facilities & link to Locations
    facilities = data.get("facilities", [])
    print(f"Creating {len(facilities)} Facility nodes...")
    for fac in facilities:
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

    # 4. Import RoadSegments, Bidirectional ROAD_TO edges, CONNECTS edges, and Observations
    roads = data.get("road_segments") or data.get("roads", [])
    print(f"Creating {len(roads)} RoadSegment nodes and topology edges...")
    for road in roads:
        raw_status = road.get("status", "unknown")
        initial_obs = road.get("initial_observation")
        # Epistemic safety rule: roads without explicit verified observation cannot default to verified_passable
        if raw_status == "verified_passable" and not initial_obs:
            effective_status = "unknown"
        elif initial_obs and "status" in initial_obs:
            effective_status = initial_obs["status"]
        else:
            effective_status = raw_status

        r_params = {
            "id": road["id"],
            "name": road["name"],
            "u": road["u"],
            "v": road["v"],
            "length_km": float(road.get("length_km", 1.0)),
            "is_bridge": bool(road.get("is_bridge", False)),
            "status": effective_status,
            "coordinates_json": json.dumps(road.get("coordinates", []))
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
            r_params
        )
        cypher_count += 1

        # Connect road-associated bridges if specified in road segment
        for b in road.get("bridges", []):
            b_id = f"bridge_{b['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MATCH (r:RoadSegment {id: $road_id}) "
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon "
                "MERGE (r)-[:INCLUDES_BRIDGE]->(b)",
                {
                    "road_id": road["id"],
                    "id": b_id,
                    "name": b["name"],
                    "status": b.get("status", "unknown"),
                    "location": b.get("location", ""),
                    "district": b.get("district", ""),
                    "lat": float(b.get("lat", 0.0)),
                    "lon": float(b.get("lon", 0.0))
                }
            )
            cypher_count += 1

        # Create initial observation if present
        if initial_obs:
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
                    "source_id": initial_obs.get("source_id", "src_ndrrma"),
                    "obs_id": obs_id,
                    "status": effective_status,
                    "note": initial_obs.get("note", "Initial observation"),
                    "timestamp": initial_obs.get("timestamp", datetime.now(timezone.utc).isoformat())
                }
            )
            cypher_count += 1

    # 5. Import Surveyed Bridges (from top-level master bridge collection or fallback geojson)
    bridges = data.get("bridges", [])
    if bridges:
        print(f"Indexing all {len(bridges)} surveyed bridge points in FalkorDB...")
        for b in bridges:
            graph.query(
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon",
                b
            )
            cypher_count += 1

    # 6. Import Initial Episodic Memory and Log Graph Import (Track 02)
    mem_timestamp = datetime.now(timezone.utc).isoformat()
    if "initial_episodic_memory" in data and data["initial_episodic_memory"]:
        init_mem = data["initial_episodic_memory"]
        graph.query(
            "CREATE (e:EpisodicMemory { "
            "  id: $id, "
            "  session_id: $session_id, "
            "  agent_role: $agent_role, "
            "  event_type: $event_type, "
            "  summary: $summary, "
            "  road_id: $road_id, "
            "  timestamp: $timestamp "
            "})",
            init_mem
        )
        cypher_count += 1

    # Record GRAPH_IMPORT episodic event
    import_mem = {
        "id": f"mem_import_{int(datetime.now(timezone.utc).timestamp())}",
        "session_id": "operator_import",
        "agent_role": "Field Coordinator",
        "event_type": "GRAPH_IMPORT",
        "summary": f"Imported disaster topology JSON from '{target_name}' into FalkorDB ({len(locations)} locations, {len(roads)} roads, {len(bridges)} bridges).",
        "road_id": "",
        "timestamp": mem_timestamp
    }
    graph.query(
        "CREATE (e:EpisodicMemory { "
        "  id: $id, "
        "  session_id: $session_id, "
        "  agent_role: $agent_role, "
        "  event_type: $event_type, "
        "  summary: $summary, "
        "  road_id: $road_id, "
        "  timestamp: $timestamp "
        "})",
        import_mem
    )
    cypher_count += 1

    # 7. Compute Initial Prioritized Missions
    print("Computing initial prioritized recon missions...")
    missions = recompute_missions()
    print(f"Computed {len(missions)} initial reconnaissance missions.")

    summary = {
        "status": "success",
        "source": target_name,
        "imported_counts": {
            "sources": len(sources),
            "locations": len(locations),
            "facilities": len(facilities),
            "road_segments": len(roads),
            "bridges": len(bridges),
            "active_missions": len(missions)
        },
        "cypher_statements_executed": cypher_count
    }
    print(f"FalkorDB graph ingestion completed successfully: {summary['imported_counts']}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Import disaster response dataset into FalkorDB")
    parser.add_argument("--file", "-f", type=str, default=str(DEFAULT_DATA_FILE), help="Path to dataset JSON file")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing graph data before importing")
    args = parser.parse_args()

    import_dataset(file_path=Path(args.file), clear_existing=not args.no_clear)

if __name__ == "__main__":
    main()
