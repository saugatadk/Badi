"""Ingestion module for the Badi disaster response system.

Loads the Trishuli flood corridor dataset into FalkorDB, establishing
topological relationships, facility links, bridge associations,
observational sources, and computing initial recon missions.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from backend.config import FALKORDB_GRAPH
from backend.graph import get_falkor_client
from backend.missions import recompute_missions

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_FILE = DATA_DIR / "trishuli_flood_dataset.json"


def resolve_default_file() -> Path:
    """Finds the first existing candidate dataset file in data directory."""
    for candidate in [
        DATA_DIR / "trishuli_flood_dataset.json",
        DATA_DIR / "badi_trishuli_complete_dataset.json",
        DATA_DIR / "trishuli_flood_network.json",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No baseline flood dataset found in {DATA_DIR}")


def normalize_topology_dataset(raw: Any) -> dict[str, Any]:
    """Normalizes arbitrary disaster JSON files into standard Badi topology format.

    Handles standard Badi schemas, node-link formats, and GeoJSON FeatureCollections,
    ensuring consistent keys for locations, facilities, road segments, and bridges.
    """
    if isinstance(raw, list):
        raw_dict = {"nodes": raw}
    elif isinstance(raw, dict):
        raw_dict = raw
    else:
        return {"sources": [], "locations": [], "facilities": [], "road_segments": [], "bridges": []}

    locations_dict: dict[str, dict[str, Any]] = {}
    roads_list: list[dict[str, Any]] = []
    facilities_list: list[dict[str, Any]] = []
    bridges_list: list[dict[str, Any]] = []
    sources_list: list[dict[str, Any]] = []

    # Parse observational sources
    for source_entry in raw_dict.get("sources", []):
        if isinstance(source_entry, dict):
            source_id = str(source_entry.get("id") or source_entry.get("source_id") or "src_custom")
            sources_list.append({
                "id": source_id,
                "name": str(source_entry.get("name") or source_entry.get("label") or source_id),
                "url": str(source_entry.get("url") or "local://import"),
                "license": str(source_entry.get("license") or "Open Access"),
            })

    # Ensure baseline operational sources are available
    known_source_ids = {s["id"] for s in sources_list}
    default_sources = [
        {"id": "src_ndrrma", "name": "NDRRMA Nepal", "url": "https://ndrrma.gov.np", "license": "Open Data Nepal"},
        {"id": "src_scout", "name": "Field Recon Scout", "url": "local://scout", "license": "Operational"},
        {"id": "src_user", "name": "Field Community Report", "url": "local://user", "license": "Community"},
        {"id": "src_operator", "name": "Emergency Operations Center", "url": "local://eoc", "license": "Operational"},
    ]
    for default_source in default_sources:
        if default_source["id"] not in known_source_ids:
            sources_list.append(default_source)
            known_source_ids.add(default_source["id"])

    # Extract location or settlement node candidates
    raw_nodes = (
        raw_dict.get("locations")
        or raw_dict.get("nodes")
        or raw_dict.get("vertices")
        or raw_dict.get("settlements")
        or raw_dict.get("points")
        or []
    )

    # Process GeoJSON features if present
    features = raw_dict.get("features", [])
    if isinstance(features, list) and features:
        for feature in features:
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry", {}) or {}
            properties = feature.get("properties", {}) or {}
            geom_type = geometry.get("type", "")
            if geom_type == "Point":
                coords = geometry.get("coordinates", [85.3, 28.1])
                loc_id = str(properties.get("id") or properties.get("name") or f"loc_{len(locations_dict) + 1}")
                locations_dict[loc_id] = {
                    "id": loc_id,
                    "name": str(properties.get("name") or properties.get("label") or loc_id),
                    "type": str(properties.get("type") or "settlement"),
                    "district": str(properties.get("district") or properties.get("adm2_name") or "Corridor"),
                    "lat": float(coords[1]) if len(coords) >= 2 else 28.1,
                    "lon": float(coords[0]) if len(coords) >= 2 else 85.3,
                    "households": int(properties.get("households") or properties.get("population") or 0),
                }
            elif geom_type in ["LineString", "MultiLineString"]:
                coords = geometry.get("coordinates", [])
                if geom_type == "MultiLineString" and coords:
                    coords = coords[0]
                road_id = str(properties.get("id") or f"road_{len(roads_list) + 1}")
                origin = str(properties.get("u") or properties.get("source") or properties.get("from") or f"pt_{road_id}_u")
                dest = str(properties.get("v") or properties.get("target") or properties.get("to") or f"pt_{road_id}_v")
                roads_list.append({
                    "id": road_id,
                    "name": str(properties.get("name") or road_id),
                    "u": origin,
                    "v": dest,
                    "length_km": float(properties.get("length_km") or properties.get("length") or 2.0),
                    "is_bridge": bool(properties.get("is_bridge", False)),
                    "status": str(properties.get("status") or "unknown"),
                    "coordinates": coords,
                })

    # Process standard node records
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or node.get("_id") or node.get("node_id") or node.get("key") or node.get("name") or f"loc_{index + 1}")
        name = str(node.get("name") or node.get("label") or node.get("title") or node_id)

        lat = 28.1 + (index * 0.01)
        lon = 85.3 + (index * 0.01)
        if "lat" in node:
            lat = float(node["lat"])
        elif "latitude" in node:
            lat = float(node["latitude"])
        elif "y" in node:
            lat = float(node["y"])

        if "lon" in node:
            lon = float(node["lon"])
        elif "lng" in node:
            lon = float(node["lng"])
        elif "longitude" in node:
            lon = float(node["longitude"])
        elif "x" in node:
            lon = float(node["x"])

        if "coordinates" in node and isinstance(node["coordinates"], (list, tuple)) and len(node["coordinates"]) >= 2:
            lon = float(node["coordinates"][0])
            lat = float(node["coordinates"][1])

        households = int(node.get("households") or node.get("household_count") or node.get("population") or node.get("pop") or 0)
        district = str(node.get("district") or node.get("admin") or node.get("region") or node.get("municipality") or "Disaster Area")
        node_type = str(node.get("type") or node.get("category") or node.get("role") or "settlement")

        locations_dict[node_id] = {
            "id": node_id,
            "name": name,
            "type": node_type,
            "district": district,
            "lat": lat,
            "lon": lon,
            "households": households,
        }

    # Extract road segments and connectivity
    raw_edges = (
        raw_dict.get("road_segments")
        or raw_dict.get("roads")
        or raw_dict.get("edges")
        or raw_dict.get("links")
        or raw_dict.get("lines")
        or raw_dict.get("connections")
        or []
    )

    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        origin = str(edge.get("u") or edge.get("source") or edge.get("from") or edge.get("start") or edge.get("from_id") or edge.get("node1") or "")
        dest = str(edge.get("v") or edge.get("target") or edge.get("to") or edge.get("end") or edge.get("to_id") or edge.get("node2") or "")
        if not origin or not dest:
            continue

        road_id = str(edge.get("id") or edge.get("road_id") or edge.get("edge_id") or f"road_{origin}_{dest}")
        name = str(edge.get("name") or edge.get("label") or edge.get("title") or f"{origin} to {dest}")

        # Synthesize missing endpoint locations so edges are never orphaned
        if origin not in locations_dict:
            locations_dict[origin] = {
                "id": origin,
                "name": origin.replace("_", " ").title(),
                "type": "settlement",
                "district": "Disaster Area",
                "lat": 28.1 + (len(locations_dict) * 0.01),
                "lon": 85.3 + (len(locations_dict) * 0.01),
                "households": 0,
            }
        if dest not in locations_dict:
            locations_dict[dest] = {
                "id": dest,
                "name": dest.replace("_", " ").title(),
                "type": "settlement",
                "district": "Disaster Area",
                "lat": 28.1 + (len(locations_dict) * 0.01),
                "lon": 85.3 + (len(locations_dict) * 0.01),
                "households": 0,
            }

        coords = edge.get("coordinates") or edge.get("coords")
        if not coords or not isinstance(coords, list):
            coords = [
                [locations_dict[origin]["lon"], locations_dict[origin]["lat"]],
                [locations_dict[dest]["lon"], locations_dict[dest]["lat"]],
            ]

        roads_list.append({
            "id": road_id,
            "name": name,
            "u": origin,
            "v": dest,
            "length_km": float(edge.get("length_km") or edge.get("length") or edge.get("distance") or edge.get("weight") or 2.5),
            "is_bridge": bool(edge.get("is_bridge", False)),
            "status": str(edge.get("status") or "unknown"),
            "coordinates": coords,
            "bridges": edge.get("bridges", []),
            "initial_observation": edge.get("initial_observation"),
        })

    # Extract or infer medical facilities for evacuation pathfinding
    raw_facilities = raw_dict.get("facilities", [])
    facility_location_ids = set()
    for facility_entry in raw_facilities:
        if isinstance(facility_entry, dict):
            facility_id = str(facility_entry.get("id") or f"fac_{len(facilities_list) + 1}")
            loc_id = str(facility_entry.get("location_id") or facility_entry.get("location") or "")
            if loc_id and loc_id in locations_dict:
                facilities_list.append({
                    "id": facility_id,
                    "name": str(facility_entry.get("name") or locations_dict[loc_id]["name"]),
                    "location_id": loc_id,
                    "type": str(facility_entry.get("type") or "hospital"),
                    "importance_weight": float(facility_entry.get("importance_weight") or 1.5),
                })
                facility_location_ids.add(loc_id)

    # Detect medical facilities from location names/types if none provided explicitly
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
                    "importance_weight": 1.5,
                })
                facility_location_ids.add(loc["id"])

    # If still no facilities, assign first location as default emergency hub
    if not facilities_list and locations_dict:
        first_loc = next(iter(locations_dict.values()))
        facilities_list.append({
            "id": f"fac_{first_loc['id']}",
            "name": f"{first_loc['name']} Emergency Hub",
            "location_id": first_loc["id"],
            "type": "hospital",
            "importance_weight": 1.0,
        })

    # Extract bridge structures
    raw_bridges = raw_dict.get("bridges", [])
    for bridge_entry in raw_bridges:
        if isinstance(bridge_entry, dict):
            bridge_id = str(bridge_entry.get("id") or f"bridge_{len(bridges_list) + 1}")
            bridges_list.append({
                "id": bridge_id,
                "name": str(bridge_entry.get("name") or bridge_id),
                "status": str(bridge_entry.get("status") or "unknown"),
                "location": str(bridge_entry.get("location") or ""),
                "district": str(bridge_entry.get("district") or ""),
                "lat": float(bridge_entry.get("lat") or 28.1),
                "lon": float(bridge_entry.get("lon") or 85.3),
            })

    return {
        "sources": sources_list,
        "locations": list(locations_dict.values()),
        "facilities": facilities_list,
        "road_segments": roads_list,
        "bridges": bridges_list,
        "initial_episodic_memory": raw_dict.get("initial_episodic_memory"),
    }


def import_dataset(
    file_path: Optional[Union[Path, str]] = None,
    data_dict: Optional[dict[str, Any]] = None,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Imports a disaster topology dataset into FalkorDB and computes initial missions."""
    target_name = "in-memory-json"
    if data_dict is not None:
        raw_data = data_dict
    else:
        target_file = Path(file_path) if file_path else resolve_default_file()
        if not target_file.exists():
            target_file = resolve_default_file()
        target_name = target_file.name
        logger.info("Loading unified dataset from %s...", target_file.name)
        with open(target_file, "r", encoding="utf-8") as file_handle:
            raw_data = json.load(file_handle)

    data = normalize_topology_dataset(raw_data)

    logger.info("Connecting to FalkorDB and selecting graph '%s'...", FALKORDB_GRAPH)
    client = get_falkor_client()
    graph = client.select_graph(FALKORDB_GRAPH)
    cypher_count = 0

    if clear_existing:
        logger.info("Clearing existing graph data...")
        graph.query("MATCH (n) DETACH DELETE n")
        cypher_count += 1

    # Ingest observational sources
    sources = data.get("sources", [])
    logger.info("Creating %d Source nodes...", len(sources))
    for source in sources:
        graph.query(
            "MERGE (s:Source {id: $id}) "
            "ON CREATE SET s.name = $name, s.url = $url, s.license = $license "
            "ON MATCH SET s.name = $name",
            source,
        )

    # Ingest locations
    locations = data.get("locations", [])
    logger.info("Creating %d Location nodes...", len(locations))
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

    # Ingest facilities and attach to locations
    facilities = data.get("facilities", [])
    logger.info("Creating %d Facility nodes...", len(facilities))
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
    roads = data.get("road_segments") or data.get("roads", [])
    logger.info("Creating %d RoadSegment nodes and topology edges...", len(roads))
    for road in roads:
        raw_status = road.get("status", "unknown")
        initial_obs = road.get("initial_observation")
        # Epistemic safety rule: unverified roads cannot default to verified_passable
        if raw_status == "verified_passable" and not initial_obs:
            effective_status = "unknown"
        elif initial_obs and "status" in initial_obs:
            effective_status = initial_obs["status"]
        else:
            effective_status = raw_status

        road_params = {
            "id": road["id"],
            "name": road["name"],
            "u": road["u"],
            "v": road["v"],
            "length_km": float(road.get("length_km", 1.0)),
            "is_bridge": bool(road.get("is_bridge", False)),
            "status": effective_status,
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
        cypher_count += 1

        for bridge_info in road.get("bridges", []):
            bridge_id = f"bridge_{bridge_info['name'].replace(' ', '_').replace(',', '_')[:30]}"
            graph.query(
                "MATCH (r:RoadSegment {id: $road_id}) "
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon "
                "MERGE (r)-[:INCLUDES_BRIDGE]->(b)",
                {
                    "road_id": road["id"],
                    "id": bridge_id,
                    "name": bridge_info["name"],
                    "status": bridge_info.get("status", "unknown"),
                    "location": bridge_info.get("location", ""),
                    "district": bridge_info.get("district", ""),
                    "lat": float(bridge_info.get("lat", 0.0)),
                    "lon": float(bridge_info.get("lon", 0.0)),
                },
            )
            cypher_count += 1

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
                    "timestamp": initial_obs.get("timestamp", datetime.now(timezone.utc).isoformat()),
                },
            )
            cypher_count += 1

    # Ingest surveyed bridge markers
    bridges = data.get("bridges", [])
    if bridges:
        logger.info("Indexing %d surveyed bridge points in FalkorDB...", len(bridges))
        for bridge in bridges:
            graph.query(
                "MERGE (b:Bridge {name: $name}) "
                "ON CREATE SET b.id = $id, b.status = $status, b.location = $location, "
                "              b.district = $district, b.lat = $lat, b.lon = $lon",
                bridge,
            )
            cypher_count += 1

    # Record initial episodic memory if present
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
            init_mem,
        )
        cypher_count += 1

    import_memory = {
        "id": f"mem_import_{int(datetime.now(timezone.utc).timestamp())}",
        "session_id": "operator_import",
        "agent_role": "Field Coordinator",
        "event_type": "GRAPH_IMPORT",
        "summary": (
            f"Imported disaster topology JSON from '{target_name}' into FalkorDB "
            f"({len(locations)} locations, {len(roads)} roads, {len(bridges)} bridges)."
        ),
        "road_id": "",
        "timestamp": mem_timestamp,
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
        import_memory,
    )
    cypher_count += 1

    logger.info("Computing initial prioritized recon missions...")
    missions = recompute_missions()
    logger.info("Computed %d initial reconnaissance missions.", len(missions))

    summary = {
        "status": "success",
        "source": target_name,
        "imported_counts": {
            "sources": len(sources),
            "locations": len(locations),
            "facilities": len(facilities),
            "road_segments": len(roads),
            "bridges": len(bridges),
            "active_missions": len(missions),
        },
        "cypher_statements_executed": cypher_count,
    }
    logger.info("FalkorDB graph ingestion completed successfully: %s", summary["imported_counts"])
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Import disaster response dataset into FalkorDB")
    parser.add_argument("--file", "-f", type=str, default=str(DEFAULT_DATA_FILE), help="Path to dataset JSON file")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing graph data before importing")
    args = parser.parse_args()

    import_dataset(file_path=Path(args.file), clear_existing=not args.no_clear)


if __name__ == "__main__":
    main()
