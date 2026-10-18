import difflib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.graph import get_graph
from backend.models import (
    EpisodicMemoryModel,
    FacilityModel,
    KPISummary,
    LocationModel,
    ObservationModel,
    PathInfo,
    RoadSegmentModel,
    RoadStatus,
    SettlementAccessState,
)

logger = logging.getLogger(__name__)


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_path_info(path: Any, facility_id: str, facility_name: str, facility_weight: float) -> PathInfo:
    """Extracts structured path details from a FalkorDB Cypher path object."""
    nodes = path.nodes() if callable(getattr(path, "nodes", None)) else path.nodes
    edges = path.edges() if callable(getattr(path, "edges", None)) else path.edges

    location_ids = [n.properties.get("id", "") for n in nodes]
    location_names = [n.properties.get("name", "") for n in nodes]
    road_ids = [e.properties.get("road_id", "") for e in edges]
    road_names = [e.properties.get("road_name", "") for e in edges]
    total_km = round(sum(float(e.properties.get("length_km", 0.0)) for e in edges), 2)
    unverified_road_ids = [
        e.properties.get("road_id", "")
        for e in edges
        if e.properties.get("status") in ["unknown", "disputed"]
    ]

    return PathInfo(
        locations=location_ids,
        location_names=location_names,
        road_ids=road_ids,
        road_names=road_names,
        total_km=total_km,
        facility_id=facility_id,
        facility_name=facility_name,
        unverified_road_ids=unverified_road_ids,
    )


def find_verified_paths(settlement_id: str) -> Tuple[List[PathInfo], str]:
    """Finds all paths to any facility where all edges are verified passable."""
    cypher = (
        "MATCH p = (s:Location {id: $settlement_id})-[:ROAD_TO*1..7]->(hub:Location) "
        "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
        "WHERE ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable') "
        "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "RETURN p, f.id AS facility_id, f.name AS facility_name, f.importance_weight AS facility_weight "
        "ORDER BY length(p) ASC"
    )
    graph = get_graph()
    query_result = graph.query(cypher, {"settlement_id": settlement_id})
    paths: List[PathInfo] = []
    if query_result and query_result.result_set:
        for row in query_result.result_set:
            path_obj, fac_id, fac_name, fac_weight = row[0], str(row[1]), str(row[2]), float(row[3])
            paths.append(_extract_path_info(path_obj, fac_id, fac_name, fac_weight))
    return paths, cypher


def find_potential_paths(settlement_id: str) -> Tuple[List[PathInfo], str]:
    """Finds paths where no edges are blocked, but at least one is unknown or disputed."""
    cypher = (
        "MATCH p = (s:Location {id: $settlement_id})-[:ROAD_TO*1..7]->(hub:Location) "
        "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
        "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked') "
        "  AND ANY(rel IN relationships(p) WHERE rel.status IN ['unknown', 'disputed']) "
        "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "RETURN p, f.id AS facility_id, f.name AS facility_name, f.importance_weight AS facility_weight "
        "ORDER BY length(p) ASC"
    )
    graph = get_graph()
    query_result = graph.query(cypher, {"settlement_id": settlement_id})
    paths: List[PathInfo] = []
    if query_result and query_result.result_set:
        for row in query_result.result_set:
            path_obj, fac_id, fac_name, fac_weight = row[0], str(row[1]), str(row[2]), float(row[3])
            paths.append(_extract_path_info(path_obj, fac_id, fac_name, fac_weight))
    return paths, cypher


def classify_settlement_access(
    settlement_id: str,
) -> Tuple[SettlementAccessState, str, List[PathInfo], List[PathInfo]]:
    """Classifies settlement access into verified, needs-recon, isolated, or unmapped."""
    graph = get_graph()

    node_check = graph.query("MATCH (s:Location {id: $id}) RETURN s", {"id": settlement_id})
    if not node_check or not node_check.result_set:
        return "UNAVAILABLE", f"Settlement node '{settlement_id}' was not found in the graph database.", [], []

    node = node_check.result_set[0][0]
    node_props = node.properties if hasattr(node, "properties") else getattr(node, "_properties", {})
    settlement_name = node_props.get("name", settlement_id) if isinstance(node_props, dict) else settlement_id

    verified_paths, _ = find_verified_paths(settlement_id)
    potential_paths, _ = find_potential_paths(settlement_id)

    if verified_paths:
        reason = f"Verified passable route available to {verified_paths[0].facility_name} ({verified_paths[0].total_km} km)"
        return "VERIFIED_ACCESS", reason, verified_paths, potential_paths
    elif potential_paths:
        reason = f"No verified paths. {len(potential_paths)} potential route(s) require physical reconnaissance"
        return "NEEDS_RECON", reason, verified_paths, potential_paths
    else:
        edge_check = graph.query("MATCH (s:Location {id: $id})-[:ROAD_TO]-() RETURN count(*) AS edges", {"id": settlement_id})
        has_edges = edge_check and edge_check.result_set and edge_check.result_set[0][0] > 0
        if not has_edges:
            return "NO_MAPPED_ROUTE", f"No mapped road connections exist for {settlement_name} ({settlement_id}) in the network graph.", [], []

        any_path = graph.query(
            "MATCH p = (s:Location {id: $id})-[:ROAD_TO*1..7]->(hub:Location)<-[:AT_LOCATION]-(f:Facility) RETURN count(p) LIMIT 1",
            {"id": settlement_id},
        )
        if any_path and any_path.result_set and any_path.result_set[0][0] > 0:
            reason = "All topological routes to healthcare facilities are blocked by flood surges or debris"
            return "ISOLATED_ON_MAP", reason, verified_paths, potential_paths
        else:
            reason = f"No mapped routes connect {settlement_name} to any healthcare facility within the 7-hop horizon."
            return "NO_MAPPED_ROUTE", reason, verified_paths, potential_paths


def find_affected_settlements(road_id: str) -> Tuple[List[Dict[str, Any]], str]:
    """Finds all settlements whose non-blocked paths to any facility traverse road_id."""
    cypher = (
        "MATCH p = (s:Location {type: 'settlement'})-[:ROAD_TO*1..7]->(hub:Location) "
        "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
        "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked') "
        "  AND ANY(rel IN relationships(p) WHERE rel.road_id = $road_id) "
        "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "RETURN DISTINCT s.id AS settlement_id, s.name AS settlement_name, s.households AS households "
        "ORDER BY s.households DESC"
    )
    graph = get_graph()
    query_result = graph.query(cypher, {"road_id": road_id})
    affected: List[Dict[str, Any]] = []
    if query_result and query_result.result_set:
        for row in query_result.result_set:
            affected.append({
                "id": str(row[0]),
                "name": str(row[1]),
                "households": int(row[2]) if row[2] is not None else 0,
            })
    return affected, cypher


def get_road_evidence(road_id: str) -> Dict[str, Any]:
    """Retrieves full evidence chain for a road segment."""
    graph = get_graph()
    cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "OPTIONAL MATCH (u:Location {id: r.u}) "
        "OPTIONAL MATCH (v:Location {id: r.v}) "
        "RETURN r, [u, v] AS endpoints"
    )
    road_result = graph.query(cypher, {"road_id": road_id})
    if not road_result or not road_result.result_set:
        raise ValueError(f"RoadSegment with id '{road_id}' not found in FalkorDB")

    road_node = road_result.result_set[0][0]
    endpoints_raw = road_result.result_set[0][1]

    endpoints = [
        LocationModel(
            id=loc.properties["id"],
            name=loc.properties["name"],
            type=loc.properties["type"],
            district=loc.properties["district"],
            lat=float(loc.properties["lat"]),
            lon=float(loc.properties["lon"]),
            households=int(loc.properties.get("households", 0)),
        )
        for loc in endpoints_raw
        if loc
    ]

    obs_cypher = (
        "MATCH (r:RoadSegment {id: $road_id})-[:HAS_OBSERVATION]->(o:Observation)-[:FROM_SOURCE]->(s:Source) "
        "RETURN o, s "
        "ORDER BY o.timestamp DESC"
    )
    obs_res = graph.query(obs_cypher, {"road_id": road_id})
    observations = []
    if obs_res and obs_res.result_set:
        for row in obs_res.result_set:
            o_node, s_node = row[0], row[1]
            observations.append(ObservationModel(
                id=o_node.properties["id"],
                road_id=o_node.properties["road_id"],
                source_id=s_node.properties["id"],
                source_name=s_node.properties["name"],
                status=o_node.properties["status"],
                note=o_node.properties["note"],
                timestamp=o_node.properties["timestamp"],
                is_simulated=bool(o_node.properties.get("is_simulated", False)),
            ))

    mem_cypher = (
        "MATCH (e:EpisodicMemory)-[:AFFECTED_ROAD]->(r:RoadSegment {id: $road_id}) "
        "RETURN e "
        "ORDER BY e.timestamp DESC"
    )
    mem_res = graph.query(mem_cypher, {"road_id": road_id})
    memories = []
    if mem_res and mem_res.result_set:
        for row in mem_res.result_set:
            e_node = row[0]
            memories.append(EpisodicMemoryModel(
                id=e_node.properties["id"],
                session_id=e_node.properties["session_id"],
                agent_role=e_node.properties["agent_role"],
                event_type=e_node.properties["event_type"],
                summary=e_node.properties["summary"],
                road_id=e_node.properties.get("road_id"),
                timestamp=e_node.properties["timestamp"],
            ))

    bridge_cypher = (
        "MATCH (r:RoadSegment {id: $road_id})-[:INCLUDES_BRIDGE]->(b:Bridge) "
        "RETURN b "
        "ORDER BY b.name ASC"
    )
    bridge_res = graph.query(bridge_cypher, {"road_id": road_id})
    road_bridges = []
    if bridge_res and bridge_res.result_set:
        for row in bridge_res.result_set:
            b_node = row[0]
            road_bridges.append({
                "id": b_node.properties.get("id", ""),
                "name": b_node.properties.get("name", ""),
                "status": b_node.properties.get("status", "Unknown"),
                "location": b_node.properties.get("location", ""),
                "district": b_node.properties.get("district", ""),
                "lat": float(b_node.properties.get("lat", 0.0)),
                "lon": float(b_node.properties.get("lon", 0.0)),
            })

    affected_settlements, aff_cypher = find_affected_settlements(road_id)

    coords = None
    if "coordinates_json" in road_node.properties and road_node.properties["coordinates_json"]:
        try:
            coords = json.loads(road_node.properties["coordinates_json"])
        except Exception as exc:
            logger.warning("Could not parse coordinates_json for road %s: %s", road_id, exc)

    road_model = RoadSegmentModel(
        id=road_node.properties["id"],
        name=road_node.properties["name"],
        u=road_node.properties["u"],
        v=road_node.properties["v"],
        length_km=float(road_node.properties["length_km"]),
        is_bridge=bool(road_node.properties["is_bridge"]),
        status=road_node.properties["status"],
        observations=observations,
        coordinates=coords,
        bridges=road_bridges,
        bridge_count=len(road_bridges),
    )

    full_cypher = f"{cypher}\n-- Observations:\n{obs_cypher}\n-- Episodic Memory:\n{mem_cypher}\n-- Bridges:\n{bridge_cypher}\n-- Downstream:\n{aff_cypher}"

    return {
        "road": road_model,
        "endpoints": endpoints,
        "observations": observations,
        "memories": memories,
        "affected_settlements": affected_settlements,
        "cypher_executed": full_cypher,
    }


def get_decision_chain(road_id: str) -> Dict[str, Any]:
    """Traverses sources, observations, episodic memories, and missions for a road."""
    graph = get_graph()
    cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "OPTIONAL MATCH (r)-[:CONNECTS]->(loc:Location) "
        "OPTIONAL MATCH (r)-[:HAS_OBSERVATION]->(o:Observation) "
        "OPTIONAL MATCH (o)-[:FROM_SOURCE]->(s:Source) "
        "OPTIONAL MATCH (e:EpisodicMemory)-[:AFFECTED_ROAD]->(r) "
        "OPTIONAL MATCH (m:Mission)-[:INVESTIGATES]->(r) "
        "RETURN r.id AS road_id, r.name AS road_name, r.status AS road_status, "
        "       collect(DISTINCT loc.name) AS connected_settlements, "
        "       collect(DISTINCT {id: s.id, name: s.name}) AS sources, "
        "       collect(DISTINCT {id: o.id, status: o.status, note: o.note, timestamp: o.timestamp}) AS observations, "
        "       collect(DISTINCT {id: e.id, agent_role: e.agent_role, event_type: e.event_type, summary: e.summary, timestamp: e.timestamp}) AS memories, "
        "       collect(DISTINCT {id: m.id, priority_score: m.priority_score, reason: m.reason}) AS missions"
    )
    chain_result = graph.query(cypher, {"road_id": road_id})
    if not chain_result or not chain_result.result_set:
        raise ValueError(f"RoadSegment with id '{road_id}' not found in FalkorDB")

    row = chain_result.result_set[0]
    return {
        "road_id": row[0],
        "road_name": row[1],
        "road_status": row[2],
        "connected_settlements": [s for s in row[3] if s],
        "people_sources": [s for s in row[4] if s.get("id")],
        "observations": [o for o in row[5] if o.get("id")],
        "decisions_memories": [m for m in row[6] if m.get("id")],
        "project_missions": [m for m in row[7] if m.get("id")],
        "cypher_executed": cypher,
    }


def add_observation_and_sync(
    road_id: str,
    reported_status: RoadStatus,
    note: str,
    source_id: str = "src_scout",
    is_simulated: bool = False,
    override: bool = False,
    session_id: str = "ops_session",
    agent_role: str = "Field Coordinator",
) -> Dict[str, Any]:
    """Records an observation, resolves consensus status, and syncs graph edges."""
    from backend.missions import recompute_missions

    graph = get_graph()
    obs_id = f"obs_{uuid.uuid4().hex[:8]}"
    ts = _current_timestamp()

    road_check = graph.query("MATCH (r:RoadSegment {id: $road_id}) RETURN r.id", {"road_id": road_id})
    if not road_check or not road_check.result_set:
        raise ValueError(f"Road corridor '{road_id}' not found in FalkorDB. Cannot record observation.")

    graph.query(
        "MERGE (s:Source {id: $source_id}) "
        "ON CREATE SET s.name = CASE $source_id "
        "  WHEN 'src_operator' THEN 'Operator Console Submission' "
        "  WHEN 'src_user' THEN 'Community Field Report' "
        "  WHEN 'src_scout' THEN 'Field Recon Scout Report' "
        "  ELSE $source_id END",
        {"source_id": source_id},
    )
    create_obs_cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "MATCH (s:Source {id: $source_id}) "
        "CREATE (o:Observation { "
        "  id: $obs_id, "
        "  road_id: $road_id, "
        "  status: $reported_status, "
        "  note: $note, "
        "  timestamp: $timestamp, "
        "  is_simulated: $is_simulated "
        "}) "
        "CREATE (r)-[:HAS_OBSERVATION]->(o) "
        "CREATE (o)-[:FROM_SOURCE]->(s) "
        "RETURN o"
    )
    create_obs_result = graph.query(create_obs_cypher, {
        "road_id": road_id,
        "source_id": source_id,
        "obs_id": obs_id,
        "reported_status": reported_status,
        "note": note,
        "timestamp": ts,
        "is_simulated": is_simulated,
    })
    if not create_obs_result or not create_obs_result.result_set:
        raise ValueError(f"Failed to record observation on road '{road_id}' from source '{source_id}'")

    all_obs_cypher = (
        "MATCH (r:RoadSegment {id: $road_id})-[:HAS_OBSERVATION]->(o:Observation) "
        "RETURN o.status AS status, o.source_id AS source_id"
    )
    all_obs_result = graph.query(all_obs_cypher, {"road_id": road_id})
    statuses = set()
    if all_obs_result and all_obs_result.result_set:
        for r in all_obs_result.result_set:
            statuses.add(r[0])

    # Consensus resolution rules
    if (source_id == "src_scout" and override) or override:
        new_status = reported_status
        resolution_reason = "Authoritative field scout or operator with override authority resolved dispute"
    elif "blocked" in statuses and "verified_passable" in statuses:
        new_status = "disputed"
        resolution_reason = "Conflicting field observations detected: marked as disputed pending ground recon"
    elif statuses == {"blocked"}:
        new_status = "blocked"
        resolution_reason = "Consensus: all observations confirm road is blocked"
    elif statuses == {"verified_passable"}:
        new_status = "verified_passable"
        resolution_reason = "Consensus: field observations verify road is passable"
    elif statuses == {"unknown"}:
        new_status = "unknown"
        resolution_reason = "No confirmed observations; remains unknown"
    else:
        new_status = reported_status
        resolution_reason = f"Status updated to {reported_status}"

    sync_cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "SET r.status = $new_status "
        "WITH r "
        "OPTIONAL MATCH ()-[rel:ROAD_TO {road_id: $road_id}]->() "
        "SET rel.status = $new_status "
        "RETURN r.id AS road_id, r.status AS status"
    )
    graph.query(sync_cypher, {"road_id": road_id, "new_status": new_status})

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    summary = (
        f"Observation recorded on {road_id}: reported '{reported_status}' from '{source_id}'. "
        f"Resolved road status to '{new_status}'. Reason: {resolution_reason}. Note: {note}"
    )
    mem_cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "CREATE (e:EpisodicMemory { "
        "  id: $memory_id, "
        "  session_id: $session_id, "
        "  agent_role: $agent_role, "
        "  event_type: 'observation_recorded', "
        "  summary: $summary, "
        "  road_id: $road_id, "
        "  timestamp: $timestamp "
        "}) "
        "CREATE (e)-[:AFFECTED_ROAD]->(r) "
        "RETURN e"
    )
    graph.query(mem_cypher, {
        "road_id": road_id,
        "memory_id": memory_id,
        "session_id": session_id,
        "agent_role": agent_role,
        "summary": summary,
        "timestamp": ts,
    })

    missions = recompute_missions()

    return {
        "observation_id": obs_id,
        "road_id": road_id,
        "new_status": new_status,
        "resolution_reason": resolution_reason,
        "memory_id": memory_id,
        "missions_count": len(missions),
        "timestamp": ts,
    }


def record_episodic_memory(
    session_id: str,
    agent_role: str,
    event_type: str,
    summary: str,
    road_id: Optional[str] = None,
) -> str:
    """Stores an episodic memory event in FalkorDB."""
    graph = get_graph()
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    ts = _current_timestamp()

    if road_id:
        cypher = (
            "MATCH (r:RoadSegment {id: $road_id}) "
            "CREATE (e:EpisodicMemory { "
            "  id: $memory_id, "
            "  session_id: $session_id, "
            "  agent_role: $agent_role, "
            "  event_type: $event_type, "
            "  summary: $summary, "
            "  road_id: $road_id, "
            "  timestamp: $timestamp "
            "}) "
            "CREATE (e)-[:AFFECTED_ROAD]->(r) "
            "RETURN e.id"
        )
        graph.query(cypher, {
            "road_id": road_id,
            "memory_id": memory_id,
            "session_id": session_id,
            "agent_role": agent_role,
            "event_type": event_type,
            "summary": summary,
            "timestamp": ts,
        })
    else:
        cypher = (
            "CREATE (e:EpisodicMemory { "
            "  id: $memory_id, "
            "  session_id: $session_id, "
            "  agent_role: $agent_role, "
            "  event_type: $event_type, "
            "  summary: $summary, "
            "  road_id: '', "
            "  timestamp: $timestamp "
            "}) "
            "RETURN e.id"
        )
        graph.query(cypher, {
            "memory_id": memory_id,
            "session_id": session_id,
            "agent_role": agent_role,
            "event_type": event_type,
            "summary": summary,
            "timestamp": ts,
        })

    return memory_id


def get_episodic_memories(limit: int = 50, road_id: Optional[str] = None) -> List[EpisodicMemoryModel]:
    """Retrieves chronological timeline of episodic memories from FalkorDB."""
    graph = get_graph()
    if road_id:
        cypher = (
            "MATCH (e:EpisodicMemory)-[:AFFECTED_ROAD]->(r:RoadSegment {id: $road_id}) "
            "RETURN e "
            "ORDER BY e.timestamp DESC "
            "LIMIT $limit"
        )
        memory_result = graph.query(cypher, {"road_id": road_id, "limit": limit})
    else:
        cypher = (
            "MATCH (e:EpisodicMemory) "
            "RETURN e "
            "ORDER BY e.timestamp DESC "
            "LIMIT $limit"
        )
        memory_result = graph.query(cypher, {"limit": limit})

    memories = []
    if memory_result and memory_result.result_set:
        for row in memory_result.result_set:
            e = row[0]
            memories.append(EpisodicMemoryModel(
                id=e.properties.get("id", ""),
                session_id=e.properties.get("session_id", ""),
                agent_role=e.properties.get("agent_role", ""),
                event_type=e.properties.get("event_type", ""),
                summary=e.properties.get("summary", ""),
                road_id=e.properties.get("road_id") or None,
                timestamp=e.properties.get("timestamp", ""),
            ))
    return memories


def get_network_snapshot() -> Dict[str, Any]:
    """Fetches the complete current state of the flood network directly from FalkorDB."""
    graph = get_graph()

    loc_result = graph.query("MATCH (l:Location) RETURN l ORDER BY l.name ASC")
    locations_dict: Dict[str, Dict[str, Any]] = {}
    if loc_result and loc_result.result_set:
        for row in loc_result.result_set:
            loc_props = row[0].properties
            locations_dict[loc_props["id"]] = {
                "id": loc_props["id"],
                "name": loc_props["name"],
                "type": loc_props["type"],
                "district": loc_props["district"],
                "lat": float(loc_props["lat"]),
                "lon": float(loc_props["lon"]),
                "households": int(loc_props.get("households", 0)),
                "access_state": None,
                "access_reason": None,
                "verified_paths_count": 0,
                "potential_paths_count": 0,
            }

    settlements = [loc for loc in locations_dict.values() if loc["type"] == "settlement"]
    verified_settlements = 0
    needs_recon_settlements = 0
    isolated_settlements = 0

    for s in settlements:
        state, reason, v_paths, p_paths = classify_settlement_access(s["id"])
        s["access_state"] = state
        s["access_reason"] = reason
        s["verified_paths_count"] = len(v_paths)
        s["potential_paths_count"] = len(p_paths)
        if state == "VERIFIED_ACCESS":
            verified_settlements += 1
        elif state == "NEEDS_RECON":
            needs_recon_settlements += 1
        elif state == "ISOLATED_ON_MAP":
            isolated_settlements += 1

    fac_result = graph.query("MATCH (f:Facility) RETURN f")
    facilities = []
    if fac_result and fac_result.result_set:
        for row in fac_result.result_set:
            f = row[0].properties
            facilities.append(FacilityModel(
                id=f["id"],
                name=f["name"],
                location_id=f["location_id"],
                type=f["type"],
                importance_weight=float(f["importance_weight"]),
            ))

    # Pre-fetch bridges grouped by road to avoid N+1 queries
    bridges_by_road: Dict[str, List[Dict[str, Any]]] = {}
    bridges_result = graph.query(
        "MATCH (r:RoadSegment)-[:INCLUDES_BRIDGE]->(b:Bridge) "
        "RETURN r.id AS road_id, b.name AS name, b.status AS status, b.lat AS lat, b.lon AS lon"
    )
    if bridges_result and bridges_result.result_set:
        for b_row in bridges_result.result_set:
            brid = str(b_row[0])
            if brid not in bridges_by_road:
                bridges_by_road[brid] = []
            bridges_by_road[brid].append({
                "name": b_row[1],
                "status": b_row[2],
                "lat": float(b_row[3]),
                "lon": float(b_row[4]),
            })

    roads_result = graph.query("MATCH (r:RoadSegment) RETURN r")
    roads = []
    verified_roads = 0
    unknown_roads = 0
    blocked_roads = 0
    disputed_roads = 0

    if roads_result and roads_result.result_set:
        for row in roads_result.result_set:
            r = row[0].properties
            status = r["status"]
            if status == "verified_passable":
                verified_roads += 1
            elif status == "unknown":
                unknown_roads += 1
            elif status == "blocked":
                blocked_roads += 1
            elif status == "disputed":
                disputed_roads += 1

            coords = None
            if "coordinates_json" in r and r["coordinates_json"]:
                try:
                    coords = json.loads(r["coordinates_json"])
                except Exception as exc:
                    logger.warning("Could not parse coordinates_json: %s", exc)

            if not coords:
                u_loc = locations_dict.get(r["u"])
                v_loc = locations_dict.get(r["v"])
                if u_loc and v_loc:
                    coords = [[u_loc["lon"], u_loc["lat"]], [v_loc["lon"], v_loc["lat"]]]

            road_bridges = bridges_by_road.get(r["id"], [])

            roads.append(RoadSegmentModel(
                id=r["id"],
                name=r["name"],
                u=r["u"],
                v=r["v"],
                length_km=float(r["length_km"]),
                is_bridge=bool(r["is_bridge"]),
                status=status,
                coordinates=coords,
                bridges=road_bridges,
                bridge_count=len(road_bridges),
            ))

    mission_result = graph.query("MATCH (m:Mission) RETURN count(m)")
    active_missions = int(mission_result.result_set[0][0]) if mission_result and mission_result.result_set else 0

    memory_result = graph.query("MATCH (e:EpisodicMemory) RETURN count(e)")
    total_memories = int(memory_result.result_set[0][0]) if memory_result and memory_result.result_set else 0

    kpis = KPISummary(
        verified_settlements=verified_settlements,
        needs_recon_settlements=needs_recon_settlements,
        isolated_settlements=isolated_settlements,
        verified_roads=verified_roads,
        unknown_roads=unknown_roads,
        blocked_roads=blocked_roads,
        disputed_roads=disputed_roads,
        active_missions=active_missions,
        episodic_memories=total_memories,
    )

    location_models = [LocationModel(**loc) for loc in locations_dict.values()]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    bldg_summary_path = data_dir / "building_damage_summary.json"
    bldg_summary = None
    if bldg_summary_path.exists():
        try:
            with open(bldg_summary_path, "r", encoding="utf-8") as f:
                bldg_summary = json.load(f)
        except Exception as exc:
            logger.warning("Could not read building_damage_summary.json: %s", exc)

    bridge_summary_result = graph.query(
        "MATCH (b:Bridge) "
        "RETURN b.status, count(b)"
    )
    bridge_summary = {}
    if bridge_summary_result and bridge_summary_result.result_set:
        for b_row in bridge_summary_result.result_set:
            bridge_summary[b_row[0]] = int(b_row[1])

    is_empty = len(location_models) == 0

    return {
        "locations": location_models,
        "facilities": facilities,
        "road_segments": roads,
        "kpis": kpis,
        "is_empty": is_empty,
        "building_summary": bldg_summary,
        "bridge_summary": bridge_summary,
        "timestamp": _current_timestamp(),
    }


def wipe_falkordb_graph() -> Dict[str, Any]:
    """Wipes the FalkorDB graph, removing all nodes and edges."""
    graph = get_graph()
    count_result = graph.query("MATCH (n) RETURN count(n)")
    prior_nodes = int(count_result.result_set[0][0]) if count_result and count_result.result_set else 0
    graph.query("MATCH (n) DETACH DELETE n")
    return {
        "status": "wiped",
        "nodes_deleted": prior_nodes,
        "is_empty": True,
        "timestamp": _current_timestamp(),
    }


def get_bridge_damage_summary() -> Dict[str, Any]:
    """Queries damage status counts of all surveyed bridges."""
    graph = get_graph()
    cypher = "MATCH (b:Bridge) RETURN b.status AS status, count(b) AS count"
    try:
        damage_result = graph.query(cypher)
        if not damage_result or not damage_result.result_set:
            return {
                "status": "unavailable",
                "reason": "No surveyed bridge records found in FalkorDB graph.",
                "counts": {},
                "total": 0,
                "cypher_executed": cypher,
            }
        counts = {str(row[0]): int(row[1]) for row in damage_result.result_set}
        return {
            "status": "ok",
            "counts": counts,
            "total": sum(counts.values()),
            "cypher_executed": cypher,
        }
    except Exception as e:
        logger.warning("FalkorDB bridge summary query error: %s", e)
        return {
            "status": "error",
            "reason": f"FalkorDB query error: {str(e)}",
            "counts": {},
            "total": 0,
            "cypher_executed": cypher,
        }


def get_on_site_facilities(settlement_id: str) -> List[Dict[str, Any]]:
    """Returns healthcare facilities situated directly at a settlement."""
    if not settlement_id:
        return []
    graph = get_graph()
    cypher = (
        "MATCH (f:Facility)-[:AT_LOCATION]->(l:Location {id: $settlement_id}) "
        "RETURN f.id, f.name, f.type, f.importance_weight"
    )
    facility_result = graph.query(cypher, {"settlement_id": settlement_id})
    facs = []
    if facility_result and facility_result.result_set:
        for row in facility_result.result_set:
            facs.append({
                "id": str(row[0]),
                "name": str(row[1]),
                "type": str(row[2]) if row[2] else "facility",
                "importance_weight": float(row[3]) if row[3] is not None else 1.0,
            })
    return facs


LOCATION_ALIASES: Dict[str, str] = {
    "syabrubesi": "loc_syabrubesi",
    "timure": "loc_timure",
    "briddhim": "loc_briddhim",
    "bridhim": "loc_briddhim",
    "goljung": "loc_goljung",
    "gatlang": "loc_gatlang",
    "dhunche": "loc_dhunche",
    "mailung": "loc_mailung",
    "kispang": "loc_kispang",
    "kalikasthan": "loc_kalikasthan",
    "betrawati": "loc_betrawati",
    "bidur": "loc_bidur",
    "devighat": "loc_devighat",
    "trishuli": "loc_bidur",
    "trisuli": "loc_bidur",
    # Facility aliases
    "betrawati hospital": "loc_betrawati",
    "betrawati health post": "loc_betrawati",
    "betrawati clinic": "loc_betrawati",
    "trishuli hospital": "loc_bidur",
    "trisuli hospital": "loc_bidur",
    "bidur hospital": "loc_bidur",
    "rasuwa hospital": "loc_dhunche",
    "rasuwa district hospital": "loc_dhunche",
    "dhunche hospital": "loc_dhunche",
}

ROAD_ALIASES: Dict[str, str] = {
    "syabrubesi-dhunche": "road_syabru_dhunche",
    "syabru_dhunche": "road_syabru_dhunche",
    "pasang lhamu": "road_syabru_dhunche",
    "timure-syabru": "road_timure_syabru",
    "timure_syabru": "road_timure_syabru",
    "briddhim-syabru": "road_briddhim_syabru",
    "bridhim-syabru": "road_briddhim_syabru",
    "syabru-goljung": "road_syabru_goljung",
    "syabru_goljung": "road_syabru_goljung",
    "goljung-gatlang": "road_goljung_gatlang",
    "gatlang-dhunche": "road_gatlang_dhunche",
    "dhunche-kalika": "road_dhunche_kalika",
    "kalika-betrawati": "road_kalika_betrawati",
    "mailung-betrawati": "road_mailung_betrawati",
    "gatlang-kispang": "road_gatlang_kispang",
    "kispang-betrawati": "road_kispang_betrawati",
    "betrawati-bidur": "road_betrawati_bidur",
    "devighat-bidur": "road_devighat_bidur",
}

SETTLEMENT_CANONICAL_NAMES: Dict[str, str] = {
    "loc_syabrubesi": "Syabrubesi",
    "loc_timure": "Timure",
    "loc_briddhim": "Briddhim",
    "loc_goljung": "Goljung",
    "loc_gatlang": "Gatlang",
    "loc_dhunche": "Dhunche",
    "loc_mailung": "Mailung",
    "loc_kispang": "Kispang",
    "loc_kalikasthan": "Kalikasthan",
    "loc_betrawati": "Betrawati",
    "loc_bidur": "Bidur / Trishuli",
    "loc_devighat": "Devighat",
}


def resolve_settlement_id(name_or_alias: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolves a settlement identifier or alias. Returns (id, display_name, candidates)."""
    if not name_or_alias or not isinstance(name_or_alias, str):
        return None, None, list(LOCATION_ALIASES.keys())

    clean = name_or_alias.strip().lower()

    if clean.startswith("loc_"):
        if clean in SETTLEMENT_CANONICAL_NAMES:
            return clean, SETTLEMENT_CANONICAL_NAMES[clean], []
        for alias, lid in LOCATION_ALIASES.items():
            if lid == clean:
                return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, alias.capitalize()), []
        try:
            graph = get_graph()
            check = graph.query("MATCH (s:Location {id: $id}) RETURN s.name", {"id": clean})
            if check and check.result_set:
                return clean, check.result_set[0][0], []
        except Exception as exc:
            logger.debug("Failed direct settlement lookup for %s: %s", clean, exc)
        return None, None, list(LOCATION_ALIASES.keys())[:5]

    if clean in LOCATION_ALIASES:
        lid = LOCATION_ALIASES[clean]
        return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, clean.capitalize()), []

    candidates = []
    for alias, lid in LOCATION_ALIASES.items():
        if alias in clean or clean in alias:
            candidates.append((alias, lid))

    unique_lids = list(dict.fromkeys(c[1] for c in candidates))
    if len(unique_lids) == 1:
        lid = unique_lids[0]
        return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, candidates[0][0].capitalize()), []
    elif len(unique_lids) > 1:
        return None, None, [c[0].capitalize() for c in candidates]

    fuzzy = difflib.get_close_matches(clean, list(LOCATION_ALIASES.keys()), n=3, cutoff=0.6)
    if len(fuzzy) == 1:
        matched_alias = fuzzy[0]
        lid = LOCATION_ALIASES[matched_alias]
        return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, matched_alias.capitalize()), []

    return None, None, [f.capitalize() for f in fuzzy] if fuzzy else list(LOCATION_ALIASES.keys())[:5]


def resolve_road_id(name_or_alias: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolves a road corridor identifier or alias. Returns (road_id, display_name, candidates)."""
    if not name_or_alias or not isinstance(name_or_alias, str):
        return None, None, list(ROAD_ALIASES.keys())

    clean = name_or_alias.strip().lower()

    if clean.startswith("road_"):
        for alias, rid in ROAD_ALIASES.items():
            if rid == clean:
                return rid, alias.replace("-", "–").title(), []
        try:
            graph = get_graph()
            check = graph.query("MATCH (r:RoadSegment {id: $id}) RETURN r.name", {"id": clean})
            if check and check.result_set:
                return clean, check.result_set[0][0], []
        except Exception as exc:
            logger.debug("Failed direct road lookup for %s: %s", clean, exc)
        return None, None, list(ROAD_ALIASES.keys())[:5]

    if clean in ROAD_ALIASES:
        return ROAD_ALIASES[clean], clean.replace("-", "–").title(), []

    # Common corridor endpoint pairs
    if ("briddhim" in clean or "bridhim" in clean) and "syabru" in clean:
        return "road_briddhim_syabru", "Briddhim–Syabrubesi", []
    if "timure" in clean and "syabru" in clean:
        return "road_timure_syabru", "Timure–Syabrubesi", []
    if "syabru" in clean and "dhunche" in clean:
        return "road_syabru_dhunche", "Syabrubesi–Dhunche (Pasang Lhamu)", []
    if "syabru" in clean and "goljung" in clean:
        return "road_syabru_goljung", "Syabrubesi–Goljung", []
    if "goljung" in clean and "gatlang" in clean:
        return "road_goljung_gatlang", "Goljung–Gatlang", []
    if "gatlang" in clean and "dhunche" in clean:
        return "road_gatlang_dhunche", "Gatlang–Dhunche", []
    if "dhunche" in clean and "kalika" in clean:
        return "road_dhunche_kalika", "Dhunche–Kalikasthan", []
    if "kalika" in clean and "betrawati" in clean:
        return "road_kalika_betrawati", "Kalikasthan–Betrawati", []
    if "mailung" in clean and "betrawati" in clean:
        return "road_mailung_betrawati", "Mailung–Betrawati", []
    if "gatlang" in clean and "kispang" in clean:
        return "road_gatlang_kispang", "Gatlang–Kispang", []
    if "kispang" in clean and "betrawati" in clean:
        return "road_kispang_betrawati", "Kispang–Betrawati", []
    if "betrawati" in clean and "bidur" in clean:
        return "road_betrawati_bidur", "Betrawati–Bidur", []
    if "devighat" in clean and "bidur" in clean:
        return "road_devighat_bidur", "Devighat–Bidur", []

    candidates = []
    for alias, rid in ROAD_ALIASES.items():
        if alias in clean or clean in alias:
            candidates.append((alias, rid))

    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0].replace("-", "–").title(), []
    elif len(candidates) > 1:
        unique_rids = set(c[1] for c in candidates)
        if len(unique_rids) == 1:
            rid = list(unique_rids)[0]
            return rid, candidates[0][0].replace("-", "–").title(), []
        return None, None, [c[0].replace("-", "–").title() for c in candidates]

    fuzzy = difflib.get_close_matches(clean, list(ROAD_ALIASES.keys()), n=3, cutoff=0.55)
    if len(fuzzy) == 1:
        matched = fuzzy[0]
        return ROAD_ALIASES[matched], matched.replace("-", "–").title(), []

    return None, None, [f.replace("-", "–").title() for f in fuzzy] if fuzzy else list(ROAD_ALIASES.keys())[:5]


_LOCATION_CATALOG_CACHE: Optional[Dict[str, str]] = None
_LOCATION_CATALOG_TIME: float = 0.0


def get_dynamic_location_catalog() -> Dict[str, str]:
    """Retrieves all locations from FalkorDB with 30-second caching."""
    global _LOCATION_CATALOG_CACHE, _LOCATION_CATALOG_TIME
    now = time.time()
    if _LOCATION_CATALOG_CACHE is not None and (now - _LOCATION_CATALOG_TIME) < 30.0:
        return _LOCATION_CATALOG_CACHE

    catalog = dict(LOCATION_ALIASES)
    try:
        graph = get_graph()
        query_result = graph.query("MATCH (l:Location) RETURN l.id, l.name")
        if query_result and query_result.result_set:
            for row in query_result.result_set:
                lid = str(row[0])
                lname = str(row[1])
                catalog[lname.lower()] = lid
                for part in re.split(r"[\s/–-]+", lname.lower()):
                    if len(part) >= 3 and part not in catalog:
                        catalog[part] = lid
    except Exception as exc:
        logger.warning("Could not refresh dynamic location catalog: %s", exc)

    _LOCATION_CATALOG_CACHE = catalog
    _LOCATION_CATALOG_TIME = now
    return catalog


def extract_corridor_endpoints(text: str) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    """Extracts origin and destination locations from natural language queries."""
    if not text or not isinstance(text, str):
        return None, None

    clean = text.lower()
    catalog = get_dynamic_location_catalog()

    matches = []
    for alias, lid in catalog.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, clean):
            matches.append((m.start(), m.end(), alias, lid))

    if not matches:
        return None, None

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, alias, lid in matches:
        if start >= last_end:
            filtered.append((start, end, alias, lid))
            last_end = end

    unique_locs = []
    seen_lids = set()
    for start, end, alias, lid in filtered:
        if lid not in seen_lids:
            seen_lids.add(lid)
            cname = SETTLEMENT_CANONICAL_NAMES.get(lid, alias.capitalize())
            unique_locs.append((start, lid, cname))

    if len(unique_locs) == 0:
        return None, None
    if len(unique_locs) == 1:
        return (unique_locs[0][1], unique_locs[0][2]), None

    loc_a = (unique_locs[0][1], unique_locs[0][2])
    loc_b = (unique_locs[1][1], unique_locs[1][2])

    # Directionality: reach B from A / to B from A
    reach_from = re.search(r"\b(reach|to|access|go to)\b.*?\bfrom\b", clean)
    if reach_from:
        from_idx = clean.find("from")
        if unique_locs[0][0] < from_idx and unique_locs[1][0] > from_idx:
            return loc_b, loc_a
        elif unique_locs[1][0] < from_idx and unique_locs[0][0] > from_idx:
            return loc_a, loc_b

    # Directionality: from A to B
    from_to = re.search(r"\bfrom\b.*?\bto\b", clean)
    if from_to:
        from_idx = clean.find("from")
        to_idx = clean.find("to", from_idx)
        if from_idx < unique_locs[0][0] < to_idx and to_idx < unique_locs[1][0]:
            return loc_a, loc_b
        elif from_idx < unique_locs[1][0] < to_idx and to_idx < unique_locs[0][0]:
            return loc_b, loc_a

    return loc_a, loc_b


def find_route_between(origin_input: str, destination_input: str) -> Dict[str, Any]:
    """Evaluates multi-hop paths between two locations, classifying routes by passability tier."""
    orig_id, orig_name, orig_cands = resolve_settlement_id(origin_input)
    dest_id, dest_name, dest_cands = resolve_settlement_id(destination_input)

    if not orig_id:
        return {
            "status": "unresolved",
            "message": f"Origin location '{origin_input}' could not be identified in FalkorDB.",
            "candidates": orig_cands,
        }
    if not dest_id:
        return {
            "status": "unresolved",
            "message": f"Destination location '{destination_input}' could not be identified in FalkorDB.",
            "candidates": dest_cands,
        }

    cypher = (
        "MATCH p = (start:Location {id: $origin_id})-[:ROAD_TO*1..8]->(dest:Location {id: $dest_id}) "
        "WHERE ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "WITH p, "
        "     reduce(total = 0.0, r IN relationships(p) | total + r.length_km) AS total_km, "
        "     CASE "
        "       WHEN ALL(r IN relationships(p) WHERE r.status = 'verified_passable') THEN 1 "
        "       WHEN NONE(r IN relationships(p) WHERE r.status = 'blocked') THEN 2 "
        "       ELSE 3 "
        "     END AS route_tier "
        "RETURN [n IN nodes(p) | {id: n.id, name: n.name}] AS locations, "
        "       [r IN relationships(p) | {road_id: r.road_id, name: r.road_name, status: r.status, length_km: r.length_km, is_bridge: r.is_bridge}] AS segments, "
        "       total_km, "
        "       route_tier "
        "ORDER BY route_tier ASC, total_km ASC "
        "LIMIT 6"
    )

    graph = get_graph()
    route_result = graph.query(cypher, {"origin_id": orig_id, "dest_id": dest_id})
    routes = []
    all_unverified_rids = set()

    if route_result and route_result.result_set:
        for row in route_result.result_set:
            locs = row[0]
            segs = row[1]
            total_km = round(float(row[2]), 2)
            tier = int(row[3])

            tier_label = (
                "Verified Passable" if tier == 1
                else "Needs Reconnaissance" if tier == 2
                else "Severed / Blocked"
            )

            bottlenecks = [s for s in segs if s["status"] in ["unknown", "disputed", "blocked"]]
            for b in bottlenecks:
                if b["status"] in ["unknown", "disputed"]:
                    all_unverified_rids.add(b["road_id"])

            routes.append({
                "tier": tier,
                "tier_label": tier_label,
                "locations": locs,
                "location_names": [loc["name"] for loc in locs],
                "segments": segs,
                "road_names": [s["name"] for s in segs],
                "road_ids": [s["road_id"] for s in segs],
                "total_km": total_km,
                "bottlenecks": bottlenecks,
                "linked_missions": [],
            })

    # Link unverified segments with active missions
    missions_by_rid = {}
    if all_unverified_rids:
        try:
            mission_result = graph.query(
                "MATCH (m:Mission) WHERE m.road_id IN $rids RETURN m.road_id, m.id, m.priority_score, m.reason",
                {"rids": list(all_unverified_rids)},
            )
            if mission_result and mission_result.result_set:
                for m_row in mission_result.result_set:
                    missions_by_rid[str(m_row[0])] = {
                        "road_id": str(m_row[0]),
                        "mission_id": str(m_row[1]),
                        "priority_score": float(m_row[2]),
                        "reason": str(m_row[3]),
                    }
        except Exception as exc:
            logger.warning("Could not link missions to unverified routes: %s", exc)

    for r in routes:
        for b in r["bottlenecks"]:
            rid = b["road_id"]
            if rid in missions_by_rid:
                r["linked_missions"].append(missions_by_rid[rid])

    if not routes:
        overall_state = "NO_ROUTE_FOUND"
    elif any(r["tier"] == 1 for r in routes):
        overall_state = "VERIFIED_ACCESS"
    elif any(r["tier"] == 2 for r in routes):
        overall_state = "NEEDS_RECON"
    else:
        overall_state = "SEVERED"

    return {
        "status": "ok",
        "origin_id": orig_id,
        "origin_name": orig_name,
        "destination_id": dest_id,
        "destination_name": dest_name,
        "overall_state": overall_state,
        "routes_count": len(routes),
        "routes": routes,
        "primary_route": routes[0] if routes else None,
        "cypher_executed": cypher.replace("$origin_id", f"'{orig_id}'").replace("$dest_id", f"'{dest_id}'"),
    }
