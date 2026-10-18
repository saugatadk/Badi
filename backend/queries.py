import re
import difflib
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from backend.graph import get_graph
from backend.models import (
    RoadStatus,
    SettlementAccessState,
    PathInfo,
    LocationModel,
    FacilityModel,
    RoadSegmentModel,
    ObservationModel,
    EpisodicMemoryModel,
    KPISummary
)

def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def _extract_path_info(path, facility_id: str, facility_name: str, facility_weight: float) -> PathInfo:
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
        unverified_road_ids=unverified_road_ids
    )

def find_verified_paths(settlement_id: str) -> Tuple[List[PathInfo], str]:
    """
    Finds all paths to any facility where ALL edges are verified_passable.
    Enforces strict epistemic safety: no unobserved or blocked roads allowed.
    """
    cypher = (
        "MATCH p = (s:Location {id: $settlement_id})-[:ROAD_TO*1..7]->(hub:Location) "
        "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
        "WHERE ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable') "
        "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "RETURN p, f.id AS facility_id, f.name AS facility_name, f.importance_weight AS facility_weight "
        "ORDER BY length(p) ASC"
    )
    graph = get_graph()
    res = graph.query(cypher, {"settlement_id": settlement_id})
    paths: List[PathInfo] = []
    if res and res.result_set:
        for row in res.result_set:
            path_obj, fac_id, fac_name, fac_weight = row[0], str(row[1]), str(row[2]), float(row[3])
            paths.append(_extract_path_info(path_obj, fac_id, fac_name, fac_weight))
    return paths, cypher

def find_potential_paths(settlement_id: str) -> Tuple[List[PathInfo], str]:
    """
    Finds all paths where NONE of the edges are blocked, but at least ONE edge
    is unknown or disputed. These represent potential lifelines needing reconnaissance.
    """
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
    res = graph.query(cypher, {"settlement_id": settlement_id})
    paths: List[PathInfo] = []
    if res and res.result_set:
        for row in res.result_set:
            path_obj, fac_id, fac_name, fac_weight = row[0], str(row[1]), str(row[2]), float(row[3])
            paths.append(_extract_path_info(path_obj, fac_id, fac_name, fac_weight))
    return paths, cypher

def classify_settlement_access(settlement_id: str) -> Tuple[SettlementAccessState, str, List[PathInfo], List[PathInfo]]:
    """
    Classifies settlement access using strict Epistemic Safety Rules:
    - VERIFIED_ACCESS: at least 1 path where ALL edges are verified_passable
    - NEEDS_RECON: 0 verified paths, but >= 1 path with unknown/disputed edges and 0 blocked edges
    - ISOLATED_ON_MAP: 0 verified paths and 0 potential paths (all mapped routes cut by blocked edges)
    - NO_MAPPED_ROUTE: No mapped road corridors or facility paths connect to this settlement
    - UNAVAILABLE: Settlement node does not exist in FalkorDB
    """
    graph = get_graph()

    # 1. Verify existence of settlement node in FalkorDB
    node_check = graph.query("MATCH (s:Location {id: $id}) RETURN s", {"id": settlement_id})
    if not node_check or not node_check.result_set:
        return "UNAVAILABLE", f"Settlement node '{settlement_id}' was not found in the graph database.", [], []

    node = node_check.result_set[0][0]
    s_props = node.properties if hasattr(node, "properties") else (getattr(node, "_properties", {}) if hasattr(node, "_properties") else {})
    settlement_name = s_props.get("name", settlement_id) if isinstance(s_props, dict) else settlement_id

    verified_paths, _ = find_verified_paths(settlement_id)
    potential_paths, _ = find_potential_paths(settlement_id)

    if verified_paths:
        reason = f"Verified passable route available to {verified_paths[0].facility_name} ({verified_paths[0].total_km} km)"
        return "VERIFIED_ACCESS", reason, verified_paths, potential_paths
    elif potential_paths:
        reason = f"No verified paths. {len(potential_paths)} potential route(s) require physical reconnaissance"
        return "NEEDS_RECON", reason, verified_paths, potential_paths
    else:
        # Check if settlement has any road edges or paths to a facility
        edge_check = graph.query("MATCH (s:Location {id: $id})-[:ROAD_TO]-() RETURN count(*) AS edges", {"id": settlement_id})
        has_edges = edge_check and edge_check.result_set and edge_check.result_set[0][0] > 0
        if not has_edges:
            return "NO_MAPPED_ROUTE", f"No mapped road connections exist for {settlement_name} ({settlement_id}) in the network graph.", [], []

        any_path = graph.query(
            "MATCH p = (s:Location {id: $id})-[:ROAD_TO*1..7]->(hub:Location)<-[:AT_LOCATION]-(f:Facility) RETURN count(p) LIMIT 1",
            {"id": settlement_id}
        )
        if any_path and any_path.result_set and any_path.result_set[0][0] > 0:
            reason = "All topological routes to healthcare facilities are blocked by flood surges or debris"
            return "ISOLATED_ON_MAP", reason, verified_paths, potential_paths
        else:
            reason = f"No mapped routes connect {settlement_name} to any healthcare facility within the 7-hop horizon."
            return "NO_MAPPED_ROUTE", reason, verified_paths, potential_paths

def find_affected_settlements(road_id: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Finds all settlements whose non-blocked paths to any facility traverse road_id.
    """
    cypher = (
        "MATCH p = (s:Location {type: 'settlement'})-[:ROAD_TO*1..7]->(hub:Location) "
        "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
        "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked') "
        "  AND ANY(rel IN relationships(p) WHERE rel.road_id = $road_id) "
        "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
        "RETURN DISTINCT s.id AS id, s.name AS name, s.households AS households, s.district AS district"
    )
    graph = get_graph()
    res = graph.query(cypher, {"road_id": road_id})
    affected = []
    if res and res.result_set:
        for row in res.result_set:
            affected.append({
                "id": str(row[0]),
                "name": str(row[1]),
                "households": int(row[2]),
                "district": str(row[3])
            })
    else:
        # If road_id is currently marked blocked in FalkorDB, evaluate counterfactually
        # (which settlements lose topological paths through this corridor if blocked)
        cf_cypher = (
            "MATCH p = (s:Location {type: 'settlement'})-[:ROAD_TO*1..7]->(hub:Location) "
            "MATCH (f:Facility)-[:AT_LOCATION]->(hub) "
            "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked' AND rel.road_id <> $road_id) "
            "  AND ANY(rel IN relationships(p) WHERE rel.road_id = $road_id) "
            "  AND ALL(n IN nodes(p) WHERE 1 = size([m IN nodes(p) WHERE m = n])) "
            "RETURN DISTINCT s.id AS id, s.name AS name, s.households AS households, s.district AS district"
        )
        cf_res = graph.query(cf_cypher, {"road_id": road_id})
        if cf_res and cf_res.result_set:
            for row in cf_res.result_set:
                affected.append({
                    "id": str(row[0]),
                    "name": str(row[1]),
                    "households": int(row[2]),
                    "district": str(row[3])
                })
    return affected, cypher

def get_road_evidence(road_id: str) -> Dict[str, Any]:
    """
    Returns full evidence chain: RoadSegment, endpoints, Observations, Sources, and EpisodicMemories.
    """
    graph = get_graph()
    cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "OPTIONAL MATCH (r)-[:CONNECTS]->(loc:Location) "
        "RETURN r, collect(DISTINCT loc) AS endpoints"
    )
    res = graph.query(cypher, {"road_id": road_id})
    if not res or not res.result_set:
        raise ValueError(f"RoadSegment with id '{road_id}' not found in FalkorDB")

    road_node = res.result_set[0][0]
    endpoints_raw = res.result_set[0][1]

    endpoints = [
        LocationModel(
            id=loc.properties["id"],
            name=loc.properties["name"],
            type=loc.properties["type"],
            district=loc.properties["district"],
            lat=float(loc.properties["lat"]),
            lon=float(loc.properties["lon"]),
            households=int(loc.properties.get("households", 0))
        )
        for loc in endpoints_raw if loc
    ]

    # Observations and sources
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
                is_simulated=bool(o_node.properties.get("is_simulated", False))
            ))

    # Episodic memories
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
                timestamp=e_node.properties["timestamp"]
            ))

    # Bridges linked to this road segment
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
                "lon": float(b_node.properties.get("lon", 0.0))
            })

    affected_settlements, aff_cypher = find_affected_settlements(road_id)

    coords = None
    if "coordinates_json" in road_node.properties and road_node.properties["coordinates_json"]:
        try:
            coords = json.loads(road_node.properties["coordinates_json"])
        except Exception:
            pass

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
        bridge_count=len(road_bridges)
    )

    full_cypher = f"{cypher}\n-- Observations:\n{obs_cypher}\n-- Episodic Memory:\n{mem_cypher}\n-- Bridges:\n{bridge_cypher}\n-- Downstream:\n{aff_cypher}"

    return {
        "road": road_model,
        "endpoints": endpoints,
        "observations": observations,
        "memories": memories,
        "affected_settlements": affected_settlements,
        "cypher_executed": full_cypher
    }

def get_decision_chain(road_id: str) -> Dict[str, Any]:
    """
    Connects People, Projects, and Decisions in FalkorDB:
    Traverses (People:Source)-[:REPORTED]->(Observation)-[:LED_TO]->(Decision:EpisodicMemory)
    and (Project:Mission)-[:INVESTIGATES]->(Asset:RoadSegment)-[:CONNECTS]->(Settlement).
    """
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
    res = graph.query(cypher, {"road_id": road_id})
    if not res or not res.result_set:
        raise ValueError(f"RoadSegment with id '{road_id}' not found in FalkorDB")

    row = res.result_set[0]
    return {
        "road_id": row[0],
        "road_name": row[1],
        "road_status": row[2],
        "connected_settlements": [s for s in row[3] if s],
        "people_sources": [s for s in row[4] if s.get("id")],
        "observations": [o for o in row[5] if o.get("id")],
        "decisions_memories": [m for m in row[6] if m.get("id")],
        "project_missions": [m for m in row[7] if m.get("id")],
        "cypher_executed": cypher
    }

def add_observation_and_sync(
    road_id: str,
    reported_status: RoadStatus,
    note: str,
    source_id: str = "src_scout",
    is_simulated: bool = False,
    override: bool = False,
    session_id: str = "ops_session",
    agent_role: str = "Field Coordinator"
) -> Dict[str, Any]:
    """
    Atomically adds an observation, evaluates status conflict resolution,
    synchronizes edge properties in FalkorDB, records episodic memory,
    and returns the resulting road status.
    """
    from backend.missions import recompute_missions

    graph = get_graph()
    obs_id = f"obs_{uuid.uuid4().hex[:8]}"
    ts = _current_timestamp()

    # 0. Check if road segment exists
    road_check = graph.query("MATCH (r:RoadSegment {id: $road_id}) RETURN r.id", {"road_id": road_id})
    if not road_check or not road_check.result_set:
        raise ValueError(f"Road corridor '{road_id}' not found in FalkorDB. Cannot record observation.")

    # 1. Create Observation and connect to RoadSegment and Source
    graph.query(
        "MERGE (s:Source {id: $source_id}) "
        "ON CREATE SET s.name = CASE $source_id "
        "  WHEN 'src_operator' THEN 'Operator Console Submission' "
        "  WHEN 'src_user' THEN 'Community Field Report' "
        "  WHEN 'src_scout' THEN 'Field Recon Scout Report' "
        "  ELSE $source_id END",
        {"source_id": source_id}
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
    res_obs = graph.query(create_obs_cypher, {
        "road_id": road_id,
        "source_id": source_id,
        "obs_id": obs_id,
        "reported_status": reported_status,
        "note": note,
        "timestamp": ts,
        "is_simulated": is_simulated
    })
    if not res_obs or not res_obs.result_set:
        raise ValueError(f"Failed to record observation on road '{road_id}' from source '{source_id}'")

    # 2. Query all observations for road_id to evaluate epistemic consensus
    all_obs_cypher = (
        "MATCH (r:RoadSegment {id: $road_id})-[:HAS_OBSERVATION]->(o:Observation) "
        "RETURN o.status AS status, o.source_id AS source_id"
    )
    all_obs_res = graph.query(all_obs_cypher, {"road_id": road_id})
    statuses = set()
    if all_obs_res and all_obs_res.result_set:
        for r in all_obs_res.result_set:
            statuses.add(r[0])

    # Epistemic resolution logic
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

    # 3. Synchronize RoadSegment node status AND all [:ROAD_TO {road_id: $road_id}] edges atomically
    sync_cypher = (
        "MATCH (r:RoadSegment {id: $road_id}) "
        "SET r.status = $new_status "
        "WITH r "
        "OPTIONAL MATCH ()-[rel:ROAD_TO {road_id: $road_id}]->() "
        "SET rel.status = $new_status "
        "RETURN r.id AS road_id, r.status AS status"
    )
    graph.query(sync_cypher, {"road_id": road_id, "new_status": new_status})

    # 4. Record EpisodicMemory in FalkorDB (Track 02 Proof)
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
        "timestamp": ts
    })

    # 5. Recompute missions dynamically
    missions = recompute_missions()

    return {
        "observation_id": obs_id,
        "road_id": road_id,
        "new_status": new_status,
        "resolution_reason": resolution_reason,
        "memory_id": memory_id,
        "missions_count": len(missions),
        "timestamp": ts
    }

def record_episodic_memory(
    session_id: str,
    agent_role: str,
    event_type: str,
    summary: str,
    road_id: Optional[str] = None
) -> str:
    """
    Stores an episodic memory event in FalkorDB for agent coordination (Track 02).
    """
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
            "timestamp": ts
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
            "timestamp": ts
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
        res = graph.query(cypher, {"road_id": road_id, "limit": limit})
    else:
        cypher = (
            "MATCH (e:EpisodicMemory) "
            "RETURN e "
            "ORDER BY e.timestamp DESC "
            "LIMIT $limit"
        )
        res = graph.query(cypher, {"limit": limit})

    memories = []
    if res and res.result_set:
        for row in res.result_set:
            e = row[0]
            memories.append(EpisodicMemoryModel(
                id=e.properties.get("id", ""),
                session_id=e.properties.get("session_id", ""),
                agent_role=e.properties.get("agent_role", ""),
                event_type=e.properties.get("event_type", ""),
                summary=e.properties.get("summary", ""),
                road_id=e.properties.get("road_id") or None,
                timestamp=e.properties.get("timestamp", "")
            ))
    return memories

def get_network_snapshot() -> Dict[str, Any]:
    """
    Fetches the complete current state of the flood network directly from FalkorDB.
    Every metric, classification, and KPI is computed live.
    """
    graph = get_graph()

    # 1. Fetch Locations
    loc_res = graph.query("MATCH (l:Location) RETURN l ORDER BY l.name ASC")
    locations_dict: Dict[str, Dict[str, Any]] = {}
    if loc_res and loc_res.result_set:
        for row in loc_res.result_set:
            l = row[0].properties
            locations_dict[l["id"]] = {
                "id": l["id"],
                "name": l["name"],
                "type": l["type"],
                "district": l["district"],
                "lat": float(l["lat"]),
                "lon": float(l["lon"]),
                "households": int(l.get("households", 0)),
                "access_state": None,
                "access_reason": None,
                "verified_paths_count": 0,
                "potential_paths_count": 0
            }

    # Classify settlements
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

    # 2. Fetch Facilities
    fac_res = graph.query("MATCH (f:Facility) RETURN f")
    facilities = []
    if fac_res and fac_res.result_set:
        for row in fac_res.result_set:
            f = row[0].properties
            facilities.append(FacilityModel(
                id=f["id"],
                name=f["name"],
                location_id=f["location_id"],
                type=f["type"],
                importance_weight=float(f["importance_weight"])
            ))

    # 3. Fetch RoadSegments
    # Pre-fetch all bridges grouped by road to eliminate N+1 Cypher query bottleneck (Problem H)
    bridges_by_road: Dict[str, List[Dict[str, Any]]] = {}
    b_all_res = graph.query(
        "MATCH (r:RoadSegment)-[:INCLUDES_BRIDGE]->(b:Bridge) "
        "RETURN r.id AS road_id, b.name AS name, b.status AS status, b.lat AS lat, b.lon AS lon"
    )
    if b_all_res and b_all_res.result_set:
        for b_row in b_all_res.result_set:
            brid = str(b_row[0])
            if brid not in bridges_by_road:
                bridges_by_road[brid] = []
            bridges_by_road[brid].append({
                "name": b_row[1],
                "status": b_row[2],
                "lat": float(b_row[3]),
                "lon": float(b_row[4])
            })

    road_res = graph.query("MATCH (r:RoadSegment) RETURN r")
    roads = []
    verified_roads = 0
    unknown_roads = 0
    blocked_roads = 0
    disputed_roads = 0

    if road_res and road_res.result_set:
        for row in road_res.result_set:
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
                except Exception:
                    pass

            if not coords:
                u_loc = locations_dict.get(r["u"])
                v_loc = locations_dict.get(r["v"])
                if u_loc and v_loc:
                    coords = [[u_loc["lon"], u_loc["lat"]], [v_loc["lon"], v_loc["lat"]]]

            # Fetch bridge count and names for this road from pre-fetched map
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
                bridge_count=len(road_bridges)
            ))

    # 4. Count active missions and memories
    mission_res = graph.query("MATCH (m:Mission) RETURN count(m)")
    active_missions = int(mission_res.result_set[0][0]) if mission_res and mission_res.result_set else 0

    mem_res = graph.query("MATCH (e:EpisodicMemory) RETURN count(e)")
    total_memories = int(mem_res.result_set[0][0]) if mem_res and mem_res.result_set else 0

    kpis = KPISummary(
        verified_settlements=verified_settlements,
        needs_recon_settlements=needs_recon_settlements,
        isolated_settlements=isolated_settlements,
        verified_roads=verified_roads,
        unknown_roads=unknown_roads,
        blocked_roads=blocked_roads,
        disputed_roads=disputed_roads,
        active_missions=active_missions,
        episodic_memories=total_memories
    )

    location_models = [LocationModel(**loc) for loc in locations_dict.values()]

    # Load real building and bridge summaries from data/
    data_dir = Path(__file__).resolve().parent.parent / "data"
    bldg_summary_path = data_dir / "building_damage_summary.json"
    bldg_summary = None
    if bldg_summary_path.exists():
        try:
            with open(bldg_summary_path, "r", encoding="utf-8") as f:
                bldg_summary = json.load(f)
        except Exception:
            pass

    bridge_summary_res = graph.query(
        "MATCH (b:Bridge) "
        "RETURN b.status, count(b)"
    )
    bridge_summary = {}
    if bridge_summary_res and bridge_summary_res.result_set:
        for b_row in bridge_summary_res.result_set:
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
        "timestamp": _current_timestamp()
    }

def wipe_falkordb_graph() -> Dict[str, Any]:
    """
    Executes a complete graph wipe on FalkorDB (MATCH (n) DETACH DELETE n),
    resetting the database to 0 nodes and 0 edges.
    """
    graph = get_graph()
    count_res = graph.query("MATCH (n) RETURN count(n)")
    prior_nodes = int(count_res.result_set[0][0]) if count_res and count_res.result_set else 0
    graph.query("MATCH (n) DETACH DELETE n")
    return {
        "status": "wiped",
        "nodes_deleted": prior_nodes,
        "is_empty": True,
        "timestamp": _current_timestamp()
    }

def get_bridge_damage_summary() -> Dict[str, Any]:
    """
    Queries FalkorDB for the flood damage status counts of all surveyed bridges.
    Distinguishes database absence/errors from zero counts, and never uses hardcoded defaults.
    """
    graph = get_graph()
    cypher = "MATCH (b:Bridge) RETURN b.status AS status, count(b) AS count"
    try:
        res = graph.query(cypher)
        if not res or not res.result_set:
            return {
                "status": "unavailable",
                "reason": "No surveyed bridge records found in FalkorDB graph.",
                "counts": {},
                "total": 0,
                "cypher_executed": cypher
            }
        counts = {str(row[0]): int(row[1]) for row in res.result_set}
        return {
            "status": "ok",
            "counts": counts,
            "total": sum(counts.values()),
            "cypher_executed": cypher
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"FalkorDB query error: {str(e)}",
            "counts": {},
            "total": 0,
            "cypher_executed": cypher
        }

def get_on_site_facilities(settlement_id: str) -> List[Dict[str, Any]]:
    """
    Returns healthcare facilities directly situated at this settlement/location.
    """
    if not settlement_id:
        return []
    graph = get_graph()
    cypher = (
        "MATCH (f:Facility)-[:AT_LOCATION]->(l:Location {id: $settlement_id}) "
        "RETURN f.id, f.name, f.type, f.importance_weight"
    )
    res = graph.query(cypher, {"settlement_id": settlement_id})
    facs = []
    if res and res.result_set:
        for row in res.result_set:
            facs.append({
                "id": str(row[0]),
                "name": str(row[1]),
                "type": str(row[2]) if row[2] else "facility",
                "importance_weight": float(row[3]) if row[3] is not None else 1.0
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
    # Facility-specific aliases
    "betrawati hospital": "loc_betrawati",
    "betrawati health post": "loc_betrawati",
    "betrawati clinic": "loc_betrawati",
    "trishuli hospital": "loc_bidur",
    "trisuli hospital": "loc_bidur",
    "bidur hospital": "loc_bidur",
    "rasuwa hospital": "loc_dhunche",
    "rasuwa district hospital": "loc_dhunche",
    "dhunche hospital": "loc_dhunche"
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
    "devighat-bidur": "road_devighat_bidur"
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
    "loc_devighat": "Devighat"
}

def resolve_settlement_id(name_or_alias: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Deterministically resolves a settlement identifier or alias.
    Returns (settlement_id, display_name, candidates).
    NEVER silently defaults to a fallback settlement.
    """
    if not name_or_alias or not isinstance(name_or_alias, str):
        return None, None, list(LOCATION_ALIASES.keys())

    clean = name_or_alias.strip().lower()
    # 1. Exact match on ID
    if clean.startswith("loc_"):
        if clean in SETTLEMENT_CANONICAL_NAMES:
            return clean, SETTLEMENT_CANONICAL_NAMES[clean], []
        for alias, lid in LOCATION_ALIASES.items():
            if lid == clean:
                return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, alias.capitalize()), []
        try:
            g = get_graph()
            check = g.query("MATCH (s:Location {id: $id}) RETURN s.name", {"id": clean})
            if check and check.result_set:
                return clean, check.result_set[0][0], []
        except Exception:
            pass
        return None, None, list(LOCATION_ALIASES.keys())[:5]

    # 2. Exact match on alias
    if clean in LOCATION_ALIASES:
        lid = LOCATION_ALIASES[clean]
        return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, clean.capitalize()), []

    # 3. Substring match
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

    # 4. Fuzzy match close candidates
    import difflib
    fuzzy = difflib.get_close_matches(clean, list(LOCATION_ALIASES.keys()), n=3, cutoff=0.6)
    if len(fuzzy) == 1:
        matched_alias = fuzzy[0]
        lid = LOCATION_ALIASES[matched_alias]
        return lid, SETTLEMENT_CANONICAL_NAMES.get(lid, matched_alias.capitalize()), []

    return None, None, [f.capitalize() for f in fuzzy] if fuzzy else list(LOCATION_ALIASES.keys())[:5]

def resolve_road_id(name_or_alias: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Deterministically resolves a road corridor identifier or alias.
    Returns (road_id, display_name, candidates).
    NEVER silently defaults to a fallback road.
    """
    if not name_or_alias or not isinstance(name_or_alias, str):
        return None, None, list(ROAD_ALIASES.keys())

    clean = name_or_alias.strip().lower()
    # 1. Exact match on road_id
    if clean.startswith("road_"):
        for alias, rid in ROAD_ALIASES.items():
            if rid == clean:
                return rid, alias.replace("-", "–").title(), []
        try:
            g = get_graph()
            check = g.query("MATCH (r:RoadSegment {id: $id}) RETURN r.name", {"id": clean})
            if check and check.result_set:
                return clean, check.result_set[0][0], []
        except Exception:
            pass
        return None, None, list(ROAD_ALIASES.keys())[:5]

    # 2. Exact match on alias
    if clean in ROAD_ALIASES:
        return ROAD_ALIASES[clean], clean.replace("-", "–").title(), []

    # 3. Keyword / endpoint pairs
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

    # 4. Substring match
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

    # 5. Fuzzy match
    import difflib
    fuzzy = difflib.get_close_matches(clean, list(ROAD_ALIASES.keys()), n=3, cutoff=0.55)
    if len(fuzzy) == 1:
        matched = fuzzy[0]
        return ROAD_ALIASES[matched], matched.replace("-", "–").title(), []

    return None, None, [f.replace("-", "–").title() for f in fuzzy] if fuzzy else list(ROAD_ALIASES.keys())[:5]

_LOCATION_CATALOG_CACHE = None
_LOCATION_CATALOG_TIME = 0.0

def get_dynamic_location_catalog() -> Dict[str, str]:
    """
    Retrieves all Locations directly from FalkorDB.
    Returns mapping of lowercase alias/name -> location_id.
    Cached for 30 seconds to support dynamic graph imports without latency.
    """
    global _LOCATION_CATALOG_CACHE, _LOCATION_CATALOG_TIME
    import time
    now = time.time()
    if _LOCATION_CATALOG_CACHE is not None and (now - _LOCATION_CATALOG_TIME) < 30.0:
        return _LOCATION_CATALOG_CACHE

    catalog = dict(LOCATION_ALIASES)
    try:
        g = get_graph()
        res = g.query("MATCH (l:Location) RETURN l.id, l.name")
        if res and res.result_set:
            for row in res.result_set:
                lid = str(row[0])
                lname = str(row[1])
                catalog[lname.lower()] = lid
                for part in re.split(r"[\s/–-]+", lname.lower()):
                    if len(part) >= 3 and part not in catalog:
                        catalog[part] = lid
    except Exception:
        pass

    _LOCATION_CATALOG_CACHE = catalog
    _LOCATION_CATALOG_TIME = now
    return catalog

def extract_corridor_endpoints(text: str) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    """
    Dynamically extracts origin and destination locations from natural language queries.
    Returns ((origin_id, origin_name), (dest_id, dest_name)) or (None, None).
    Handles directionality patterns:
      - 'reach <B> from <A>' / 'to <B> from <A>' -> origin = A, dest = B
      - 'from <A> to <B>' / 'between <A> and <B>' -> origin = A, dest = B
      - default: origin = first_mentioned, dest = second_mentioned
    """
    if not text or not isinstance(text, str):
        return None, None

    clean = text.lower()
    catalog = get_dynamic_location_catalog()

    # Find all location matches with their character spans in the text
    matches = []
    for alias, lid in catalog.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, clean):
            matches.append((m.start(), m.end(), alias, lid))

    if not matches:
        return None, None

    # Filter overlapping matches, preferring longer aliases
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, alias, lid in matches:
        if start >= last_end:
            filtered.append((start, end, alias, lid))
            last_end = end

    # Group by unique location ID in order of appearance
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

    # Default order: first mentioned is origin, second is destination
    return loc_a, loc_b

def find_route_between(origin_input: str, destination_input: str) -> Dict[str, Any]:
    """
    Evaluates topological connectivity and multi-hop paths between two locations.
    Classifies routes by tier (1=Verified Passable, 2=Needs Recon, 3=Blocked)
    and cross-references unverified segments with active (:Mission) nodes.
    """
    orig_id, orig_name, orig_cands = resolve_settlement_id(origin_input)
    dest_id, dest_name, dest_cands = resolve_settlement_id(destination_input)

    if not orig_id:
        return {
            "status": "unresolved",
            "message": f"Origin location '{origin_input}' could not be identified in FalkorDB.",
            "candidates": orig_cands
        }
    if not dest_id:
        return {
            "status": "unresolved",
            "message": f"Destination location '{destination_input}' could not be identified in FalkorDB.",
            "candidates": dest_cands
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
    res = graph.query(cypher, {"origin_id": orig_id, "dest_id": dest_id})
    routes = []
    all_unverified_rids = set()

    if res and res.result_set:
        for row in res.result_set:
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
                "location_names": [l["name"] for l in locs],
                "segments": segs,
                "road_names": [s["name"] for s in segs],
                "road_ids": [s["road_id"] for s in segs],
                "total_km": total_km,
                "bottlenecks": bottlenecks,
                "linked_missions": []
            })

    # Auto-link unverified segments with active (:Mission) nodes in FalkorDB
    missions_by_rid = {}
    if all_unverified_rids:
        try:
            m_res = graph.query(
                "MATCH (m:Mission) WHERE m.road_id IN $rids RETURN m.road_id, m.id, m.priority_score, m.reason",
                {"rids": list(all_unverified_rids)}
            )
            if m_res and m_res.result_set:
                for m_row in m_res.result_set:
                    missions_by_rid[str(m_row[0])] = {
                        "road_id": str(m_row[0]),
                        "mission_id": str(m_row[1]),
                        "priority_score": float(m_row[2]),
                        "reason": str(m_row[3])
                    }
        except Exception:
            pass

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
        "cypher_executed": cypher.replace("$origin_id", f"'{orig_id}'").replace("$dest_id", f"'{dest_id}'")
    }
