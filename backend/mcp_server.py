"""
Badi Disaster Response Graph Agent - Model Context Protocol (MCP) Server.
Exposes FalkorDB connected graph tools, episodic/procedural memory,
and dynamic topological algorithms to external agent runners via MCP.
"""

import sys
import os
import json
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP
from backend.queries import (
    classify_settlement_access,
    get_road_evidence,
    add_observation_and_sync,
    record_episodic_memory,
    get_episodic_memories,
    get_decision_chain,
    get_bridge_damage_summary,
    get_on_site_facilities,
    find_route_between,
    resolve_settlement_id,
    resolve_road_id
)
from backend.missions import get_ranked_missions
from backend.graph import get_graph

server = FastMCP(
    name="badi-falkordb-mcp",
    instructions="FalkorDB Connected Disaster Graph Tools & Multi-Agent Memory Server for Trishuli Flood Corridor"
)

# Standard Disaster Response Procedural Playbooks (Track 02: Procedural Memory)
PROCEDURAL_PLAYBOOKS = {
    "bridge_washout": {
        "id": "SOP-BRIDGE-WASHOUT",
        "name": "River Crossing Structural Severance Protocol",
        "trigger": "Bridge confirmed washed out or submerged in high-velocity flood",
        "required_role": "Field Coordinator",
        "steps": [
            "1. Sever associated RoadSegment edge in active FalkorDB routing topology.",
            "2. Execute analyze_road_impact to identify upstream and downstream isolated settlements.",
            "3. Compute cut-edge betweenness centrality and promote alternate mountain feeders to Priority Recon Missions.",
            "4. Log EpisodicMemory with affected road, trigger timestamp, and tenant scope."
        ],
        "verification_criteria": "Physical drone or ground scout confirmation required to restore passability."
    },
    "isolated_community": {
        "id": "SOP-ISOLATED-COMMUNITY",
        "name": "Zero-Lifeline Settlement Emergency Triage",
        "trigger": "Settlement classified as ISOLATED_ON_MAP (no non-blocked route to hospital)",
        "required_role": "Route Analyst",
        "steps": [
            "1. Query on-site health posts to evaluate immediate medical stabilization capacity.",
            "2. Identify the closest unverified road segment whose verification restores hospital transit.",
            "3. Issue urgent Reconnaissance Mission with maximum priority weighting (>80.0/100).",
            "4. Alert humanitarian relief coordinators with household count and emergency coordinates."
        ],
        "verification_criteria": "Requires continuous monitoring until verified passable bypass is established."
    },
    "conflicting_reports": {
        "id": "SOP-CONTESTED-REPORT",
        "name": "Epistemic Conflict & Disputed Corridor Protocol",
        "trigger": "Conflicting passability observations received from unverified sources",
        "required_role": "GuardrailEnforcer",
        "steps": [
            "1. Enforce safety rule: mark corridor status as 'disputed' to prevent premature ambulance dispatch.",
            "2. Preserve all conflicting observations in the (r)-[:HAS_OBSERVATION]->(o) graph chain.",
            "3. Retain existing mission priority queue without promoting unverified claims.",
            "4. Await authoritative field scout or operator override with recorded provenance."
        ],
        "verification_criteria": "Only physical ground scout or official NDRRMA override can resolve dispute."
    }
}

@server.tool(name="find_route_between", description="Computes connected road routes between any two settlements or hubs in the Trishuli flood corridor in FalkorDB, returning verified passable paths, unverified segments, and active bottlenecks.")
def tool_find_route_between(origin: str, destination: str) -> str:
    res = find_route_between(origin, destination)
    return json.dumps(res, indent=2)

@server.tool(name="trace_settlement_access", description="Evaluates on-site clinic vs regional hospital access through verified and potential paths for a settlement in FalkorDB.")
def tool_trace_settlement_access(settlement_name_or_id: str) -> str:
    sid, sname, candidates = resolve_settlement_id(settlement_name_or_id)
    if not sid:
        return json.dumps({
            "status": "unresolved",
            "error": f"Settlement '{settlement_name_or_id}' could not be identified in FalkorDB.",
            "available_candidates": candidates
        }, indent=2)

    on_site = get_on_site_facilities(sid)
    state, reason, v_paths, p_paths = classify_settlement_access(sid)
    return json.dumps({
        "settlement_id": sid,
        "settlement_name": sname,
        "access_state": state,
        "access_reason": reason,
        "on_site_facilities": on_site,
        "verified_paths": [p.model_dump() for p in v_paths],
        "potential_paths": [p.model_dump() for p in p_paths]
    }, indent=2)

@server.tool(name="query_bridge_damage", description="Returns live bridge damage status counts (Washed out, Damaged, Intact) from surveyed flood infrastructure in FalkorDB.")
def tool_query_bridge_damage() -> str:
    res = get_bridge_damage_summary()
    return json.dumps(res, indent=2)

@server.tool(name="analyze_road_impact", description="Calculates the topological blast radius (number of settlements and households cut off from healthcare hubs) if a road corridor is blocked.")
def tool_analyze_road_impact(road_name_or_id: str) -> str:
    rid, rname, candidates = resolve_road_id(road_name_or_id)
    if not rid:
        return json.dumps({
            "status": "unresolved",
            "error": f"Road corridor '{road_name_or_id}' could not be identified.",
            "available_candidates": candidates
        }, indent=2)
    res = get_road_evidence(rid)
    return json.dumps({
        "road_id": rid,
        "road_name": rname,
        "status": res["road"].status,
        "affected_settlements": res["affected_settlements"],
        "total_households_affected": sum(s.get("households", 0) for s in res["affected_settlements"])
    }, indent=2)

@server.tool(name="explain_decision_chain", description="Traces people (field scouts), observations, episodic memories, and governing humanitarian missions connected to a road corridor in FalkorDB.")
def tool_explain_decision_chain(road_name_or_id: str) -> str:
    rid, rname, candidates = resolve_road_id(road_name_or_id)
    if not rid:
        return json.dumps({
            "status": "unresolved",
            "error": f"Road corridor '{road_name_or_id}' could not be identified.",
            "available_candidates": candidates
        }, indent=2)
    chain = get_decision_chain(rid)
    return json.dumps(chain, indent=2)

@server.tool(name="get_active_recon_missions", description="Retrieves active field reconnaissance missions ranked by cut-edge topological bottleneck centrality in FalkorDB.")
def tool_get_active_recon_missions() -> str:
    missions = get_ranked_missions()
    return json.dumps([m.model_dump() for m in missions], indent=2)

@server.tool(name="recall_episodic_memory", description="Retrieves persistent episodic memory records from FalkorDB, optionally filtered by road or tenant.")
def tool_recall_episodic_memory(limit: int = 10, road_id: Optional[str] = None) -> str:
    mems = get_episodic_memories(limit=limit, road_id=road_id)
    return json.dumps([m.model_dump() for m in mems], indent=2)

@server.tool(name="record_road_observation", description="Submits a field observation to FalkorDB, resolving status according to epistemic safety rules and triggering mission re-ranking.")
def tool_record_road_observation(road_name_or_id: str, status: str, note: str, source: str = "mcp_scout", override: bool = False) -> str:
    rid, rname, candidates = resolve_road_id(road_name_or_id)
    if not rid:
        return json.dumps({
            "status": "unresolved",
            "error": f"Road corridor '{road_name_or_id}' could not be identified.",
            "available_candidates": candidates
        }, indent=2)

    res = add_observation_and_sync(
        road_id=rid,
        reported_status=status,
        note=note,
        source_id=source,
        override=override,
        session_id="mcp_session",
        agent_role="Field Coordinator"
    )
    return json.dumps({
        "road_id": rid,
        "road_name": rname,
        "recorded_status": status,
        "resolved_network_status": res["new_status"],
        "reason": res["resolution_reason"],
        "active_missions_count": res["missions_count"]
    }, indent=2)

@server.tool(name="get_procedural_playbook", description="Retrieves standard disaster triage SOP playbooks stored in FalkorDB procedural memory (e.g. 'bridge_washout', 'isolated_community', 'conflicting_reports').")
def tool_get_procedural_playbook(trigger_condition: str) -> str:
    key = trigger_condition.lower().strip()
    for k, pb in PROCEDURAL_PLAYBOOKS.items():
        if k in key or key in k or key in pb["name"].lower():
            return json.dumps(pb, indent=2)
    return json.dumps({
        "status": "not_found",
        "message": f"No playbook matched '{trigger_condition}'. Available SOPs: {list(PROCEDURAL_PLAYBOOKS.keys())}"
    }, indent=2)

if __name__ == "__main__":
    import asyncio
    print("Starting Badi MCP Server over stdio...", file=sys.stderr)
    asyncio.run(server.run_stdio_async())
