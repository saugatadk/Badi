import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.graph import get_graph
from backend.missions import get_ranked_missions
from backend.models import AgentQueryResponse
from backend.queries import (
    LOCATION_ALIASES,
    ROAD_ALIASES,
    add_observation_and_sync,
    classify_settlement_access,
    extract_corridor_endpoints,
    find_affected_settlements,
    find_route_between,
    get_bridge_damage_summary,
    get_decision_chain,
    get_episodic_memories,
    get_on_site_facilities,
    get_road_evidence,
    record_episodic_memory,
    resolve_road_id,
    resolve_settlement_id,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "ops_session"
DEFAULT_AGENT_ROLE = "Route Analyst"
COORDINATOR_ROLE = "Field Coordinator"
GUARDRAIL_ROLE = "GuardrailEnforcer"
SCOPE_ROLE = "ScopeEnforcer"

ALLOWED_ROAD_STATUSES = {"blocked", "verified_passable", "unknown", "disputed"}

GUARDRAIL_PATTERNS = [
    r"\bcasualt(y|ies)\b",
    r"\bdeath(s)?\b",
    r"\bfatalit(y|ies)\b",
    r"\bhelicopter(s)?\b",
    r"\bairlift(s)?\b",
    r"\b100%\s*safe\b",
    r"\bcompletely\s*safe\b",
    r"\bguaranteed\s*safe\b",
]

BOUNDARY_REFUSAL_MESSAGE = (
    "Boundary Restriction: Badi only evaluates mapped topological connectivity "
    "and recorded observations inside FalkorDB. It cannot verify physical terrain safety, "
    "casualties, or unmapped hazards."
)

FALKORDB_SCHEMA_PROMPT = """
FalkorDB Graph Schema:
Labels & Properties:
- Location {id: string, name: string, district: string, households: int, type: 'settlement'|'facility_hub'|'junction'}
- Facility {id: string, name: string, location_id: string, type: string, importance_weight: float}
- RoadSegment {id: string, name: string, u: string, v: string, length_km: float, is_bridge: bool, status: 'verified_passable'|'blocked'|'unknown'|'disputed'}
- Bridge {id: string, name: string, status: 'Washed out'|'Damaged'|'Intact', location: string, district: string, lat: float, lon: float}
- Observation {id: string, road_id: string, status: string, note: string, timestamp: string}
- Source {id: string, name: string}
- Mission {id: string, road_id: string, priority_score: float, reason: string, sole_lifeline_count: int, total_dependent_count: int}
- EpisodicMemory {id: string, session_id: string, agent_role: string, event_type: string, summary: string, timestamp: string}

Relationships:
- (u:Location)-[:ROAD_TO {road_id, status, length_km, is_bridge}]->(v:Location)
- (f:Facility)-[:AT_LOCATION]->(l:Location)
- (r:RoadSegment)-[:INCLUDES_BRIDGE]->(b:Bridge)
- (r:RoadSegment)-[:HAS_OBSERVATION]->(o:Observation)
- (o:Observation)-[:FROM_SOURCE]->(s:Source)
- (m:Mission)-[:INVESTIGATES]->(r:RoadSegment)
- (e:EpisodicMemory)-[:AFFECTED_ROAD]->(r:RoadSegment)
"""


def get_falkordb_live_context() -> str:
    """Builds a text snapshot of current graph state for prompt context."""
    try:
        graph = get_graph()

        loc_result = graph.query("MATCH (l:Location) RETURN l.id, l.name, l.district, l.households, l.type ORDER BY l.name")
        loc_rows = loc_result.result_set or []
        total_households = sum(r[3] for r in loc_rows if r[3])
        location_summary = ", ".join([f"{r[1]} ({r[2]}, {r[3]} hh)" for r in loc_rows])

        road_result = graph.query("MATCH (r:RoadSegment) RETURN r.id, r.name, r.status, r.length_km, r.is_bridge ORDER BY r.status, r.name")
        road_rows = road_result.result_set or []
        blocked_roads = [f"{r[1]} (`{r[0]}`)" for r in road_rows if r[2] == "blocked"]
        passable_roads = [f"{r[1]} (`{r[0]}`)" for r in road_rows if r[2] == "verified_passable"]
        unknown_roads = [f"{r[1]} (`{r[0]}` - {r[2]})" for r in road_rows if r[2] in ["unknown", "disputed"]]

        bridge_summary = get_bridge_damage_summary()

        missions = get_ranked_missions()
        mission_lines = []
        for i, m in enumerate(missions[:4], 1):
            mission_lines.append(
                f"- Rank {i}: {m.road_name} (`{m.road_id}`) — Priority: {m.priority_score:.1f}/100. "
                f"Reason: {m.reason}. Sole lifeline for {m.sole_lifeline_count} settlement(s), "
                f"total dependent: {m.total_dependent_count} settlement(s)."
            )

        memory_result = graph.query("MATCH (e:EpisodicMemory) RETURN e.summary ORDER BY e.timestamp DESC LIMIT 3")
        recent_memory_lines = [f"- {r[0]}" for r in (memory_result.result_set or [])]
        memories_formatted = "\n".join(recent_memory_lines) if recent_memory_lines else "- None recorded yet"

        return (
            "LIVE FALKORDB DISASTER GRAPH SNAPSHOT (Trishuli & Bhote Koshi Flood Corridor, Nepal):\n"
            f"1. Surveyed Settlements & Facilities ({len(loc_rows)} locations, ~{total_households:,} surveyed households):\n"
            f"   {location_summary}\n\n"
            f"2. Road Network Passability ({len(road_rows)} total corridors):\n"
            f"   - Confirmed Blocked ({len(blocked_roads)}): {', '.join(blocked_roads) or 'None'}\n"
            f"   - Verified Passable ({len(passable_roads)}): {', '.join(passable_roads) or 'None'}\n"
            f"   - Unknown / Needs Reconnaissance ({len(unknown_roads)}): {', '.join(unknown_roads) or 'None'}\n\n"
            f"3. Bridge Infrastructure Survey ({bridge_summary.get('total', 0)} surveyed bridges):\n"
            f"   - Washed Out: {bridge_summary.get('washed_out', 0)}\n"
            f"   - Damaged: {bridge_summary.get('damaged', 0)}\n"
            f"   - Intact: {bridge_summary.get('intact', 0)}\n\n"
            "4. Priority Reconnaissance Bottlenecks (Active Missions in FalkorDB):\n"
            + "\n".join(mission_lines) + "\n\n"
            "5. Recent Graph Memory & Operational Events:\n"
            + memories_formatted
        )
    except Exception as exc:
        logger.warning("Failed to build live FalkorDB context: %s", exc)
        return f"Live FalkorDB Graph Snapshot unavailable: {exc}"


def query_bridge_damage() -> Dict[str, Any]:
    """Queries bridge damage counts from FalkorDB."""
    return get_bridge_damage_summary()


def trace_settlement_access(settlement_id: str) -> Dict[str, Any]:
    """Evaluates health facility access for a settlement."""
    sid, sname, candidates = resolve_settlement_id(settlement_id)
    if not sid:
        return {
            "status": "unresolved",
            "message": f"Settlement '{settlement_id}' could not be identified in FalkorDB. No access analysis was performed.",
            "candidates": candidates
        }

    on_site_facilities = get_on_site_facilities(sid)
    state, reason, verified_paths, potential_paths = classify_settlement_access(sid)

    active_missions = []
    if state == "NEEDS_RECON" and potential_paths:
        unverified_road_ids = list(set(r for p in potential_paths for r in p.unverified_road_ids))
        if unverified_road_ids:
            try:
                graph = get_graph()
                mission_result = graph.query(
                    "MATCH (m:Mission) WHERE m.road_id IN $rids RETURN m.road_id, m.priority_score, m.reason ORDER BY m.priority_score DESC",
                    {"rids": unverified_road_ids}
                )
                if mission_result and mission_result.result_set:
                    active_missions = [
                        {"road_id": row[0], "priority_score": row[1], "reason": row[2]}
                        for row in mission_result.result_set
                    ]
            except Exception as exc:
                logger.warning("Could not load missions for unverified roads: %s", exc)
                active_missions = []

    return {
        "settlement_id": sid,
        "settlement_name": sname,
        "access_state": state,
        "access_reason": reason,
        "on_site_facilities": on_site_facilities,
        "verified_paths_count": len(verified_paths),
        "potential_paths_count": len(potential_paths),
        "verified_paths": [p.model_dump() for p in verified_paths],
        "potential_paths": [p.model_dump() for p in potential_paths],
        "active_recon_missions": active_missions,
        "cypher_executed": (
            f"MATCH (f:Facility)-[:AT_LOCATION]->(l:Location {{id: '{sid}'}}); "
            f"MATCH p = (s:Location {{id: '{sid}'}})-[:ROAD_TO*1..7]->(hub:Location)<-[:AT_LOCATION]-(f:Facility) "
            "WHERE ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable') ... (and potential paths query)"
        )
    }


def inspect_road_evidence(road_id: str) -> Dict[str, Any]:
    """Retrieves observations, memories, and dependent settlements for a road."""
    resolved_id, resolved_name, candidates = resolve_road_id(road_id)
    if not resolved_id:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB.",
            "candidates": candidates
        }

    road_data = get_road_evidence(resolved_id)
    return {
        "road_id": resolved_id,
        "name": road_data["road"].name,
        "status": road_data["road"].status,
        "length_km": road_data["road"].length_km,
        "is_bridge": road_data["road"].is_bridge,
        "endpoints": [e.model_dump() for e in road_data["endpoints"]],
        "observations": [o.model_dump() for o in road_data["observations"]],
        "memories": [m.model_dump() for m in road_data["memories"]],
        "affected_settlements": road_data["affected_settlements"],
        "cypher_executed": road_data["cypher_executed"]
    }


def analyze_road_impact(road_id: str) -> Dict[str, Any]:
    """Calculates settlements and households affected if a road is blocked."""
    resolved_id, resolved_name, candidates = resolve_road_id(road_id)
    if not resolved_id:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB. No impact calculation was made.",
            "candidates": candidates
        }

    affected, cypher = find_affected_settlements(resolved_id)
    total_households = sum(s["households"] for s in affected)
    return {
        "road_id": resolved_id,
        "road_name": resolved_name,
        "affected_settlements": affected,
        "affected_count": len(affected),
        "total_households_affected": total_households,
        "cypher_executed": cypher
    }


def explain_decision_chain(road_id: str) -> Dict[str, Any]:
    """Traces observations, decisions, and missions connected to a road."""
    resolved_id, resolved_name, candidates = resolve_road_id(road_id)
    if not resolved_id:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB.",
            "candidates": candidates
        }
    return get_decision_chain(resolved_id)


def get_active_recon_missions() -> Dict[str, Any]:
    """Returns prioritized reconnaissance missions from FalkorDB."""
    missions = get_ranked_missions()
    return {
        "missions_count": len(missions),
        "missions": [m.model_dump() for m in missions],
        "cypher_executed": "MATCH (m:Mission) RETURN m ORDER BY m.priority_score DESC"
    }


def recall_episodic_memory(limit: int) -> Dict[str, Any]:
    """Retrieves recent episodic memory events."""
    memories = get_episodic_memories(limit=limit, road_id=None)
    return {
        "memories_count": len(memories),
        "memories": [m.model_dump() for m in memories],
        "cypher_executed": "MATCH (e:EpisodicMemory) RETURN e ORDER BY e.timestamp DESC LIMIT $limit"
    }


def record_road_observation(
    road_name_or_id: str,
    reported_status: str,
    note: str,
    source: str = "operator_chat",
    scout_verified: bool = False,
    session_id: str = DEFAULT_SESSION_ID,
    agent_role: str = COORDINATOR_ROLE
) -> Dict[str, Any]:
    """Records a field observation in FalkorDB with passability checks."""
    resolved_id, resolved_name, candidates = resolve_road_id(road_name_or_id)
    if not resolved_id:
        return {
            "status": "unresolved",
            "message": (
                f"I could not identify road '{road_name_or_id}'. "
                "No database changes were made to FalkorDB."
            ),
            "candidates": candidates
        }

    normalized_status = reported_status.lower().strip()
    if normalized_status not in ALLOWED_ROAD_STATUSES:
        normalized_status = "unknown"

    source_id = "src_operator"
    if "scout" in source.lower():
        source_id = "src_scout"
    elif "community" in source.lower() or "user" in source.lower():
        source_id = "src_user"

    override = False
    warning_note = None

    # Unverified field reports cannot promote a road to verified_passable without scout verification.
    if normalized_status == "verified_passable":
        if source_id == "src_scout" or scout_verified:
            override = True
        else:
            normalized_status = "unknown"
            warning_note = (
                "Epistemic Safety Constraint: Unverified field report cannot mark a road as 'verified_passable' "
                "without on-ground scout verification. Recorded report as 'unknown' pending reconnaissance."
            )
            note = f"{note} [Unverified passability report held pending scout survey]"

    sync_result = add_observation_and_sync(
        road_id=resolved_id,
        reported_status=normalized_status,
        note=note,
        source_id=source_id,
        override=override,
        session_id=session_id,
        agent_role=agent_role
    )

    return {
        "status": "ok",
        "road_id": resolved_id,
        "road_name": resolved_name,
        "reported_status": reported_status,
        "recorded_status": normalized_status,
        "resolved_network_status": sync_result["new_status"],
        "resolution_reason": sync_result["resolution_reason"],
        "warning": warning_note or "",
        "observation_id": sync_result["observation_id"],
        "missions_count": sync_result["missions_count"],
        "memory_id": sync_result["memory_id"],
        "cypher_executed": (
            f"MATCH (r:RoadSegment {{id: '{resolved_id}'}}) "
            f"CREATE (o:Observation {{road_id: '{resolved_id}', status: '{normalized_status}'}}) "
            f"CREATE (r)-[:HAS_OBSERVATION]->(o) SET r.status = '{sync_result['new_status']}'"
        )
    }


def _check_guardrails(prompt: str) -> Optional[str]:
    """Checks prompt against boundary restrictions."""
    prompt_lower = prompt.lower()
    for pattern in GUARDRAIL_PATTERNS:
        if re.search(pattern, prompt_lower):
            # Allow helicopter or airlift mentions if they refer to observation reports rather than dispatch/rescue requests.
            if "helicopter" in pattern or "airlift" in pattern:
                is_dispatch = bool(re.search(r"\b(dispatch|deploy|send|order|request|call|evacuat|rescue|fly|land|service|airlift)\b", prompt_lower))
                is_observation = bool(re.search(r"\b(view|spotted|saw|found|report|observed|observation|blocked|passable|update|record)\b", prompt_lower))
                if is_observation and not is_dispatch:
                    continue
            return BOUNDARY_REFUSAL_MESSAGE
    return None


def _deterministic_agent_router(prompt: str, session_id: str) -> AgentQueryResponse:
    """Dispatches operational queries directly using FalkorDB."""
    prompt_lower = prompt.lower()
    graph = get_graph()

    # Greetings and capabilities overview
    if re.search(r"\b(hello|hi|namaste|hey|help)\b|who are you|what can you do", prompt_lower):
        role = COORDINATOR_ROLE
        loc_result = graph.query("MATCH (l:Location) RETURN count(l)")
        location_count = loc_result.result_set[0][0] if loc_result and loc_result.result_set else 0

        road_result = graph.query("MATCH (r:RoadSegment) RETURN count(r)")
        road_count = road_result.result_set[0][0] if road_result and road_result.result_set else 0

        bridge_result = graph.query("MATCH (b:Bridge) RETURN count(b)")
        bridge_count = bridge_result.result_set[0][0] if bridge_result and bridge_result.result_set else 0

        answer = (
            f"Namaste! I am **Badi (बाढी)**, your AI agent grounded directly in **FalkorDB** for flood reconnaissance "
            f"and disaster triage across Nepal's Bhote Koshi–Trishuli corridor.\n\n"
            f"Currently tracking in live FalkorDB graph (`badi_trishuli_graph`):\n"
            f"• **{location_count} Settlements & Hubs** (Rasuwa & Nuwakot Districts)\n"
            f"• **{road_count} Critical Road Corridors** (with real curved lifelines & passability states)\n"
            f"• **{bridge_count} Surveyed Flood Bridges** (HOT Flood ground-truth data)\n"
            f"• **Active Prioritized Recon Missions** (Cut-edge bottleneck detection)\n\n"
            f"You can ask me:\n"
            f"1. *'How many bridges have been washed out?'*\n"
            f"2. *'Trace hospital access for Syabrubesi'*\n"
            f"3. *'What is the blast radius if Syabrubesi–Dhunche is blocked?'*\n"
            f"4. *'Which settlements are in Rasuwa district?'*\n"
            f"5. *'Connect people, projects, and decisions for Syabrubesi–Dhunche'*\n"
            f"6. *'Recall recent episodic memories'*"
        )
        cypher = "MATCH (l:Location) RETURN count(l); MATCH (r:RoadSegment) RETURN count(r); MATCH (b:Bridge) RETURN count(b)"
        mem_id = record_episodic_memory(session_id, role, "greeting", "Agent greeted operator and presented graph capabilities.")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="system_overview",
            cypher_executed=cypher,
            supporting_evidence={"locations": location_count, "roads": road_count, "bridges": bridge_count},
            memory_id=mem_id
        )

    # Road observation / passability status update
    is_obs_keyword = bool(re.search(r"\b(update|record|report(ed)?|observed?|found|spotted|saw)\b", prompt_lower))
    is_status_keyword = bool(re.search(r"\b(passable|blocked|damaged|open|landslide|hazard|washed)\b", prompt_lower))
    is_explicit_obs = bool(re.search(r"\b(record\s+observation|update\s+road|update\s+corridor|mark\s+road|mark\s+as|set\s+status|change\s+status|reflect\s+the\s+change)\b", prompt_lower))

    if (is_obs_keyword and is_status_keyword) or is_explicit_obs:
        role = COORDINATOR_ROLE
        resolved_id, resolved_name, candidates = resolve_road_id(prompt)
        if not resolved_id:
            endpoint_origin, endpoint_dest = extract_corridor_endpoints(prompt)
            if endpoint_origin and endpoint_dest:
                resolved_id, resolved_name, candidates = resolve_road_id(f"{endpoint_origin[1]} {endpoint_dest[1]}")

        if not resolved_id:
            candidates_str = ", ".join(candidates[:4]) if candidates else "None"
            answer = (
                f"I could not identify which road corridor you wish to update from your query: *'{prompt}'*.\n\n"
                f"Please specify a surveyed corridor name or road ID. Valid options include: {candidates_str}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="record_road_observation",
                cypher_executed="None: road unresolved",
                supporting_evidence={"status": "unresolved", "candidates": candidates},
                memory_id=None
            )

        # Resolve status from explicit directives like 'update status to blocked'
        status_directive_match = re.search(
            r"(?:update|set|change|mark|switch)(?:\s+(?:the|this|corridor|road))?(?:\s+status)?\s+(?:to|as)\s+([a-z_]+)",
            prompt_lower
        ) or re.search(
            r"status\s+(?:to|is|=)\s+([a-z_]+)",
            prompt_lower
        )
        target_token = status_directive_match.group(1) if status_directive_match else ""

        if target_token in ["blocked", "closed", "impassable", "severed", "landslide", "washed"]:
            reported_st = "blocked"
        elif target_token in ["passable", "verified_passable", "open", "clear"]:
            reported_st = "verified_passable"
        elif target_token in ["unknown", "unverified"]:
            reported_st = "unknown"
        elif target_token in ["disputed"]:
            reported_st = "disputed"
        else:
            # If both blocked and passable keywords exist, use the one appearing later in the text.
            has_blocked = any(k in prompt_lower for k in ["blocked", "landslide", "closed", "washed", "severed", "impassable"])
            has_passable = any(k in prompt_lower for k in ["passable", "open", "clear"])
            if has_blocked and not has_passable:
                reported_st = "blocked"
            elif has_passable and not has_blocked:
                reported_st = "verified_passable"
            elif has_blocked and has_passable:
                pos_blocked = max(prompt_lower.rfind(k) for k in ["blocked", "landslide", "closed", "washed", "severed", "impassable"] if k in prompt_lower)
                pos_passable = max(prompt_lower.rfind(k) for k in ["passable", "open", "clear"] if k in prompt_lower)
                reported_st = "blocked" if pos_blocked > pos_passable else "verified_passable"
            else:
                reported_st = "unknown"

        scout_verified = bool(re.search(r"\b(scout|physically\s*survey|ground\s*survey|field\s*inspection)\b", prompt_lower))
        source = (
            "aerial_recon_helicopter" if ("helicopter" in prompt_lower or "aerial" in prompt_lower)
            else "field_scout" if scout_verified
            else "operator_chat"
        )

        observation_result = record_road_observation(
            road_name_or_id=resolved_id,
            reported_status=reported_st,
            note=prompt,
            source=source,
            scout_verified=scout_verified,
            session_id=session_id,
            agent_role=COORDINATOR_ROLE
        )

        if observation_result.get("status") == "ok":
            recorded_display_status = observation_result["recorded_status"].replace("_", " ").title()
            network_display_status = observation_result["resolved_network_status"].replace("_", " ").title()

            lines = [
                "**Field Observation Recorded in FalkorDB**\n",
                f"• **Corridor:** {observation_result['road_name']} (`{observation_result['road_id']}`)",
                f"• **Source:** {source.replace('_', ' ').title()}",
                f"• **Reported Condition:** {reported_st.replace('_', ' ').title()}",
                f"• **Recorded In Graph:** {recorded_display_status}",
                f"• **Active Network Passability:** {network_display_status}",
                f"• **Resolution:** {observation_result.get('resolution_reason', 'Observation committed.')}"
            ]
            if observation_result.get("warning"):
                lines.append(f"\n> **Epistemic Safety Rule:** {observation_result['warning']}")

            lines.append(f"\n• **Episodic Memory Node:** `{observation_result.get('memory_id')}`")
            lines.append(f"• **Triage Queue Recalibration:** {observation_result.get('missions_count', 0)} active reconnaissance bottleneck missions updated.")

            answer = "\n".join(lines)
            cypher = (
                f"MATCH (r:RoadSegment {{id: '{resolved_id}'}}) "
                f"CREATE (o:Observation {{id: '{observation_result.get('observation_id')}', status: '{observation_result['recorded_status']}', source: '{source}'}})-[:HAS_OBSERVATION]->(r);\n"
                f"MATCH (u)-[rel:ROAD_TO {{road_id: '{resolved_id}'}}]->(v) SET rel.status = '{observation_result['resolved_network_status']}';\n"
                "MATCH (m:Mission) ... Dynamic cut-edge triage recalculation"
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="record_road_observation",
                cypher_executed=cypher,
                supporting_evidence=observation_result,
                memory_id=observation_result.get("memory_id")
            )

    # Point-to-point corridor pathfinding
    if not any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "what happens if", "blocked if", "depend on", "update", "record", "report"]):
        endpoint_origin, endpoint_dest = extract_corridor_endpoints(prompt)
        if endpoint_origin and endpoint_dest and endpoint_origin[0] != endpoint_dest[0]:
            role = DEFAULT_AGENT_ROLE
            orig_id, orig_name = endpoint_origin
            dest_id, dest_name = endpoint_dest
            route_evidence = find_route_between(orig_id, dest_id)

            state = route_evidence.get("overall_state", "NO_ROUTE_FOUND")
            routes = route_evidence.get("routes", [])

            header = f"**Transit Corridor: {orig_name} to {dest_name}**\n\n"
            if state == "NO_ROUTE_FOUND":
                answer = header + f"• **Status:** No topological path found connecting {orig_name} to {dest_name} in the surveyed network."
            else:
                state_label = (
                    "Verified Passable" if state == "VERIFIED_ACCESS"
                    else "Needs Reconnaissance" if state == "NEEDS_RECON"
                    else "Severed / Blocked"
                )
                bullets = [f"• **Corridor State:** {state_label}"]
                for idx, r in enumerate(routes[:3], start=1):
                    path_str = " -> ".join(r["location_names"])
                    tier_name = r["tier_label"]
                    bullets.append(f"\n• **Option {idx} ({tier_name} — {r['total_km']} km):**")
                    bullets.append(f"  - Route: {path_str}")

                    seg_conditions = []
                    for seg in r["segments"]:
                        s_name = seg["name"]
                        s_st = seg["status"].replace("_", " ").capitalize()
                        seg_conditions.append(f"{s_name} ({s_st})")
                    bullets.append(f"  - Road Corridors: {'; '.join(seg_conditions)}")

                    if r.get("linked_missions"):
                        m_lines = []
                        for m in r["linked_missions"][:2]:
                            m_lines.append(f"Active Recon Mission (Priority: {m['priority_score']}/100 — {m['reason'][:80]}...)")
                        bullets.append(f"  - Reconnaissance: {'; '.join(m_lines)}")
                answer = header + "\n".join(bullets)

            primary_rid = "corridor"
            if route_evidence.get("primary_route") and route_evidence["primary_route"].get("road_ids"):
                primary_rid = route_evidence["primary_route"]["road_ids"][0]

            mem_id = record_episodic_memory(
                session_id, role, "corridor_route_query",
                f"Route Analyst evaluated transit corridor between {orig_name} and {dest_name}: {state}.",
                primary_rid
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="find_route_between",
                cypher_executed=route_evidence.get("cypher_executed", "MATCH p = (start:Location)-[:ROAD_TO*1..8]->(dest:Location)..."),
                supporting_evidence=route_evidence,
                memory_id=mem_id
            )

    # Bridge damage query
    if any(k in prompt_lower for k in ["bridge", "bridges", "washed out", "destroyed"]):
        role = DEFAULT_AGENT_ROLE
        bridge_summary = get_bridge_damage_summary()
        if bridge_summary.get("status") == "unavailable" or bridge_summary.get("total", 0) == 0:
            answer = (
                "**Surveyed Bridge Damage Status in FalkorDB:**\n\n"
                "No surveyed bridge records were found in the FalkorDB graph database. "
                "Ensure HOT Flood bridge survey data has been imported into the active graph."
            )
            bridge_counts = {}
        else:
            bridge_counts = bridge_summary.get("counts", {})
            total_bridges = bridge_summary.get("total", 0)
            washed = bridge_counts.get("Washed out", 0)
            damaged = bridge_counts.get("Damaged", 0)
            intact = bridge_counts.get("Intact", 0)

            answer = (
                f"**Surveyed Bridge Damage Status in FalkorDB (HOT Flood Nepal Dataset):**\n\n"
                f"• **Washed Out Bridges:** **{washed}** (completely severed by flood waters)\n"
                f"• **Damaged Bridges:** **{damaged}** (structural compromise / hazardous)\n"
                f"• **Intact Bridges:** **{intact}** (confirmed operational)\n"
                f"• **Total Surveyed Bridges:** **{total_bridges}**\n\n"
                f"All {total_bridges} bridges are geographically pinned along the Trishuli River gorge in the interactive map."
            )

        cypher = bridge_summary.get("cypher_executed", "MATCH (b:Bridge) RETURN b.status, count(b)")
        mem_id = record_episodic_memory(session_id, role, "bridge_query", f"Reported bridge survey metrics from FalkorDB: {bridge_counts}.")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="get_bridge_damage_summary",
            cypher_executed=cypher,
            supporting_evidence=bridge_counts,
            memory_id=mem_id
        )

    # Settlement registry and demographics
    if any(k in prompt_lower for k in ["settlement", "settlements", "village", "households", "population", "district", "rasuwa", "nuwakot"]) and not any(k in prompt_lower for k in ["hospital", "access", "reach", "route", "trace", "evacuat"]):
        role = DEFAULT_AGENT_ROLE
        district_filter = ""
        params = {}
        if "rasuwa" in prompt_lower:
            district_filter = "WHERE s.district CONTAINS 'Rasuwa'"
        elif "nuwakot" in prompt_lower:
            district_filter = "WHERE s.district CONTAINS 'Nuwakot'"

        cypher = f"MATCH (s:Location {{type: 'settlement'}}) {district_filter} RETURN s.name, s.district, s.households ORDER BY s.households DESC"
        settlement_result = graph.query(cypher, params)
        settlements = settlement_result.result_set if settlement_result and settlement_result.result_set else []

        if not settlements:
            answer = "No settlements matching the specified district query were found in FalkorDB."
            total_households = 0
        else:
            total_households = sum(int(r[2]) for r in settlements)
            lines = [f"• **{r[0]}** ({r[1]}): {r[2]} households" for r in settlements]

            clarification = ""
            if "population" in prompt_lower:
                clarification = (
                    "\n\n*Note on Demographics: FalkorDB stores surveyed **household counts** (total: "
                    f"{total_households:,} households), not individual population figures. "
                    "At Nepal's average rural household size of ~4.5 persons, this represents approximately "
                    f"{int(total_households * 4.5):,} individuals across these surveyed communities.*"
                )

            answer = (
                f"**FalkorDB Settlement Registry ({len(settlements)} settlements, {total_households:,} total households):**\n\n" +
                "\n".join(lines) + clarification
            )

        mem_id = record_episodic_memory(session_id, role, "settlement_query", f"Retrieved {len(settlements)} settlements ({total_households} households).")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="query_settlements",
            cypher_executed=cypher,
            supporting_evidence={"settlements_count": len(settlements), "total_households": total_households},
            memory_id=mem_id
        )

    # Memory recall
    if any(k in prompt_lower for k in ["episodic", "memory", "recent event", "history", "recall", "log"]):
        role = COORDINATOR_ROLE
        memory_evidence = recall_episodic_memory(10)
        memories = memory_evidence["memories"]
        if memories:
            summary_lines = [f"• [{m['timestamp'][:19]}] ({m['agent_role']}) {m['summary']}" for m in memories[:5]]
            answer = (
                f"Recalled {len(memories)} recent episodic memory events from FalkorDB:\n" +
                "\n".join(summary_lines)
            )
        else:
            answer = "No episodic memory events found in FalkorDB."

        mem_id = record_episodic_memory(session_id, role, "memory_recall", f"Field Coordinator retrieved {len(memories)} episodic memories.")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="recall_episodic_memory",
            cypher_executed=memory_evidence["cypher_executed"],
            supporting_evidence=memory_evidence,
            memory_id=mem_id
        )

    # Blast radius and road severance impact
    if any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "blocked if", "depend on", "what happens if"]):
        role = DEFAULT_AGENT_ROLE
        resolved_id, resolved_name, candidates = resolve_road_id(prompt_lower)
        if not resolved_id:
            available_roads = ", ".join(candidates[:6])
            answer = (
                "I could not identify which road corridor you are inquiring about. "
                "No road impact calculation was performed for an unverified target.\n\n"
                f"Available mapped corridors include: {available_roads}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_road_impact",
                cypher_executed="None: road resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        impact_evidence = analyze_road_impact(resolved_id)
        affected = impact_evidence["affected_settlements"]
        total_households = impact_evidence["total_households_affected"]
        names = ", ".join([s["name"] for s in affected]) if affected else "None"

        answer = (
            f"**Topological Blast Radius for {resolved_name} (`{resolved_id}`):**\n"
            f"If {resolved_name} is blocked, {len(affected)} settlement(s) ({names}) "
            f"representing {total_households:,} surveyed households lose their primary non-blocked topological transit path to regional healthcare hubs.\n"
            f"FalkorDB dynamic re-ranking will automatically elevate alternate bypass corridors into high-priority Recon Missions."
        )

        mem_id = record_episodic_memory(
            session_id, role, "impact_analysis",
            f"Route Analyst evaluated blast radius of {resolved_id}: {len(affected)} settlements affected.",
            resolved_id
        )
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="analyze_road_impact",
            cypher_executed=impact_evidence["cypher_executed"],
            supporting_evidence=impact_evidence,
            memory_id=mem_id
        )

    # Decision chain connecting observers, decisions, and missions
    if any(k in prompt_lower for k in ["decision chain", "connect people", "people, projects", "who reported", "chain of decision", "scout reports", "decisions connect"]):
        role = COORDINATOR_ROLE
        resolved_id, resolved_name, candidates = resolve_road_id(prompt_lower)
        if not resolved_id:
            available_roads = ", ".join(candidates[:6])
            answer = (
                "I could not identify which road corridor you wish to trace decisions for. "
                f"Please specify a known road corridor (e.g. {available_roads})."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_decision_chain",
                cypher_executed="None: road resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        decision_evidence = explain_decision_chain(resolved_id)
        people_names = ", ".join([s["name"] for s in decision_evidence["people_sources"]]) or "Registered Field Scouts & Authorities"
        obs_count = len(decision_evidence["observations"])
        mem_count = len(decision_evidence["decisions_memories"])
        missions_count = len(decision_evidence["project_missions"])

        answer = (
            f"**People, Projects & Decisions Graph Traversal for {decision_evidence['road_name']} (`{resolved_id}`):**\n\n"
            f"• **Connected People / Observers ({len(decision_evidence['people_sources'])}):** {people_names}\n"
            f"• **Field Observations Recorded ({obs_count}):** Ground-truth reports submitted by scouts and authorities.\n"
            f"• **Episodic Decisions / Memories ({mem_count}):** FalkorDB consensus events, dispute resolutions, and operator notes.\n"
            f"• **Governing Humanitarian Projects ({missions_count}):** Active prioritized Field Missions targeting this corridor.\n"
            f"• **Protected Communities:** {', '.join(decision_evidence['connected_settlements']) or 'Corridor Settlements'}\n\n"
            f"FalkorDB actively unifies field actors, operational triage decisions, and physical topology into an auditable graph chain."
        )

        mem_id = record_episodic_memory(
            session_id, role, "decision_chain_traced",
            f"Field Coordinator traced People-Project-Decision graph chain for {resolved_id}.",
            resolved_id
        )
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="explain_decision_chain",
            cypher_executed=decision_evidence["cypher_executed"],
            supporting_evidence=decision_evidence,
            memory_id=mem_id
        )

    # Road passability condition inquiry
    if any(k in prompt_lower for k in ["status", "condition", "how is", "passable", "blocked", "open", "traffic"]) and not any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "what happens if", "hospital", "access", "reach"]):
        resolved_id, resolved_name, candidates = resolve_road_id(prompt_lower)
        if resolved_id:
            role = DEFAULT_AGENT_ROLE
            road_evidence = inspect_road_evidence(resolved_id)
            st_raw = road_evidence.get("status", "unknown")
            st_clean = st_raw.replace("_", " ").capitalize()
            km = road_evidence.get("length_km", 0.0)
            is_br = road_evidence.get("is_bridge", False)
            ends = " -> ".join([e["name"] for e in road_evidence.get("endpoints", [])])
            aff_count = len(road_evidence.get("affected_settlements", []))

            answer = (
                f"**{resolved_name}: Corridor Status**\n\n"
                f"• **Status:** {st_clean}\n"
                f"• **Endpoints:** {ends} ({km} km)\n"
                f"• **Infrastructure:** {'Bridge structure' if is_br else 'Mountain road corridor'}\n"
                f"• **Dependent Communities:** {aff_count} settlements depend on this link"
            )
            mem_id = record_episodic_memory(session_id, role, "road_status_query", f"Route Analyst checked status of {resolved_id}: {st_raw}.", resolved_id)
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="inspect_road_evidence",
                cypher_executed=road_evidence.get("cypher_executed", f"MATCH (r:RoadSegment {{id: '{resolved_id}'}}) RETURN r"),
                supporting_evidence=road_evidence,
                memory_id=mem_id
            )

    # Health facility access triage
    if any(k in prompt_lower for k in ["hospital", "health post", "clinic", "access", "reach", "route", "trace", "path", "evacuat", "status", "condition", "how is", "open", "isolated"]):
        role = DEFAULT_AGENT_ROLE
        settlement_id_resolved, settlement_name, candidates = resolve_settlement_id(prompt_lower)
        if not settlement_id_resolved:
            available_settlements = ", ".join(candidates[:6])
            answer = (
                "I could not identify which settlement you are inquiring about.\n\n"
                f"Available mapped settlements include: {available_settlements}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_settlement_access",
                cypher_executed="None: settlement resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        access_evidence = trace_settlement_access(settlement_id_resolved)
        state = access_evidence["access_state"]
        on_site = access_evidence.get("on_site_facilities", [])
        verified_paths = access_evidence.get("verified_paths", [])
        potential_paths = access_evidence.get("potential_paths", [])
        active_missions = access_evidence.get("active_recon_missions", [])

        header = f"**{settlement_name}: Health Facility & Access Status**\n\n"
        bullets = []

        if on_site:
            fac_names = ", ".join([f["name"] for f in on_site])
            bullets.append(f"• **On-Site Facility:** {fac_names}")
        else:
            bullets.append("• **On-Site Facility:** None (relies on regional referral)")

        if state == "VERIFIED_ACCESS":
            v_p = verified_paths[0]
            route_str = " -> ".join(v_p["location_names"])
            if on_site and any(f["name"] == v_p["facility_name"] for f in on_site):
                bullets.append(f"• **Regional Referral:** On-site {v_p['facility_name']} operational")
            else:
                bullets.append(f"• **Regional Hospital:** {v_p['facility_name']} ({v_p['total_km']} km via {route_str})")
            bullets.append("• **Corridor Status:** Open and verified passable")

        elif state == "NEEDS_RECON":
            if potential_paths:
                p_p = potential_paths[0]
                route_str = " -> ".join(p_p["location_names"])
                bullets.append(f"• **Nearest Facility:** {p_p['facility_name']} ({p_p['total_km']} km via {route_str})")
                bullets.append("• **Corridor Status:** Needs Reconnaissance — route contains unverified road segments")
                if active_missions:
                    m_names = ", ".join([f"`{m['road_id']}` (Priority: {m['priority_score']}/100)" for m in active_missions[:2]])
                    bullets.append(f"• **Active Recon Mission:** {m_names}")
            else:
                bullets.append("• **Corridor Status:** Needs Reconnaissance — awaiting field scout survey")

        elif state == "ISOLATED_ON_MAP":
            bullets.append("• **Access:** Isolated — all mapped routes cut by confirmed road blockages")

        elif state == "NO_MAPPED_ROUTE":
            bullets.append("• **Access:** No mapped road corridors connect this settlement to regional healthcare hubs")

        else:
            bullets.append(f"• **Access:** {state} — {access_evidence.get('access_reason', '')}")

        answer = header + "\n".join(bullets)

        mem_id = record_episodic_memory(session_id, role, "settlement_triage", f"Route Analyst traced access for {settlement_id_resolved}: classified as {state}.", settlement_id_resolved)
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="trace_settlement_access",
            cypher_executed=access_evidence["cypher_executed"],
            supporting_evidence=access_evidence,
            memory_id=mem_id
        )

    # Mission priority queue
    if any(k in prompt_lower for k in ["mission", "missions", "scout", "recon", "queue", "priority", "bottleneck", "rank"]):
        role = COORDINATOR_ROLE
        ranked_missions = get_ranked_missions()
        if ranked_missions:
            top_mission = ranked_missions[0]
            answer = (
                f"**Current Priority Recon Mission Queue in FalkorDB:**\n\n"
                f"Rank 1: Road `{top_mission.road_id}` ({top_mission.road_name}) - **Priority Score: {top_mission.priority_score}/100**\n"
                f"• **Reason:** {top_mission.reason}\n"
                f"• **Sole Lifeline for:** {top_mission.sole_lifeline_count} settlements | Total Dependent: {top_mission.total_dependent_count} settlements.\n\n"
                f"Total active missions verified in FalkorDB: {len(ranked_missions)}."
            )
            evidence = {"top_mission": top_mission.model_dump(), "total_missions": len(ranked_missions)}
        else:
            answer = "All network corridors currently have verified status. No active recon missions required."
            evidence = {"total_missions": 0}

        mem_id = record_episodic_memory(session_id, role, "mission_triage", f"Field Coordinator retrieved mission queue ({len(ranked_missions)} active missions).")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="get_ranked_missions",
            cypher_executed="MATCH (m:Mission) RETURN m ORDER BY m.priority_score DESC",
            supporting_evidence=evidence,
            memory_id=mem_id
        )

    if GEMINI_API_KEY:
        gemini_response = _execute_gemini_agent_query(prompt, session_id)
        if gemini_response:
            return gemini_response

    # Fallback graph counts overview
    role = DEFAULT_AGENT_ROLE
    cypher = (
        "MATCH (l:Location) WITH count(l) AS locs "
        "MATCH (r:RoadSegment) WITH locs, count(r) AS roads "
        "MATCH (b:Bridge) WITH locs, roads, count(b) AS bridges "
        "MATCH (m:Mission) WITH locs, roads, bridges, count(m) AS missions "
        "RETURN locs, roads, bridges, missions"
    )
    overview_result = graph.query(cypher)
    row = overview_result.result_set[0] if overview_result and overview_result.result_set else [0, 0, 0, 0]

    answer = (
        f"I received your query: *'{prompt}'*.\n\n"
        f"I am actively monitoring the **Trishuli Flood Corridor Graph in FalkorDB**:\n"
        f"• **{row[0]} Locations** (Villages, hubs, and clinics)\n"
        f"• **{row[1]} Road Corridors** (with passability and bridge associations)\n"
        f"• **{row[2]} Surveyed Bridges** (flood-risk infrastructure)\n"
        f"• **{row[3]} Active Recon Missions** prioritized by cut-edge topological centrality\n\n"
        f"To inspect specific disaster data, ask about:\n"
        f"• *Hospital access paths* (e.g. 'Trace hospital access for Syabrubesi')\n"
        f"• *Bridge conditions* (e.g. 'How many bridges are washed out?')\n"
        f"• *Blast radius / cut-edges* (e.g. 'What is the blast radius if Syabrubesi–Dhunche is blocked?')\n"
        f"• *People and decisions* (e.g. 'Connect people, projects, and decisions for Syabrubesi–Dhunche')\n"
        f"• *Episodic memory timeline* (e.g. 'Recall recent episodic memories')"
    )
    mem_id = record_episodic_memory(session_id, role, "general_query", f"Agent provided graph overview for query: '{prompt[:50]}'")
    return AgentQueryResponse(
        answer=answer,
        agent_role=role,
        tool_called="falkordb_overview",
        cypher_executed=cypher,
        supporting_evidence={"locations": row[0], "roads": row[1], "bridges": row[2], "missions": row[3]},
        memory_id=mem_id
    )


FALKORDB_DOMAIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_route_between",
            "description": "Calculates connected road routes between any two settlements or hubs in the Trishuli flood corridor, identifying verified passable paths, unverified segments needing reconnaissance with active missions, and blocked corridors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin settlement name or ID (e.g. Betrawati, Syabrubesi, Dhunche, Bidur)."
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination settlement name or ID (e.g. Syabrubesi, Dhunche, Gatlang, Kispang)."
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_settlement_condition",
            "description": "Evaluates hospital access, lifeline status, and connected road conditions for a settlement (e.g. Syabrubesi, Briddhim, Dhunche, Gatlang, Timure, Goljung, Kispang, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "settlement_name": {
                        "type": "string",
                        "description": "The name or alias of the settlement."
                    }
                },
                "required": ["settlement_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bridge_damage_summary",
            "description": "Returns surveyed bridge damage statistics across the flood corridor from FalkorDB (washed out, damaged, intact).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_road_impact",
            "description": "Calculates the topological blast radius (affected settlements and households cut off from healthcare) if a road corridor is blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "road_name_or_id": {
                        "type": "string",
                        "description": "The road corridor name or ID (e.g. Syabrubesi–Dhunche, Timure-Syabru, Briddhim-Syabru)."
                    }
                },
                "required": ["road_name_or_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_recon_missions",
            "description": "Retrieves active reconnaissance missions prioritized by cut-edge topological bottleneck score.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_decision_chain",
            "description": "Traces people (field scouts), observations, episodic decisions, and missions connected to a road corridor in FalkorDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "road_name_or_id": {
                        "type": "string",
                        "description": "The road corridor name or ID."
                    }
                },
                "required": ["road_name_or_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_episodic_memory",
            "description": "Retrieves recent episodic memory events logged in FalkorDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent episodic memories to retrieve (e.g. 5 or 10)."
                    }
                },
                "required": ["limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_road_observation",
            "description": "Records an aerial or ground field observation in FalkorDB. Passability promotion to verified_passable requires physical scout verification. Updates road passability status, graph edges, and dynamically recomputes recon missions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "road_name_or_id": {
                        "type": "string",
                        "description": "Name or ID of the road corridor (e.g. Briddhim to Syabrubesi, Syabrubesi–Dhunche)."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "verified_passable", "unknown", "disputed"],
                        "description": "The observed passability status."
                    },
                    "note": {
                        "type": "string",
                        "description": "Observation note or description."
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of observation (e.g. operator_chat, scout_field, community_report)."
                    }
                },
                "required": ["road_name_or_id", "status"]
            }
        }
    }
]


def _convert_schema(schema_dict: dict) -> types.Schema:
    """Converts a JSON-schema dictionary into a google-genai Schema."""
    if not isinstance(schema_dict, dict):
        return schema_dict
    s_type = schema_dict.get("type", "OBJECT").upper()
    kwargs: Dict[str, Any] = {"type": s_type}
    if "description" in schema_dict:
        kwargs["description"] = schema_dict["description"]
    if "properties" in schema_dict and schema_dict["properties"]:
        kwargs["properties"] = {k: _convert_schema(v) for k, v in schema_dict["properties"].items()}
    elif s_type == "OBJECT":
        kwargs["properties"] = {}
    if "required" in schema_dict and schema_dict["required"]:
        kwargs["required"] = schema_dict["required"]
    if "enum" in schema_dict and schema_dict["enum"]:
        kwargs["enum"] = schema_dict["enum"]
    return types.Schema(**kwargs)


def _get_gemini_tools() -> List[types.Tool]:
    """Builds google-genai Tool definitions for the domain tools."""
    declarations = []
    for tool_def in FALKORDB_DOMAIN_TOOLS:
        fn = tool_def["function"]
        params = _convert_schema(fn.get("parameters", {}))
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description"),
            parameters=params
        ))
    return [types.Tool(function_declarations=declarations)]


def _dispatch_domain_tool(tool_name: str, args: Dict[str, Any], session_id: str) -> Tuple[Dict[str, Any], str, str]:
    """Executes a domain tool against FalkorDB and returns result, cypher, and role."""
    tool_lower = tool_name.lower().strip()

    if tool_lower in ["find_route_between", "route_between"]:
        origin = args.get("origin") or args.get("origin_input") or ""
        destination = args.get("destination") or args.get("destination_input") or ""
        result = find_route_between(origin, destination)
        cypher = result.get("cypher_executed", "MATCH p = (start:Location)-[:ROAD_TO*1..8]->(dest:Location) ...")
        return result, cypher, DEFAULT_AGENT_ROLE

    elif tool_lower in ["get_settlement_condition", "trace_settlement_access"]:
        sname = args.get("settlement_name") or args.get("settlement_id") or args.get("settlement_name_or_id") or ""
        result = trace_settlement_access(sname)
        cypher = result.get("cypher_executed", "MATCH (s:Location)-[:ROAD_TO*1..7]->(hub:Location) ...")
        return result, cypher, DEFAULT_AGENT_ROLE

    elif tool_lower in ["get_bridge_damage_summary", "query_bridge_damage"]:
        result = query_bridge_damage()
        cypher = result.get("cypher_executed", "MATCH (b:Bridge) RETURN b.status AS status, count(b) AS count")
        return result, cypher, COORDINATOR_ROLE

    elif tool_lower in ["analyze_road_impact", "road_impact"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        result = analyze_road_impact(rid)
        cypher = result.get("cypher_executed", "MATCH (r:RoadSegment {id: $road_id}) ...")
        return result, cypher, DEFAULT_AGENT_ROLE

    elif tool_lower in ["get_active_recon_missions", "active_missions"]:
        result = get_active_recon_missions()
        cypher = result.get("cypher_executed", "MATCH (m:Mission) RETURN m ORDER BY m.priority_score DESC")
        return result, cypher, COORDINATOR_ROLE

    elif tool_lower in ["explain_decision_chain", "decision_chain"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        result = explain_decision_chain(rid)
        cypher = result.get("cypher_executed", "MATCH (r:RoadSegment {id: $road_id}) OPTIONAL MATCH ...")
        return result, cypher, COORDINATOR_ROLE

    elif tool_lower in ["recall_episodic_memory", "episodic_memory"]:
        limit = int(args.get("limit", 10))
        result = recall_episodic_memory(limit)
        cypher = result.get("cypher_executed", "MATCH (e:EpisodicMemory) RETURN e ORDER BY e.timestamp DESC LIMIT $limit")
        return result, cypher, COORDINATOR_ROLE

    elif tool_lower in ["record_road_observation", "add_observation"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        status = args.get("status") or args.get("reported_status") or "unknown"
        note = args.get("note") or f"Observation recorded via agent on {rid}"
        source = args.get("source") or "field_agent"
        scout_verified = bool(args.get("scout_verified", False))
        result = record_road_observation(
            road_name_or_id=rid,
            reported_status=status,
            note=note,
            source=source,
            scout_verified=scout_verified,
            session_id=session_id,
            agent_role=COORDINATOR_ROLE
        )
        cypher = result.get("cypher_executed", f"MATCH (r:RoadSegment {{id: '{rid}'}}) CREATE (o:Observation) ...")
        return result, cypher, COORDINATOR_ROLE

    else:
        result = {"status": "unknown_tool", "tool_name": tool_name, "args": args}
        return result, "None: Unknown tool requested", DEFAULT_AGENT_ROLE


def _format_fallback_tool_answer(tool_name: str, result: Dict[str, Any]) -> str:
    """Formats tool output when LLM answer synthesis is not possible."""
    tool_lower = tool_name.lower()
    if "route" in tool_lower:
        orig = result.get("origin_name", "Origin")
        dest = result.get("destination_name", "Destination")
        st = result.get("overall_state", "UNKNOWN")
        routes = result.get("routes", [])
        return f"Route analysis from {orig} to {dest}: Overall status is **{st}**. Found {len(routes)} evaluated path(s) in FalkorDB."
    elif "settlement" in tool_lower or "access" in tool_lower:
        name = result.get("settlement_name", "Settlement")
        state = result.get("access_state", "UNKNOWN")
        reason = result.get("access_reason", "")
        return f"Access assessment for **{name}**: Classified as **{state}**. {reason}"
    elif "bridge" in tool_lower:
        counts = result.get("counts", {})
        total = result.get("total", 0)
        return f"Surveyed bridge status summary from FalkorDB: {total} total bridges surveyed. Status breakdown: {counts}."
    elif "impact" in tool_lower:
        rname = result.get("road_name", "Corridor")
        aff = result.get("affected_count", 0)
        hh = result.get("total_households_affected", 0)
        return f"Topological impact for **{rname}**: {aff} settlement(s) and {hh} household(s) lose verified facility access."
    elif "mission" in tool_lower:
        cnt = result.get("missions_count", 0)
        return f"Active reconnaissance missions in FalkorDB: {cnt} prioritized corridor mission(s) active."
    elif "decision" in tool_lower:
        rname = result.get("road_name", "Corridor")
        obs_cnt = len(result.get("observations", []))
        mem_cnt = len(result.get("decisions_memories", []))
        return f"Decision chain for **{rname}**: Grounded in {obs_cnt} field observation(s) and {mem_cnt} episodic decision record(s)."
    elif "memory" in tool_lower:
        cnt = result.get("memories_count", len(result.get("memories", [])))
        return f"Retrieved {cnt} recent episodic memory events from FalkorDB."
    elif "observation" in tool_lower:
        rname = result.get("road_name", "Corridor")
        nst = result.get("resolved_network_status", "updated")
        return f"Field observation recorded for **{rname}**. Resolved network status in FalkorDB is now **{nst}**."
    return "Grounded response based on live FalkorDB graph data."


def _is_out_of_scope(prompt: str) -> bool:
    """Checks whether a query is outside the corridor operational scope."""
    prompt_lower = prompt.lower().strip()

    out_of_scope_patterns = [
        r"\bdashain\b",
        r"\btihar\b",
        r"\bchhath\b",
        r"\bholi\b",
        r"\beid\b",
        r"\bchristmas\b",
        r"\bfestival(s)?\b",
        r"\bholiday(s)?\b",
        r"\bcelebrat(e|ion|ing)\b",
        r"\bwho is the (president|prime minister|king|queen|actor|singer)\b",
        r"\b(write|create|generate) (python|javascript|code|script|css|html)\b",
        r"\bhow to (code|program|hack|cook|bake)\b",
        r"\bcapital of\b",
        r"\btell me a joke\b",
        r"\bpoem\b",
        r"\bquantum (physics|computing)\b",
        r"\bmovie(s)?\b",
        r"\bsong(s)?\b",
        r"\bwho won\b",
    ]
    for pattern in out_of_scope_patterns:
        if re.search(pattern, prompt_lower):
            return True

    if _has_direct_operational_intent(prompt_lower):
        return False

    corridor_keywords = [
        "disaster", "flood", "river", "trishuli", "bhotekoshi", "bhote koshi", "rasuwa", "nuwakot",
        "landslide", "road", "bridge", "settlement", "village", "hospital", "clinic", "health",
        "route", "path", "corridor", "access", "isolated", "passable", "blocked", "mission",
        "recon", "scout", "cut-edge", "centrality", "bottleneck", "household", "observation",
        "report", "falkordb", "graph", "badi", "memory", "decision", "sitrep", "briefing",
        "status", "overview", "summary", "hello", "hi", "namaste", "help",
    ]
    if any(k in prompt_lower for k in corridor_keywords):
        return False

    for alias in LOCATION_ALIASES.keys():
        if alias in prompt_lower:
            return False
    for alias in ROAD_ALIASES.keys():
        if alias in prompt_lower:
            return False

    if re.search(r"\b(what is|what's|who is|who's|where is|tell me about|explain)\b", prompt_lower):
        return True

    return False


def _handle_out_of_scope(prompt: str, session_id: str) -> AgentQueryResponse:
    """Generates out-of-scope response without graph traversal."""
    mem_id = record_episodic_memory(
        session_id=session_id,
        agent_role=SCOPE_ROLE,
        event_type="out_of_scope_query",
        summary=f"Query evaluated as out of operational scope: {prompt[:80]}"
    )
    return AgentQueryResponse(
        answer=(
            "This query is out of operational scope and not relevant to the Trishuli flood corridor in FalkorDB. "
            "Badi only evaluates mapped topological connectivity, infrastructure damage, and disaster response data "
            "inside the FalkorDB Trishuli flood corridor graph."
        ),
        agent_role=SCOPE_ROLE,
        tool_called="none_out_of_scope",
        cypher_executed="None: Query is out of operational scope. No Cypher graph traversal executed.",
        supporting_evidence={
            "status": "out_of_scope",
            "reason": "Query is not relevant to mapped FalkorDB disaster graph data.",
            "permitted_scope": [
                "road_passability",
                "bridge_damage_summary",
                "hospital_access_routes",
                "settlement_isolation",
                "reconnaissance_missions"
            ]
        },
        memory_id=mem_id
    )


def _execute_gemini_agent_query(prompt: str, session_id: str) -> Optional[AgentQueryResponse]:
    """Queries Gemini with function calling grounded in FalkorDB."""
    if not GEMINI_API_KEY:
        return None
    try:
        if _is_out_of_scope(prompt):
            return _handle_out_of_scope(prompt, session_id)

        system_instruction = (
            "You are Badi, an expert AI disaster response coordinator and route analyst grounded in the live FalkorDB graph "
            "for the Trishuli & Bhote Koshi flood corridor in Nepal.\n"
            "You act directly on connected graph data. ALWAYS call the provided domain tools to query routes, settlements, "
            "bridge damages, blast radius impact, missions, decision chains, memories, or to record field observations.\n"
            "Never guess or fabricate connectivity, bridge conditions, or household counts.\n"
            "Do not confuse surveyed household units with total population.\n"
            "STRICT DOMAIN CONSTRAINT: Only answer queries regarding mapped corridor connectivity, infrastructure damage, and disaster response. "
            "If the query is out of scope (e.g. general culture, trivia, festivals, coding), prefix your response with [OUT_OF_SCOPE].\n"
            "Do not output emojis. Maintain technical and professional disaster operations tone.\n\n"
            f"{FALKORDB_SCHEMA_PROMPT}"
        )

        client = genai.Client(api_key=GEMINI_API_KEY)
        tools = _get_gemini_tools()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                temperature=0.1
            )
        )

        function_call = None
        candidate_content = None
        if response and response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        fc = getattr(part, "function_call", None)
                        if fc:
                            function_call = fc
                            candidate_content = cand.content
                            break
                    if function_call:
                        break

        if function_call:
            tool_name = function_call.name
            tool_args = dict(function_call.args) if function_call.args else {}

            tool_result, cypher_executed, agent_role = _dispatch_domain_tool(tool_name, tool_args, session_id)

            final_answer = ""
            try:
                user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
                model_tool_call_content = candidate_content or types.Content(role="model", parts=[
                    types.Part(function_call=types.FunctionCall(name=tool_name, args=tool_args))
                ])
                tool_response_content = types.Content(role="tool", parts=[
                    types.Part(function_response=types.FunctionResponse(name=tool_name, response={"result": tool_result}))
                ])

                second_resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[user_content, model_tool_call_content, tool_response_content],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1
                    )
                )
                if second_resp and second_resp.text:
                    final_answer = second_resp.text.strip()
            except Exception as synth_err:
                logger.warning("Gemini synthesis second turn error: %s", synth_err)

            if not final_answer:
                final_answer = _format_fallback_tool_answer(tool_name, tool_result)

            mem_id = record_episodic_memory(
                session_id=session_id,
                agent_role=agent_role,
                event_type="gemini_tool_call",
                summary=f"{agent_role} executed tool '{tool_name}' via Gemini: {prompt[:80]}"
            )

            return AgentQueryResponse(
                answer=final_answer,
                agent_role=agent_role,
                tool_called=tool_name,
                cypher_executed=cypher_executed,
                supporting_evidence=tool_result,
                memory_id=mem_id
            )

        if response and response.text:
            raw_answer = response.text.strip()
            is_out_of_scope = (
                "[OUT_OF_SCOPE]" in raw_answer
                or raw_answer.lower().startswith("out of scope")
                or "outside my operational disaster" in raw_answer.lower()
                or "outside of my operational" in raw_answer.lower()
                or "outside the operational scope" in raw_answer.lower()
                or "not relevant to" in raw_answer.lower()
                or "out of scope" in raw_answer.lower()
            )

            if is_out_of_scope:
                return _handle_out_of_scope(prompt, session_id)

            if _has_direct_operational_intent(prompt.lower()):
                return _deterministic_agent_router(prompt, session_id)

            agent_role = DEFAULT_AGENT_ROLE
            graph = get_graph()
            cypher = (
                "MATCH (l:Location) WITH count(l) AS locs "
                "MATCH (r:RoadSegment) WITH locs, count(r) AS roads "
                "MATCH (b:Bridge) WITH locs, roads, count(b) AS bridges "
                "MATCH (m:Mission) WITH locs, roads, bridges, count(m) AS missions "
                "RETURN locs, roads, bridges, missions"
            )
            overview_result = graph.query(cypher)
            row = overview_result.result_set[0] if overview_result and overview_result.result_set else [0, 0, 0, 0]
            mem_id = record_episodic_memory(
                session_id=session_id,
                agent_role=agent_role,
                event_type="gemini_agent_query",
                summary=f"{agent_role} synthesized corridor overview using Gemini: {prompt[:80]}"
            )
            return AgentQueryResponse(
                answer=raw_answer,
                agent_role=agent_role,
                tool_called="falkordb_overview",
                cypher_executed=cypher,
                supporting_evidence={"locations": row[0], "roads": row[1], "bridges": row[2], "missions": row[3]},
                memory_id=mem_id
            )

    except Exception as exc:
        logger.error("Gemini agent execution error: %s", exc)
        return None
    return None


def _has_direct_operational_intent(prompt_lower: str) -> bool:
    """Checks if prompt matches a direct operational query."""
    # Exclude high-level briefing requests from fast-path routing so they get analytical synthesis.
    briefing_keywords = [
        "briefing", "executive", "sitrep", "situation report", "summary", "summarize",
        "overview", "overall", "disaster situation", "strategic", "strategy", "comprehensive",
        "corridor situation", "general situation", "assessment", "state of the corridor"
    ]
    if any(k in prompt_lower for k in briefing_keywords):
        return False

    for alias in LOCATION_ALIASES.keys():
        if alias in prompt_lower:
            return True

    for alias in ROAD_ALIASES.keys():
        if alias in prompt_lower:
            return True

    patterns = [
        r"\b(hello|hi|namaste|hey|help)\b|who are you|what can you do",
        r"\b(bridge|bridges|washed out|intact)\b",
        r"\b(blast radius|impact|cut off|blocked if|what happens if)\b",
        r"\b(decision chain|connect people|people, projects|chain of decision|decisions connect)\b",
        r"\b(hospital|health post|clinic|reach.*hospital|access.*hospital|evacuat|trace.*access|path to)\b",
        r"\b(episodic|recent event|history|recall|memory log)\b",
        r"\b(mission|missions|recon queue|priority queue|triage queue)\b",
        r"\b(settlement|settlements|village|households|district|rasuwa|nuwakot)\b",
        r"\b(passable|blocked|isolated|open)\b",
        r"\b(record|observation|report|scout|survey|landslide)\b"
    ]
    return any(re.search(pat, prompt_lower) for pat in patterns)


def execute_agent_query(prompt: str, session_id: str = DEFAULT_SESSION_ID) -> AgentQueryResponse:
    """Entrypoint for agent queries. Checks guardrails and scope, then routes to deterministic query handlers or Gemini."""
    refusal = _check_guardrails(prompt)
    if refusal:
        mem_id = record_episodic_memory(
            session_id=session_id,
            agent_role=GUARDRAIL_ROLE,
            event_type="boundary_refusal",
            summary=f"Deterministic refusal triggered for restricted prompt keywords: '{prompt[:60]}...'"
        )
        return AgentQueryResponse(
            answer=refusal,
            agent_role=GUARDRAIL_ROLE,
            tool_called="none_boundary_enforced",
            cypher_executed="EpisodicMemory write only",
            supporting_evidence={"reason": "Epistemic Safety Boundary Violation"},
            memory_id=mem_id
        )

    if _is_out_of_scope(prompt):
        return _handle_out_of_scope(prompt, session_id)

    prompt_lower = prompt.lower()

    if _has_direct_operational_intent(prompt_lower):
        return _deterministic_agent_router(prompt, session_id)

    if GEMINI_API_KEY:
        gemini_res = _execute_gemini_agent_query(prompt, session_id)
        if gemini_res:
            return gemini_res

    return _deterministic_agent_router(prompt, session_id)
