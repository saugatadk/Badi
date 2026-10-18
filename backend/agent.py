import re
import json
import difflib
import httpx
from typing import Dict, Any, Optional, List, Tuple
from google import genai
from google.genai import types
from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.models import AgentQueryResponse, RoadStatus, SettlementAccessState
from backend.queries import (
    classify_settlement_access,
    find_verified_paths,
    find_potential_paths,
    find_affected_settlements,
    get_road_evidence,
    add_observation_and_sync,
    record_episodic_memory,
    get_episodic_memories,
    get_decision_chain,
    get_bridge_damage_summary,
    get_on_site_facilities,
    find_route_between,
    extract_corridor_endpoints,
    LOCATION_ALIASES,
    ROAD_ALIASES,
    resolve_settlement_id,
    resolve_road_id
)
from backend.missions import get_ranked_missions
from backend.graph import get_graph

# Guardrail boundary patterns
GUARDRAIL_PATTERNS = [
    r"\bcasualt(y|ies)\b",
    r"\bdeath(s)?\b",
    r"\bfatalit(y|ies)\b",
    r"\bhelicopter(s)?\b",
    r"\bairlift(s)?\b",
    r"\b100%\s*safe\b",
    r"\bcompletely\s*safe\b",
    r"\bguaranteed\s*safe\b"
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
    """
    Constructs a real-time snapshot of the active FalkorDB graph state
    for grounding Gemini 3.1 Flash Lite in hard disaster telemetry.
    """
    try:
        g = get_graph()
        
        # 1. Locations and surveyed households
        loc_res = g.query("MATCH (l:Location) RETURN l.id, l.name, l.district, l.households, l.type ORDER BY l.name")
        loc_rows = loc_res.result_set or []
        total_hh = sum(r[3] for r in loc_rows if r[3])
        loc_summary = ", ".join([f"{r[1]} ({r[2]}, {r[3]} hh)" for r in loc_rows])
        
        # 2. Roads by status
        road_res = g.query("MATCH (r:RoadSegment) RETURN r.id, r.name, r.status, r.length_km, r.is_bridge ORDER BY r.status, r.name")
        road_rows = road_res.result_set or []
        blocked_roads = [f"{r[1]} (`{r[0]}`)" for r in road_rows if r[2] == "blocked"]
        passable_roads = [f"{r[1]} (`{r[0]}`)" for r in road_rows if r[2] == "verified_passable"]
        unknown_roads = [f"{r[1]} (`{r[0]}` - {r[2]})" for r in road_rows if r[2] in ["unknown", "disputed"]]
        
        # 3. Bridges
        bridge_data = get_bridge_damage_summary()
        
        # 4. Recon missions (top cut-edges)
        missions = get_ranked_missions()
        mission_lines = []
        for i, m in enumerate(missions[:4], 1):
            mission_lines.append(
                f"- Rank {i}: {m.road_name} (`{m.road_id}`) — Priority: {m.priority_score:.1f}/100. "
                f"Reason: {m.reason}. Sole lifeline for {m.sole_lifeline_count} settlement(s), "
                f"total dependent: {m.total_dependent_count} settlement(s)."
            )
        
        # 5. Recent Episodic Memories
        mem_res = g.query("MATCH (e:EpisodicMemory) RETURN e.summary ORDER BY e.timestamp DESC LIMIT 3")
        recent_mems = [f"- {r[0]}" for r in (mem_res.result_set or [])]
        mems_str = "\n".join(recent_mems) if recent_mems else "- None recorded yet"

        context_str = (
            "LIVE FALKORDB DISASTER GRAPH SNAPSHOT (Trishuli & Bhote Koshi Flood Corridor, Nepal):\n"
            f"1. Surveyed Settlements & Facilities ({len(loc_rows)} locations, ~{total_hh:,} surveyed households):\n"
            f"   {loc_summary}\n\n"
            f"2. Road Network Passability ({len(road_rows)} total corridors):\n"
            f"   - Confirmed Blocked ({len(blocked_roads)}): {', '.join(blocked_roads) or 'None'}\n"
            f"   - Verified Passable ({len(passable_roads)}): {', '.join(passable_roads) or 'None'}\n"
            f"   - Unknown / Needs Reconnaissance ({len(unknown_roads)}): {', '.join(unknown_roads) or 'None'}\n\n"
            f"3. Bridge Infrastructure Survey ({bridge_data.get('total', 0)} surveyed bridges):\n"
            f"   - Washed Out: {bridge_data.get('washed_out', 0)}\n"
            f"   - Damaged: {bridge_data.get('damaged', 0)}\n"
            f"   - Intact: {bridge_data.get('intact', 0)}\n\n"
            f"4. Priority Reconnaissance Bottlenecks (Active Missions in FalkorDB):\n"
            + "\n".join(mission_lines) + "\n\n"
            f"5. Recent Graph Memory & Operational Events:\n"
            + mems_str
        )
        return context_str
    except Exception as exc:
        return f"Live FalkorDB Graph Snapshot unavailable: {exc}"


# Tool functions with clean, validated signatures (NO default arguments for google-genai schema compatibility)
def query_bridge_damage() -> Dict[str, Any]:
    """Queries FalkorDB for the live flood damage status counts of all surveyed bridges (Washed out, Damaged, Intact)."""
    return get_bridge_damage_summary()

def trace_settlement_access(settlement_id: str) -> Dict[str, Any]:
    """Evaluates on-site facilities and verified vs potential routes to healthcare facilities for a settlement in FalkorDB."""
    sid, sname, candidates = resolve_settlement_id(settlement_id)
    if not sid:
        return {
            "status": "unresolved",
            "message": f"Settlement '{settlement_id}' could not be identified in FalkorDB. No access analysis was performed.",
            "candidates": candidates
        }

    # 1. On-site facility check
    on_site = get_on_site_facilities(sid)

    state, reason, v_paths, p_paths = classify_settlement_access(sid)

    # Verify if active recon missions actually exist for unverified potential corridors
    active_missions = []
    if state == "NEEDS_RECON" and p_paths:
        unverified_rids = list(set(r for p in p_paths for r in p.unverified_road_ids))
        if unverified_rids:
            try:
                g = get_graph()
                m_res = g.query(
                    "MATCH (m:Mission) WHERE m.road_id IN $rids RETURN m.road_id, m.priority_score, m.reason ORDER BY m.priority_score DESC",
                    {"rids": unverified_rids}
                )
                if m_res and m_res.result_set:
                    active_missions = [
                        {"road_id": row[0], "priority_score": row[1], "reason": row[2]}
                        for row in m_res.result_set
                    ]
            except Exception:
                active_missions = []

    return {
        "settlement_id": sid,
        "settlement_name": sname,
        "access_state": state,
        "access_reason": reason,
        "on_site_facilities": on_site,
        "verified_paths_count": len(v_paths),
        "potential_paths_count": len(p_paths),
        "verified_paths": [p.model_dump() for p in v_paths],
        "potential_paths": [p.model_dump() for p in p_paths],
        "active_recon_missions": active_missions,
        "cypher_executed": (
            f"MATCH (f:Facility)-[:AT_LOCATION]->(l:Location {{id: '{sid}'}}); "
            f"MATCH p = (s:Location {{id: '{sid}'}})-[:ROAD_TO*1..7]->(hub:Location)<-[:AT_LOCATION]-(f:Facility) "
            "WHERE ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable') ... (and potential paths query)"
        )
    }

def inspect_road_evidence(road_id: str) -> Dict[str, Any]:
    """Retrieves full evidence chain: observations, sources, episodic memories, and affected settlements for a road corridor."""
    rid, rname, candidates = resolve_road_id(road_id)
    if not rid:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB.",
            "candidates": candidates
        }

    data = get_road_evidence(rid)
    return {
        "road_id": rid,
        "name": data["road"].name,
        "status": data["road"].status,
        "length_km": data["road"].length_km,
        "is_bridge": data["road"].is_bridge,
        "endpoints": [e.model_dump() for e in data["endpoints"]],
        "observations": [o.model_dump() for o in data["observations"]],
        "memories": [m.model_dump() for m in data["memories"]],
        "affected_settlements": data["affected_settlements"],
        "cypher_executed": data["cypher_executed"]
    }

def analyze_road_impact(road_id: str) -> Dict[str, Any]:
    """Finds all settlements that lose topological facility paths if a specific road corridor is blocked."""
    rid, rname, candidates = resolve_road_id(road_id)
    if not rid:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB. No impact calculation was made.",
            "candidates": candidates
        }

    affected, cypher = find_affected_settlements(rid)
    total_hh = sum(s["households"] for s in affected)
    return {
        "road_id": rid,
        "road_name": rname,
        "affected_settlements": affected,
        "affected_count": len(affected),
        "total_households_affected": total_hh,
        "cypher_executed": cypher
    }

def explain_decision_chain(road_id: str) -> Dict[str, Any]:
    """Connects People (Scouts/Sources), Decisions (Memories & Consensus), Projects (Missions), and Physical Network in FalkorDB."""
    rid, rname, candidates = resolve_road_id(road_id)
    if not rid:
        return {
            "status": "unresolved",
            "message": f"Road corridor '{road_id}' could not be identified in FalkorDB.",
            "candidates": candidates
        }
    return get_decision_chain(rid)

def get_active_recon_missions() -> Dict[str, Any]:
    """Retrieves prioritized reconnaissance missions from FalkorDB based on cut-edge centrality."""
    missions = get_ranked_missions()
    return {
        "missions_count": len(missions),
        "missions": [m.model_dump() for m in missions],
        "cypher_executed": "MATCH (m:Mission) RETURN m ORDER BY m.priority_score DESC"
    }

def recall_episodic_memory(limit: int) -> Dict[str, Any]:
    """Retrieves recent episodic memory events logged in FalkorDB (Track 02)."""
    mems = get_episodic_memories(limit=limit, road_id=None)
    return {
        "memories_count": len(mems),
        "memories": [m.model_dump() for m in mems],
        "cypher_executed": "MATCH (e:EpisodicMemory) RETURN e ORDER BY e.timestamp DESC LIMIT $limit"
    }

def record_road_observation(
    road_name_or_id: str,
    reported_status: str,
    note: str,
    source: str = "operator_chat",
    scout_verified: bool = False,
    session_id: str = "ops_session",
    agent_role: str = "Field Coordinator"
) -> Dict[str, Any]:
    """
    Safely records a field observation in FalkorDB.
    Enforces Epistemic Safety Rules:
    - Never mutates unverified roads.
    - Community or operator chat reports cannot promote a road to 'verified_passable' without physical scout verification.
    - Preserves historical evidence and recomputes affected missions.
    """
    rid, rname, candidates = resolve_road_id(road_name_or_id)
    if not rid:
        return {
            "status": "unresolved",
            "message": (
                f"I could not identify road '{road_name_or_id}'. "
                "No database changes were made to FalkorDB."
            ),
            "candidates": candidates
        }

    clean_status = reported_status.lower().strip()
    if clean_status not in ["blocked", "verified_passable", "unknown", "disputed"]:
        clean_status = "unknown"

    # Epistemic Rule: Promotion to 'verified_passable' requires scout verification or explicit override
    source_id = "src_operator"
    if "scout" in source.lower():
        source_id = "src_scout"
    elif "community" in source.lower() or "user" in source.lower():
        source_id = "src_user"

    override = False
    warning_note = None

    if clean_status == "verified_passable":
        if source_id == "src_scout" or scout_verified:
            override = True
        else:
            # Cannot automatically promote to verified_passable from casual report
            clean_status = "unknown"
            warning_note = (
                "Epistemic Safety Constraint: Unverified field report cannot mark a road as 'verified_passable' "
                "without on-ground scout verification. Recorded report as 'unknown' pending reconnaissance."
            )
            note = f"{note} [Unverified passability report held pending scout survey]"

    res = add_observation_and_sync(
        road_id=rid,
        reported_status=clean_status,
        note=note,
        source_id=source_id,
        override=override,
        session_id=session_id,
        agent_role=agent_role
    )

    return {
        "status": "ok",
        "road_id": rid,
        "road_name": rname,
        "reported_status": reported_status,
        "recorded_status": clean_status,
        "resolved_network_status": res["new_status"],
        "resolution_reason": res["resolution_reason"],
        "warning": warning_note or "",
        "observation_id": res["observation_id"],
        "missions_count": res["missions_count"],
        "memory_id": res["memory_id"],
        "cypher_executed": (
            f"MATCH (r:RoadSegment {{id: '{rid}'}}) "
            f"CREATE (o:Observation {{road_id: '{rid}', status: '{clean_status}'}}) "
            f"CREATE (r)-[:HAS_OBSERVATION]->(o) SET r.status = '{res['new_status']}'"
        )
    }

def _check_guardrails(prompt: str) -> Optional[str]:
    """Returns refusal text if prompt violates epistemic boundary constraints."""
    prompt_lower = prompt.lower()
    for pattern in GUARDRAIL_PATTERNS:
        if re.search(pattern, prompt_lower):
            # If the word helicopter/airlift is merely referencing an observational source
            # rather than requesting emergency evacuation dispatch or casualty inquiries, allow it as a field report.
            if "helicopter" in pattern or "airlift" in pattern:
                is_dispatch = bool(re.search(r"\b(dispatch|deploy|send|order|request|call|evacuat|rescue|fly|land|service|airlift)\b", prompt_lower))
                is_observation = bool(re.search(r"\b(view|spotted|saw|found|report|observed|observation|blocked|passable|update|record)\b", prompt_lower))
                if is_observation and not is_dispatch:
                    continue
            return BOUNDARY_REFUSAL_MESSAGE
    return None

def _deterministic_agent_router(prompt: str, session_id: str) -> AgentQueryResponse:
    """
    Intelligent offline router that runs real FalkorDB Cypher queries,
    guaranteeing rich, accurate responses without hardcoded factual defaults or silent substitutions.
    """
    prompt_lower = prompt.lower()
    g = get_graph()

    # 1. Greetings & System Capabilities
    if re.search(r"\b(hello|hi|namaste|hey|help)\b|who are you|what can you do", prompt_lower):
        role = "Field Coordinator"
        loc_res = g.query("MATCH (l:Location) RETURN count(l)")
        loc_count = loc_res.result_set[0][0] if loc_res and loc_res.result_set else 0

        road_res = g.query("MATCH (r:RoadSegment) RETURN count(r)")
        road_count = road_res.result_set[0][0] if road_res and road_res.result_set else 0

        bridge_res = g.query("MATCH (b:Bridge) RETURN count(b)")
        bridge_count = bridge_res.result_set[0][0] if bridge_res and bridge_res.result_set else 0

        answer = (
            f"Namaste! I am **Badi (बाढी)**, your AI agent grounded directly in **FalkorDB** for flood reconnaissance "
            f"and disaster triage across Nepal's Bhote Koshi–Trishuli corridor.\n\n"
            f"Currently tracking in live FalkorDB graph (`badi_trishuli_graph`):\n"
            f"• **{loc_count} Settlements & Hubs** (Rasuwa & Nuwakot Districts)\n"
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
            supporting_evidence={"locations": loc_count, "roads": road_count, "bridges": bridge_count},
            memory_id=mem_id
        )

    # 2. Road Observation / Passability Status Update Intent
    is_obs_keyword = bool(re.search(r"\b(update|record|report(ed)?|observed?|found|spotted|saw)\b", prompt_lower))
    is_status_keyword = bool(re.search(r"\b(passable|blocked|damaged|open|landslide|hazard|washed)\b", prompt_lower))
    is_explicit_obs = bool(re.search(r"\b(record\s+observation|update\s+road|update\s+corridor|mark\s+road|mark\s+as|set\s+status|change\s+status|reflect\s+the\s+change)\b", prompt_lower))

    if (is_obs_keyword and is_status_keyword) or is_explicit_obs:
        role = "Field Coordinator"
        # 1. Resolve target road
        rid, rname, candidates = resolve_road_id(prompt)
        if not rid:
            # Try from endpoints if mentioned (e.g. from Syabrubesi to Briddhim)
            ep_orig, ep_dest = extract_corridor_endpoints(prompt)
            if ep_orig and ep_dest:
                rid, rname, candidates = resolve_road_id(f"{ep_orig[1]} {ep_dest[1]}")

        if not rid:
            cand_str = ", ".join(candidates[:4]) if candidates else "None"
            answer = (
                f"I could not identify which road corridor you wish to update from your query: *'{prompt}'*.\n\n"
                f"Please specify a surveyed corridor name or road ID. Valid options include: {cand_str}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="record_road_observation",
                cypher_executed="None: road unresolved",
                supporting_evidence={"status": "unresolved", "candidates": candidates},
                memory_id=None
            )

        # 2. Extract reported status
        # Priority A: Explicit command directive (e.g. "update status to blocked", "set to passable", "mark as blocked")
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
            # Priority B: Contextual conditions with positional resolution if both appear
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

        # 3. Detect scout verification and source
        scout_verified = bool(re.search(r"\b(scout|physically\s*survey|ground\s*survey|field\s*inspection)\b", prompt_lower))
        source = (
            "aerial_recon_helicopter" if ("helicopter" in prompt_lower or "aerial" in prompt_lower)
            else "field_scout" if scout_verified
            else "operator_chat"
        )

        obs_res = record_road_observation(
            road_name_or_id=rid,
            reported_status=reported_st,
            note=prompt,
            source=source,
            scout_verified=scout_verified,
            session_id=session_id,
            agent_role="Field Coordinator"
        )

        if obs_res.get("status") == "ok":
            rec_st = obs_res["recorded_status"].replace("_", " ").title()
            net_st = obs_res["resolved_network_status"].replace("_", " ").title()

            lines = [
                f"**Field Observation Recorded in FalkorDB**\n",
                f"• **Corridor:** {obs_res['road_name']} (`{obs_res['road_id']}`)",
                f"• **Source:** {source.replace('_', ' ').title()}",
                f"• **Reported Condition:** {reported_st.replace('_', ' ').title()}",
                f"• **Recorded In Graph:** {rec_st}",
                f"• **Active Network Passability:** {net_st}",
                f"• **Resolution:** {obs_res.get('resolution_reason', 'Observation committed.')}"
            ]
            if obs_res.get("warning"):
                lines.append(f"\n> **Epistemic Safety Rule:** {obs_res['warning']}")

            lines.append(f"\n• **Episodic Memory Node:** `{obs_res.get('memory_id')}`")
            lines.append(f"• **Triage Queue Recalibration:** {obs_res.get('missions_count', 0)} active reconnaissance bottleneck missions updated.")

            answer = "\n".join(lines)
            cypher = (
                f"MATCH (r:RoadSegment {{id: '{rid}'}}) "
                f"CREATE (o:Observation {{id: '{obs_res.get('observation_id')}', status: '{obs_res['recorded_status']}', source: '{source}'}})-[:HAS_OBSERVATION]->(r);\n"
                f"MATCH (u)-[rel:ROAD_TO {{road_id: '{rid}'}}]->(v) SET rel.status = '{obs_res['resolved_network_status']}';\n"
                f"MATCH (m:Mission) ... Dynamic cut-edge triage recalculation"
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="record_road_observation",
                cypher_executed=cypher,
                supporting_evidence=obs_res,
                memory_id=obs_res.get("memory_id")
            )

    # 3. Point-to-Point Corridor Routing Intent (A to B Transit Pathfinding & Passability)
    if not any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "what happens if", "blocked if", "depend on", "update", "record", "report"]):
        ep_orig, ep_dest = extract_corridor_endpoints(prompt)
        if ep_orig and ep_dest and ep_orig[0] != ep_dest[0]:
            role = "Route Analyst"
            orig_id, orig_name = ep_orig
            dest_id, dest_name = ep_dest
            evidence = find_route_between(orig_id, dest_id)

            state = evidence.get("overall_state", "NO_ROUTE_FOUND")
            routes = evidence.get("routes", [])

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

                    # Human-readable road names only (no raw IDs in user text)
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
            if evidence.get("primary_route") and evidence["primary_route"].get("road_ids"):
                primary_rid = evidence["primary_route"]["road_ids"][0]

            mem_id = record_episodic_memory(
                session_id, role, "corridor_route_query",
                f"Route Analyst evaluated transit corridor between {orig_name} and {dest_name}: {state}.",
                primary_rid
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="find_route_between",
                cypher_executed=evidence.get("cypher_executed", "MATCH p = (start:Location)-[:ROAD_TO*1..8]->(dest:Location)..."),
                supporting_evidence=evidence,
                memory_id=mem_id
            )

    # 3. Bridge Damage Queries 
    if any(k in prompt_lower for k in ["bridge", "bridges", "washed out", "destroyed"]):
        role = "Route Analyst"
        summary = get_bridge_damage_summary()
        if summary.get("status") == "unavailable" or summary.get("total", 0) == 0:
            answer = (
                "**Surveyed Bridge Damage Status in FalkorDB:**\n\n"
                "No surveyed bridge records were found in the FalkorDB graph database. "
                "Ensure HOT Flood bridge survey data has been imported into the active graph."
            )
            b_dict = {}
        else:
            b_dict = summary.get("counts", {})
            total_bridges = summary.get("total", 0)
            washed = b_dict.get("Washed out", 0)
            damaged = b_dict.get("Damaged", 0)
            intact = b_dict.get("Intact", 0)

            answer = (
                f"**Surveyed Bridge Damage Status in FalkorDB (HOT Flood Nepal Dataset):**\n\n"
                f"• **Washed Out Bridges:** **{washed}** (completely severed by flood waters)\n"
                f"• **Damaged Bridges:** **{damaged}** (structural compromise / hazardous)\n"
                f"• **Intact Bridges:** **{intact}** (confirmed operational)\n"
                f"• **Total Surveyed Bridges:** **{total_bridges}**\n\n"
                f"All {total_bridges} bridges are geographically pinned along the Trishuli River gorge in the interactive map."
            )

        cypher = summary.get("cypher_executed", "MATCH (b:Bridge) RETURN b.status, count(b)")
        mem_id = record_episodic_memory(session_id, role, "bridge_query", f"Reported bridge survey metrics from FalkorDB: {b_dict}.")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="get_bridge_damage_summary",
            cypher_executed=cypher,
            supporting_evidence=b_dict,
            memory_id=mem_id
        )

    # 3. Settlement / Population / District Queries (Distinguishes households from population)
    if any(k in prompt_lower for k in ["settlement", "settlements", "village", "households", "population", "district", "rasuwa", "nuwakot"]) and not any(k in prompt_lower for k in ["hospital", "access", "reach", "route", "trace", "evacuat"]):
        role = "Route Analyst"
        district_filter = ""
        params = {}
        if "rasuwa" in prompt_lower:
            district_filter = "WHERE s.district CONTAINS 'Rasuwa'"
        elif "nuwakot" in prompt_lower:
            district_filter = "WHERE s.district CONTAINS 'Nuwakot'"

        cypher = f"MATCH (s:Location {{type: 'settlement'}}) {district_filter} RETURN s.name, s.district, s.households ORDER BY s.households DESC"
        s_res = g.query(cypher, params)
        settlements = s_res.result_set if s_res and s_res.result_set else []

        if not settlements:
            answer = "No settlements matching the specified district query were found in FalkorDB."
            total_hh = 0
        else:
            total_hh = sum(int(r[2]) for r in settlements)
            lines = [f"• **{r[0]}** ({r[1]}): {r[2]} households" for r in settlements]

            clarification = ""
            if "population" in prompt_lower:
                clarification = (
                    "\n\n*Note on Demographics: FalkorDB stores surveyed **household counts** (total: "
                    f"{total_hh:,} households), not individual population figures. "
                    "At Nepal's average rural household size of ~4.5 persons, this represents approximately "
                    f"{int(total_hh * 4.5):,} individuals across these surveyed communities.*"
                )

            answer = (
                f"**FalkorDB Settlement Registry ({len(settlements)} settlements, {total_hh:,} total households):**\n\n" +
                "\n".join(lines) + clarification
            )

        mem_id = record_episodic_memory(session_id, role, "settlement_query", f"Retrieved {len(settlements)} settlements ({total_hh} households).")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="query_settlements",
            cypher_executed=cypher,
            supporting_evidence={"settlements_count": len(settlements), "total_households": total_hh},
            memory_id=mem_id
        )

    # 4. Memory Recall Intent
    if any(k in prompt_lower for k in ["episodic", "memory", "recent event", "history", "recall", "log"]):
        role = "Field Coordinator"
        evidence = recall_episodic_memory(10)
        mems = evidence["memories"]
        if mems:
            summary_lines = [f"• [{m['timestamp'][:19]}] ({m['agent_role']}) {m['summary']}" for m in mems[:5]]
            answer = (
                f"Recalled {len(mems)} recent episodic memory events from FalkorDB:\n" +
                "\n".join(summary_lines)
            )
        else:
            answer = "No episodic memory events found in FalkorDB."

        mem_id = record_episodic_memory(session_id, role, "memory_recall", f"Field Coordinator retrieved {len(mems)} episodic memories.")
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="recall_episodic_memory",
            cypher_executed=evidence["cypher_executed"],
            supporting_evidence=evidence,
            memory_id=mem_id
        )

    # 5. Blast Radius / Road Impact Intent (No silent road substitution)
    if any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "blocked if", "depend on", "what happens if"]):
        role = "Route Analyst"
        rid, rname, candidates = resolve_road_id(prompt_lower)
        if not rid:
            avail = ", ".join(candidates[:6])
            answer = (
                "I could not identify which road corridor you are inquiring about. "
                "No road impact calculation was performed for an unverified target.\n\n"
                f"Available mapped corridors include: {avail}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_road_impact",
                cypher_executed="None: road resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        evidence = analyze_road_impact(rid)
        affected = evidence["affected_settlements"]
        total_hh = evidence["total_households_affected"]
        names = ", ".join([s["name"] for s in affected]) if affected else "None"

        answer = (
            f"**Topological Blast Radius for {rname} (`{rid}`):**\n"
            f"If {rname} is blocked, {len(affected)} settlement(s) ({names}) "
            f"representing {total_hh:,} surveyed households lose their primary non-blocked topological transit path to regional healthcare hubs.\n"
            f"FalkorDB dynamic re-ranking will automatically elevate alternate bypass corridors into high-priority Recon Missions."
        )

        mem_id = record_episodic_memory(
            session_id, role, "impact_analysis",
            f"Route Analyst evaluated blast radius of {rid}: {len(affected)} settlements affected.",
            rid
        )
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="analyze_road_impact",
            cypher_executed=evidence["cypher_executed"],
            supporting_evidence=evidence,
            memory_id=mem_id
        )

    # 6. People, Projects, and Decisions Connection Intent (No silent road substitution)
    if any(k in prompt_lower for k in ["decision chain", "connect people", "people, projects", "who reported", "chain of decision", "scout reports", "decisions connect"]):
        role = "Field Coordinator"
        rid, rname, candidates = resolve_road_id(prompt_lower)
        if not rid:
            avail = ", ".join(candidates[:6])
            answer = (
                "I could not identify which road corridor you wish to trace decisions for. "
                f"Please specify a known road corridor (e.g. {avail})."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_decision_chain",
                cypher_executed="None: road resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        evidence = explain_decision_chain(rid)
        people_names = ", ".join([s["name"] for s in evidence["people_sources"]]) or "Registered Field Scouts & Authorities"
        obs_count = len(evidence["observations"])
        mem_count = len(evidence["decisions_memories"])
        missions_count = len(evidence["project_missions"])

        answer = (
            f"**People, Projects & Decisions Graph Traversal for {evidence['road_name']} (`{rid}`):**\n\n"
            f"• **Connected People / Observers ({len(evidence['people_sources'])}):** {people_names}\n"
            f"• **Field Observations Recorded ({obs_count}):** Ground-truth reports submitted by scouts and authorities.\n"
            f"• **Episodic Decisions / Memories ({mem_count}):** FalkorDB consensus events, dispute resolutions, and operator notes.\n"
            f"• **Governing Humanitarian Projects ({missions_count}):** Active prioritized Field Missions targeting this corridor.\n"
            f"• **Protected Communities:** {', '.join(evidence['connected_settlements']) or 'Corridor Settlements'}\n\n"
            f"FalkorDB actively unifies field actors, operational triage decisions, and physical topology into an auditable graph chain."
        )

        mem_id = record_episodic_memory(
            session_id, role, "decision_chain_traced",
            f"Field Coordinator traced People-Project-Decision graph chain for {rid}.",
            rid
        )
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="explain_decision_chain",
            cypher_executed=evidence["cypher_executed"],
            supporting_evidence=evidence,
            memory_id=mem_id
        )

    # 7. Road Status Intent (Direct corridor condition & passability query)
    if any(k in prompt_lower for k in ["status", "condition", "how is", "passable", "blocked", "open", "traffic"]) and not any(k in prompt_lower for k in ["blast radius", "impact", "cut off", "what happens if", "hospital", "access", "reach"]):
        rid, rname, candidates = resolve_road_id(prompt_lower)
        if rid:
            role = "Route Analyst"
            evidence = inspect_road_evidence(rid)
            st_raw = evidence.get("status", "unknown")
            st_clean = st_raw.replace("_", " ").capitalize()
            km = evidence.get("length_km", 0.0)
            is_br = evidence.get("is_bridge", False)
            ends = " -> ".join([e["name"] for e in evidence.get("endpoints", [])])
            aff_count = len(evidence.get("affected_settlements", []))

            answer = (
                f"**{rname}: Corridor Status**\n\n"
                f"• **Status:** {st_clean}\n"
                f"• **Endpoints:** {ends} ({km} km)\n"
                f"• **Infrastructure:** {'Bridge structure' if is_br else 'Mountain road corridor'}\n"
                f"• **Dependent Communities:** {aff_count} settlements depend on this link"
            )
            mem_id = record_episodic_memory(session_id, role, "road_status_query", f"Route Analyst checked status of {rid}: {st_raw}.", rid)
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="inspect_road_evidence",
                cypher_executed=evidence.get("cypher_executed", f"MATCH (r:RoadSegment {{id: '{rid}'}}) RETURN r"),
                supporting_evidence=evidence,
                memory_id=mem_id
            )

    # 8. Hospital & Settlement Access Intent (Deterministic sub-20ms FalkorDB execution)
    if any(k in prompt_lower for k in ["hospital", "health post", "clinic", "access", "reach", "route", "trace", "path", "evacuat", "status", "condition", "how is", "open", "isolated"]):
        role = "Route Analyst"
        sid, sname, candidates = resolve_settlement_id(prompt_lower)
        if not sid:
            avail = ", ".join(candidates[:6])
            answer = (
                "I could not identify which settlement you are inquiring about.\n\n"
                f"Available mapped settlements include: {avail}."
            )
            return AgentQueryResponse(
                answer=answer,
                agent_role=role,
                tool_called="unresolved_settlement_access",
                cypher_executed="None: settlement resolution failed",
                supporting_evidence={"candidates": candidates}
            )

        evidence = trace_settlement_access(sid)
        state = evidence["access_state"]
        on_site = evidence.get("on_site_facilities", [])
        v_paths = evidence.get("verified_paths", [])
        p_paths = evidence.get("potential_paths", [])
        active_m = evidence.get("active_recon_missions", [])

        # Build clean, accurate, no-slop answer
        header = f"**{sname}: Health Facility & Access Status**\n\n"
        bullets = []

        # On-site facility info
        if on_site:
            fac_names = ", ".join([f["name"] for f in on_site])
            bullets.append(f"• **On-Site Facility:** {fac_names}")
        else:
            bullets.append("• **On-Site Facility:** None (relies on regional referral)")

        # Regional Referral & Corridor access info
        if state == "VERIFIED_ACCESS":
            v_p = v_paths[0]
            route_str = " -> ".join(v_p["location_names"])
            if on_site and any(f["name"] == v_p["facility_name"] for f in on_site):
                bullets.append(f"• **Regional Referral:** On-site {v_p['facility_name']} operational")
            else:
                bullets.append(f"• **Regional Hospital:** {v_p['facility_name']} ({v_p['total_km']} km via {route_str})")
            bullets.append("• **Corridor Status:** Open and verified passable")

        elif state == "NEEDS_RECON":
            if p_paths:
                p_p = p_paths[0]
                route_str = " -> ".join(p_p["location_names"])
                bullets.append(f"• **Nearest Facility:** {p_p['facility_name']} ({p_p['total_km']} km via {route_str})")
                bullets.append("• **Corridor Status:** Needs Reconnaissance — route contains unverified road segments")
                if active_m:
                    m_names = ", ".join([f"`{m['road_id']}` (Priority: {m['priority_score']}/100)" for m in active_m[:2]])
                    bullets.append(f"• **Active Recon Mission:** {m_names}")
            else:
                bullets.append("• **Corridor Status:** Needs Reconnaissance — awaiting field scout survey")

        elif state == "ISOLATED_ON_MAP":
            bullets.append("• **Access:** Isolated — all mapped routes cut by confirmed road blockages")

        elif state == "NO_MAPPED_ROUTE":
            bullets.append("• **Access:** No mapped road corridors connect this settlement to regional healthcare hubs")

        else:
            bullets.append(f"• **Access:** {state} — {evidence.get('access_reason', '')}")

        answer = header + "\n".join(bullets)

        mem_id = record_episodic_memory(session_id, role, "settlement_triage", f"Route Analyst traced access for {sid}: classified as {state}.", sid)
        return AgentQueryResponse(
            answer=answer,
            agent_role=role,
            tool_called="trace_settlement_access",
            cypher_executed=evidence["cypher_executed"],
            supporting_evidence=evidence,
            memory_id=mem_id
        )

    # 8. Mission Triage & Recon Queue Intent
    if any(k in prompt_lower for k in ["mission", "missions", "scout", "recon", "queue", "priority", "bottleneck", "rank"]):
        role = "Field Coordinator"
        ranked_missions = get_ranked_missions()
        if ranked_missions:
            top_m = ranked_missions[0]
            answer = (
                f"**Current Priority Recon Mission Queue in FalkorDB:**\n\n"
                f"Rank 1: Road `{top_m.road_id}` ({top_m.road_name}) - **Priority Score: {top_m.priority_score}/100**\n"
                f"• **Reason:** {top_m.reason}\n"
                f"• **Sole Lifeline for:** {top_m.sole_lifeline_count} settlements | Total Dependent: {top_m.total_dependent_count} settlements.\n\n"
                f"Total active missions verified in FalkorDB: {len(ranked_missions)}."
            )
            evidence = {"top_mission": top_m.model_dump(), "total_missions": len(ranked_missions)}
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

    # 9. General or Analytical Query Fallback: If Gemini is available, delegate to Gemini grounded in live FalkorDB context
    if GEMINI_API_KEY:
        gemini_res = _execute_gemini_agent_query(prompt, session_id)
        if gemini_res:
            return gemini_res

    # Offline / Static Fallback when no LLM API key is present or LLM call fails
    role = "Route Analyst"
    cypher = (
        "MATCH (l:Location) WITH count(l) AS locs "
        "MATCH (r:RoadSegment) WITH locs, count(r) AS roads "
        "MATCH (b:Bridge) WITH locs, roads, count(b) AS bridges "
        "MATCH (m:Mission) WITH locs, roads, bridges, count(m) AS missions "
        "RETURN locs, roads, bridges, missions"
    )
    res = g.query(cypher)
    row = res.result_set[0] if res and res.result_set else [0, 0, 0, 0]

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
    fds = []
    for tool_def in FALKORDB_DOMAIN_TOOLS:
        fn = tool_def["function"]
        params = _convert_schema(fn.get("parameters", {}))
        fds.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description"),
            parameters=params
        ))
    return [types.Tool(function_declarations=fds)]

def _dispatch_domain_tool(tool_name: str, args: Dict[str, Any], session_id: str) -> Tuple[Dict[str, Any], str, str]:
    """
    Executes domain tool directly on FalkorDB connected graph.
    Returns (tool_result, cypher_executed, agent_role).
    """
    tool_lower = tool_name.lower().strip()

    if tool_lower in ["find_route_between", "route_between"]:
        origin = args.get("origin") or args.get("origin_input") or ""
        destination = args.get("destination") or args.get("destination_input") or ""
        res = find_route_between(origin, destination)
        cypher = res.get("cypher_executed", "MATCH p = (start:Location)-[:ROAD_TO*1..8]->(dest:Location) ...")
        return res, cypher, "Route Analyst"

    elif tool_lower in ["get_settlement_condition", "trace_settlement_access"]:
        sname = args.get("settlement_name") or args.get("settlement_id") or args.get("settlement_name_or_id") or ""
        res = trace_settlement_access(sname)
        cypher = res.get("cypher_executed", "MATCH (s:Location)-[:ROAD_TO*1..7]->(hub:Location) ...")
        return res, cypher, "Route Analyst"

    elif tool_lower in ["get_bridge_damage_summary", "query_bridge_damage"]:
        res = query_bridge_damage()
        cypher = res.get("cypher_executed", "MATCH (b:Bridge) RETURN b.status AS status, count(b) AS count")
        return res, cypher, "Field Coordinator"

    elif tool_lower in ["analyze_road_impact", "road_impact"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        res = analyze_road_impact(rid)
        cypher = res.get("cypher_executed", "MATCH (r:RoadSegment {id: $road_id}) ...")
        return res, cypher, "Route Analyst"

    elif tool_lower in ["get_active_recon_missions", "active_missions"]:
        res = get_active_recon_missions()
        cypher = res.get("cypher_executed", "MATCH (m:Mission) RETURN m ORDER BY m.priority_score DESC")
        return res, cypher, "Field Coordinator"

    elif tool_lower in ["explain_decision_chain", "decision_chain"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        res = explain_decision_chain(rid)
        cypher = res.get("cypher_executed", "MATCH (r:RoadSegment {id: $road_id}) OPTIONAL MATCH ...")
        return res, cypher, "Field Coordinator"

    elif tool_lower in ["recall_episodic_memory", "episodic_memory"]:
        limit = int(args.get("limit", 10))
        res = recall_episodic_memory(limit)
        cypher = res.get("cypher_executed", "MATCH (e:EpisodicMemory) RETURN e ORDER BY e.timestamp DESC LIMIT $limit")
        return res, cypher, "Field Coordinator"

    elif tool_lower in ["record_road_observation", "add_observation"]:
        rid = args.get("road_name_or_id") or args.get("road_id") or ""
        status = args.get("status") or args.get("reported_status") or "unknown"
        note = args.get("note") or f"Observation recorded via agent on {rid}"
        source = args.get("source") or "field_agent"
        scout_verified = bool(args.get("scout_verified", False))
        res = record_road_observation(
            road_name_or_id=rid,
            reported_status=status,
            note=note,
            source=source,
            scout_verified=scout_verified,
            session_id=session_id,
            agent_role="Field Coordinator"
        )
        cypher = res.get("cypher_executed", f"MATCH (r:RoadSegment {{id: '{rid}'}}) CREATE (o:Observation) ...")
        return res, cypher, "Field Coordinator"

    else:
        res = {"status": "unknown_tool", "tool_name": tool_name, "args": args}
        return res, "None: Unknown tool requested", "Route Analyst"

def _format_fallback_tool_answer(tool_name: str, result: Dict[str, Any]) -> str:
    """Provides a clear textual summary when LLM tool output synthesis turn is unavailable."""
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
    """
    Evaluates if a prompt is outside the operational scope of the
    FalkorDB Trishuli flood disaster corridor graph.
    """
    prompt_lower = prompt.lower().strip()

    # Explicit out-of-scope patterns (festivals, general culture, trivia, coding, history, etc.)
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
        r"\bwho won\b"
    ]
    for pattern in out_of_scope_patterns:
        if re.search(pattern, prompt_lower):
            return True

    # If the prompt matches greeting, help, briefing, or operational terms, it is IN SCOPE
    if _has_direct_operational_intent(prompt_lower):
        return False

    # Check for general corridor / disaster keywords
    corridor_keywords = [
        "disaster", "flood", "river", "trishuli", "bhotekoshi", "bhote koshi", "rasuwa", "nuwakot",
        "landslide", "road", "bridge", "settlement", "village", "hospital", "clinic", "health",
        "route", "path", "corridor", "access", "isolated", "passable", "blocked", "mission",
        "recon", "scout", "cut-edge", "centrality", "bottleneck", "household", "observation",
        "report", "falkordb", "graph", "badi", "memory", "decision", "sitrep", "briefing",
        "status", "overview", "summary", "hello", "hi", "namaste", "help"
    ]
    if any(k in prompt_lower for k in corridor_keywords):
        return False

    # Also check if any location or road alias appears in prompt
    for alias in LOCATION_ALIASES.keys():
        if alias in prompt_lower:
            return False
    for alias in ROAD_ALIASES.keys():
        if alias in prompt_lower:
            return False

    # If prompt is asking a question (what is, who is, where is, tell me about) with no corridor context, it is out of scope
    if re.search(r"\b(what is|what's|who is|who's|where is|tell me about|explain)\b", prompt_lower):
        return True

    return False

def _handle_out_of_scope(prompt: str, session_id: str) -> AgentQueryResponse:
    """Returns dedicated out-of-scope metadata without executing graph Cypher traversals."""
    mem_id = record_episodic_memory(
        session_id=session_id,
        agent_role="ScopeEnforcer",
        event_type="out_of_scope_query",
        summary=f"Query evaluated as out of operational scope: {prompt[:80]}"
    )
    return AgentQueryResponse(
        answer=(
            "This query is out of operational scope and not relevant to the Trishuli flood corridor in FalkorDB. "
            "Badi only evaluates mapped topological connectivity, infrastructure damage, and disaster response data "
            "inside the FalkorDB Trishuli flood corridor graph."
        ),
        agent_role="ScopeEnforcer",
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
    """
    Executes grounded agent intelligence query using Google Gemini with dynamic FalkorDB tool calling (Track 01 & 02).
    Enforces real Cypher execution, live graph evidence, and session-scoped episodic memory.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        # Check out of scope boundary first
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

        # Detect function / tool calls from Gemini
        func_call = None
        cand_content = None
        if response and response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        fc = getattr(part, "function_call", None)
                        if fc:
                            func_call = fc
                            cand_content = cand.content
                            break
                    if func_call:
                        break

        if func_call:
            tool_name = func_call.name
            tool_args = dict(func_call.args) if func_call.args else {}

            # Execute real FalkorDB connected query
            tool_result, cypher_executed, agent_role = _dispatch_domain_tool(tool_name, tool_args, session_id)

            # Second turn: synthesize grounded natural language answer from tool output
            final_answer = ""
            try:
                user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
                model_tool_call_content = cand_content or types.Content(role="model", parts=[
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
                print(f"Gemini synthesis second turn error: {synth_err}")

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

            # If user had operational intent but no tool was called, fallback to deterministic router
            if _has_direct_operational_intent(prompt.lower()):
                return _deterministic_agent_router(prompt, session_id)

            # General overview grounded in live FalkorDB counts
            agent_role = "Route Analyst"
            g = get_graph()
            cypher = (
                "MATCH (l:Location) WITH count(l) AS locs "
                "MATCH (r:RoadSegment) WITH locs, count(r) AS roads "
                "MATCH (b:Bridge) WITH locs, roads, count(b) AS bridges "
                "MATCH (m:Mission) WITH locs, roads, bridges, count(m) AS missions "
                "RETURN locs, roads, bridges, missions"
            )
            res = g.query(cypher)
            row = res.result_set[0] if res and res.result_set else [0, 0, 0, 0]
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
        print(f"Gemini agent execution error: {exc}")
        return None
    return None


def _has_direct_operational_intent(prompt_lower: str) -> bool:
    """Detects if prompt matches an explicit operational query for instant sub-30ms FalkorDB execution."""
    # High-level briefings, executive summaries, and broad analytical queries should be handled by LLM with full context
    briefing_keywords = [
        "briefing", "executive", "sitrep", "situation report", "summary", "summarize",
        "overview", "overall", "disaster situation", "strategic", "strategy", "comprehensive",
        "corridor situation", "general situation", "assessment", "state of the corridor"
    ]
    if any(k in prompt_lower for k in briefing_keywords):
        return False

    # 1. Any mention of a known settlement, hospital, or health post
    for alias in LOCATION_ALIASES.keys():
        if alias in prompt_lower:
            return True

    # 2. Any mention of a known road corridor
    for alias in ROAD_ALIASES.keys():
        if alias in prompt_lower:
            return True

    # 3. Explicit operational patterns
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

def execute_agent_query(prompt: str, session_id: str = "ops_session") -> AgentQueryResponse:
    """
    Main grounded agent workflow.
    Evaluates hard guardrails, orchestrates agent roles, queries FalkorDB via safe domain tools,
    logs episodic memory in FalkorDB, and supports Google Gemini with intelligent fallback.
    """
    # 1. Hard Guardrail Check
    refusal = _check_guardrails(prompt)
    if refusal:
        mem_id = record_episodic_memory(
            session_id=session_id,
            agent_role="GuardrailEnforcer",
            event_type="boundary_refusal",
            summary=f"Deterministic refusal triggered for restricted prompt keywords: '{prompt[:60]}...'"
        )
        return AgentQueryResponse(
            answer=refusal,
            agent_role="GuardrailEnforcer",
            tool_called="none_boundary_enforced",
            cypher_executed="EpisodicMemory write only",
            supporting_evidence={"reason": "Epistemic Safety Boundary Violation"},
            memory_id=mem_id
        )

    # 2. Out-of-Scope Domain Boundary Check
    if _is_out_of_scope(prompt):
        return _handle_out_of_scope(prompt, session_id)

    prompt_lower = prompt.lower()

    # Fast Path: Instant sub-30ms execution for operational disaster queries grounded in live FalkorDB Cypher
    if _has_direct_operational_intent(prompt_lower):
        return _deterministic_agent_router(prompt, session_id)

    # 3. Gemini Agent Path with Dynamic Tool Calling (Track 01 & Track 02)
    if GEMINI_API_KEY:
        gemini_res = _execute_gemini_agent_query(prompt, session_id)
        if gemini_res:
            return gemini_res

    # 4. Intelligent offline router grounded directly in FalkorDB Cypher
    return _deterministic_agent_router(prompt, session_id)
