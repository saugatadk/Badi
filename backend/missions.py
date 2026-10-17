import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Set
from collections import defaultdict
from backend.graph import get_graph
from backend.models import MissionModel

def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def recompute_missions() -> List[MissionModel]:
    """
    Clears old proposed Mission nodes and recomputes prioritized reconnaissance missions
    from FalkorDB based on topological cut-edges, sole lifelines, and healthcare facility weights.
    """
    from backend.queries import find_verified_paths, find_potential_paths

    graph = get_graph()

    # 1. Clear old proposed missions
    graph.query("MATCH (m:Mission) DETACH DELETE m")

    # 2. Fetch all settlements
    settlements_res = graph.query(
        "MATCH (s:Location {type: 'settlement'}) RETURN s.id AS id, s.name AS name, s.households AS households"
    )
    if not settlements_res or not settlements_res.result_set:
        return []

    settlements = [
        {"id": row[0], "name": row[1], "households": int(row[2])}
        for row in settlements_res.result_set
    ]

    # 3. Track lifeline and dependent bottlenecks for unknown / disputed roads
    sole_lifeline_count: Dict[str, int] = defaultdict(int)
    sole_lifeline_settlements: Dict[str, List[str]] = defaultdict(list)
    lifeline_households: Dict[str, int] = defaultdict(int)
    total_dependent_count: Dict[str, int] = defaultdict(int)
    dependent_settlements: Dict[str, List[str]] = defaultdict(list)
    road_target_facilities: Dict[str, Set[str]] = defaultdict(set)
    road_max_fac_weight: Dict[str, float] = defaultdict(float)

    for s in settlements:
        verified_paths, _ = find_verified_paths(s["id"])
        if len(verified_paths) == 0:
            # Settlement has NO verified access -> NEEDS_RECON or ISOLATED_ON_MAP
            potential_paths, _ = find_potential_paths(s["id"])
            if not potential_paths:
                continue

            # Candidate unverified roads across all potential paths for this settlement
            paths_road_sets: List[Set[str]] = []
            for p in potential_paths:
                unverified_in_p = set(p.unverified_road_ids)
                paths_road_sets.append(unverified_in_p)
                for r_id in unverified_in_p:
                    road_target_facilities[r_id].add(p.facility_name)
                    # Query facility weight
                    fac_q = graph.query("MATCH (f:Facility {id: $fid}) RETURN f.importance_weight", {"fid": p.facility_id})
                    w = float(fac_q.result_set[0][0]) if fac_q and fac_q.result_set else 4.0
                    if w > road_max_fac_weight[r_id]:
                        road_max_fac_weight[r_id] = w

            # All roads present in at least one potential path
            union_roads = set.union(*paths_road_sets) if paths_road_sets else set()
            for r_id in union_roads:
                total_dependent_count[r_id] += 1
                dependent_settlements[r_id].append(s["name"])

            # Roads present in ALL remaining potential paths for this settlement -> SOLE LIFELINE
            intersection_roads = set.intersection(*paths_road_sets) if paths_road_sets else set()
            for r_id in intersection_roads:
                sole_lifeline_count[r_id] += 1
                sole_lifeline_settlements[r_id].append(s["name"])
                lifeline_households[r_id] += s["households"]

    # Gather all candidate roads (unknown or disputed)
    roads_res = graph.query(
        "MATCH (r:RoadSegment) WHERE r.status IN ['unknown', 'disputed'] "
        "RETURN r.id AS id, r.name AS name, r.is_bridge AS is_bridge, r.status AS status"
    )
    if not roads_res or not roads_res.result_set:
        return []

    candidate_roads = [
        {
            "id": row[0],
            "name": row[1],
            "is_bridge": bool(row[2]),
            "status": str(row[3])
        }
        for row in roads_res.result_set
    ]

    max_sole = max(max(sole_lifeline_count.values(), default=1), 1)
    max_dep = max(max(total_dependent_count.values(), default=1), 1)

    missions: List[MissionModel] = []
    ts = _current_timestamp()

    for road in candidate_roads:
        r_id = road["id"]
        r_name = road["name"]
        is_bridge = road["is_bridge"]
        status = road["status"]

        sole_cnt = sole_lifeline_count.get(r_id, 0)
        dep_cnt = total_dependent_count.get(r_id, 0)
        fac_w = road_max_fac_weight.get(r_id, 3.8)

        # Priority scoring formula
        score = min(
            100.0,
            round(
                35.0 * (sole_cnt / max_sole) +
                30.0 * (dep_cnt / max_dep) +
                20.0 * (fac_w / 5.0) +
                15.0 * (1.0 if (is_bridge or status == "disputed") else 0.0),
                1
            )
        )

        # Generate descriptive reason
        fac_names = ", ".join(list(road_target_facilities.get(r_id, {"Regional Hospital"}))[:2])
        if sole_cnt > 0:
            settlement_names = ", ".join(sole_lifeline_settlements[r_id][:3])
            hh = lifeline_households[r_id]
            reason = (
                f"Cut-edge bottleneck: Sole remaining lifeline for {sole_cnt} settlement(s) "
                f"({settlement_names} - {hh} households) to {fac_names}."
            )
        elif dep_cnt > 0:
            settlement_names = ", ".join(dependent_settlements[r_id][:3])
            reason = (
                f"Key feeder bypass: Affects alternate transit for {dep_cnt} settlement(s) "
                f"({settlement_names}) connecting to {fac_names}."
            )
        else:
            reason = f"Unverified corridor segment ({status}) connecting to secondary branches."

        if is_bridge:
            reason += " High risk: Flood-prone bridge structure."
        if status == "disputed":
            reason += " High urgency: Conflicting scout and bulletin reports require resolution."

        m_id = f"msn_{uuid.uuid4().hex[:8]}"
        create_mission_cypher = (
            "MATCH (r:RoadSegment {id: $road_id}) "
            "CREATE (m:Mission { "
            "  id: $mission_id, "
            "  road_id: $road_id, "
            "  road_name: $road_name, "
            "  priority_score: $priority_score, "
            "  reason: $reason, "
            "  sole_lifeline_count: $sole_lifeline_count, "
            "  total_dependent_count: $total_dependent_count, "
            "  status: 'proposed', "
            "  timestamp: $timestamp "
            "}) "
            "CREATE (m)-[:INVESTIGATES]->(r) "
            "RETURN m"
        )
        graph.query(create_mission_cypher, {
            "road_id": r_id,
            "mission_id": m_id,
            "road_name": r_name,
            "priority_score": score,
            "reason": reason,
            "sole_lifeline_count": sole_cnt,
            "total_dependent_count": dep_cnt,
            "timestamp": ts
        })

        missions.append(MissionModel(
            id=m_id,
            road_id=r_id,
            road_name=r_name,
            priority_score=score,
            reason=reason,
            sole_lifeline_count=sole_cnt,
            total_dependent_count=dep_cnt,
            status="proposed",
            timestamp=ts
        ))

    # Sort descending by priority_score
    missions.sort(key=lambda x: x.priority_score, reverse=True)
    return missions

def get_ranked_missions() -> List[MissionModel]:
    """Fetches currently proposed missions from FalkorDB sorted by priority score."""
    graph = get_graph()
    cypher = (
        "MATCH (m:Mission) "
        "RETURN m.id AS id, m.road_id AS road_id, m.road_name AS road_name, "
        "       m.priority_score AS priority_score, m.reason AS reason, "
        "       m.sole_lifeline_count AS sole_lifeline_count, "
        "       m.total_dependent_count AS total_dependent_count, "
        "       m.status AS status, m.timestamp AS timestamp "
        "ORDER BY m.priority_score DESC"
    )
    res = graph.query(cypher)
    missions = []
    if res and res.result_set:
        for row in res.result_set:
            missions.append(MissionModel(
                id=str(row[0]),
                road_id=str(row[1]),
                road_name=str(row[2]),
                priority_score=float(row[3]),
                reason=str(row[4]),
                sole_lifeline_count=int(row[5]),
                total_dependent_count=int(row[6]),
                status=str(row[7]),
                timestamp=str(row[8])
            ))
    return missions
