import uuid
from collections import defaultdict
from datetime import datetime, timezone

from backend.graph import get_graph
from backend.models import MissionModel
from backend.queries import find_potential_paths, find_verified_paths

WEIGHT_SOLE_LIFELINE = 35.0
WEIGHT_DEPENDENTS = 30.0
WEIGHT_FACILITY = 20.0
WEIGHT_BRIDGE_OR_DISPUTED = 15.0
MAX_SCORE = 100.0
DEFAULT_FACILITY_WEIGHT = 4.0
FALLBACK_FACILITY_WEIGHT = 3.8


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def recompute_missions() -> list[MissionModel]:
    """Clears old proposed missions and ranks unverified road segments by urgency.

    Priority balances sole-lifeline cut-edges, count of dependent settlements,
    hospital/health facility weights, and structural hazards (bridges/disputes).
    """
    graph = get_graph()

    # Clear previously proposed missions before recalculating
    graph.query("MATCH (m:Mission) DETACH DELETE m")

    settlements_res = graph.query(
        "MATCH (s:Location {type: 'settlement'}) RETURN s.id AS id, s.name AS name, s.households AS households"
    )
    if not settlements_res or not settlements_res.result_set:
        return []

    settlements = [
        {"id": row[0], "name": row[1], "households": int(row[2])}
        for row in settlements_res.result_set
    ]

    sole_lifeline_count: dict[str, int] = defaultdict(int)
    sole_lifeline_settlements: dict[str, list[str]] = defaultdict(list)
    lifeline_households: dict[str, int] = defaultdict(int)
    total_dependent_count: dict[str, int] = defaultdict(int)
    dependent_settlements: dict[str, list[str]] = defaultdict(list)
    road_target_facilities: dict[str, set[str]] = defaultdict(set)
    road_max_fac_weight: dict[str, float] = defaultdict(float)

    for settlement in settlements:
        verified_paths, _ = find_verified_paths(settlement["id"])
        if len(verified_paths) == 0:
            potential_paths, _ = find_potential_paths(settlement["id"])
            if not potential_paths:
                continue

            paths_road_sets: list[set[str]] = []
            for path in potential_paths:
                unverified_in_path = set(path.unverified_road_ids)
                paths_road_sets.append(unverified_in_path)
                for road_id in unverified_in_path:
                    road_target_facilities[road_id].add(path.facility_name)
                    fac_result = graph.query(
                        "MATCH (f:Facility {id: $fid}) RETURN f.importance_weight",
                        {"fid": path.facility_id},
                    )
                    weight = (
                        float(fac_result.result_set[0][0])
                        if fac_result and fac_result.result_set
                        else DEFAULT_FACILITY_WEIGHT
                    )
                    if weight > road_max_fac_weight[road_id]:
                        road_max_fac_weight[road_id] = weight

            union_roads = set.union(*paths_road_sets) if paths_road_sets else set()
            for road_id in union_roads:
                total_dependent_count[road_id] += 1
                dependent_settlements[road_id].append(settlement["name"])

            # Roads present in all potential paths are topological cut-edges (sole lifelines)
            intersection_roads = set.intersection(*paths_road_sets) if paths_road_sets else set()
            for road_id in intersection_roads:
                sole_lifeline_count[road_id] += 1
                sole_lifeline_settlements[road_id].append(settlement["name"])
                lifeline_households[road_id] += settlement["households"]

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
            "status": str(row[3]),
        }
        for row in roads_res.result_set
    ]

    max_sole = max(max(sole_lifeline_count.values(), default=1), 1)
    max_dep = max(max(total_dependent_count.values(), default=1), 1)

    missions: list[MissionModel] = []
    timestamp_str = _current_timestamp()

    for road in candidate_roads:
        road_id = road["id"]
        road_name = road["name"]
        is_bridge = road["is_bridge"]
        status = road["status"]

        sole_cnt = sole_lifeline_count.get(road_id, 0)
        dep_cnt = total_dependent_count.get(road_id, 0)
        fac_weight = road_max_fac_weight.get(road_id, FALLBACK_FACILITY_WEIGHT)

        score = min(
            MAX_SCORE,
            round(
                WEIGHT_SOLE_LIFELINE * (sole_cnt / max_sole)
                + WEIGHT_DEPENDENTS * (dep_cnt / max_dep)
                + WEIGHT_FACILITY * (fac_weight / 5.0)
                + WEIGHT_BRIDGE_OR_DISPUTED * (1.0 if (is_bridge or status == "disputed") else 0.0),
                1,
            ),
        )

        fac_names = ", ".join(list(road_target_facilities.get(road_id, {"Regional Hospital"}))[:2])
        if sole_cnt > 0:
            settlement_names = ", ".join(sole_lifeline_settlements[road_id][:3])
            households = lifeline_households[road_id]
            reason = (
                f"Cut-edge bottleneck: Sole remaining lifeline for {sole_cnt} settlement(s) "
                f"({settlement_names} - {households} households) to {fac_names}."
            )
        elif dep_cnt > 0:
            settlement_names = ", ".join(dependent_settlements[road_id][:3])
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

        mission_id = f"msn_{uuid.uuid4().hex[:8]}"
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
        graph.query(
            create_mission_cypher,
            {
                "road_id": road_id,
                "mission_id": mission_id,
                "road_name": road_name,
                "priority_score": score,
                "reason": reason,
                "sole_lifeline_count": sole_cnt,
                "total_dependent_count": dep_cnt,
                "timestamp": timestamp_str,
            },
        )

        missions.append(
            MissionModel(
                id=mission_id,
                road_id=road_id,
                road_name=road_name,
                priority_score=score,
                reason=reason,
                sole_lifeline_count=sole_cnt,
                total_dependent_count=dep_cnt,
                status="proposed",
                timestamp=timestamp_str,
            )
        )

    missions.sort(key=lambda item: item.priority_score, reverse=True)
    return missions


def get_ranked_missions() -> list[MissionModel]:
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
    query_result = graph.query(cypher)
    missions = []
    if query_result and query_result.result_set:
        for row in query_result.result_set:
            missions.append(
                MissionModel(
                    id=str(row[0]),
                    road_id=str(row[1]),
                    road_name=str(row[2]),
                    priority_score=float(row[3]),
                    reason=str(row[4]),
                    sole_lifeline_count=int(row[5]),
                    total_dependent_count=int(row[6]),
                    status=str(row[7]),
                    timestamp=str(row[8]),
                )
            )
    return missions
