import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from falkordb.edge import Edge  # noqa: E402
from falkordb.node import Node  # noqa: E402
from falkordb.path import Path as FalkorPath  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import backend.graph as graph_module  # noqa: E402
from backend.config import FALKORDB_GRAPH  # noqa: E402
from backend.main import app  # noqa: E402

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "trishuli_flood_network.json"

class MockQueryResult:
    def __init__(self, result_set: List[Any]):
        self.result_set = result_set

class InMemoryFalkorGraph:
    """
    In-memory graph engine simulating FalkorDB graph schema and Cypher query execution
    when FalkorDB is offline, enabling 100% test reproducibility.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.sources = {s["id"]: dict(s) for s in data["sources"]}
        self.locations = {loc["id"]: dict(loc) for loc in data["locations"]}
        self.facilities = {fac["id"]: dict(fac) for fac in data["facilities"]}
        self.road_segments = {r["id"]: dict(r) for r in data["road_segments"]}
        self.observations = {}
        self.missions = {}
        self.memories = []

        # Seed initial observations
        for r_id, road in self.road_segments.items():
            if "initial_observation" in road and road["initial_observation"]:
                obs = road["initial_observation"]
                obs_id = f"obs_init_{r_id}"
                self.observations[obs_id] = {
                    "id": obs_id,
                    "road_id": r_id,
                    "source_id": obs["source_id"],
                    "status": road["status"],
                    "note": obs["note"],
                    "timestamp": obs["timestamp"],
                    "is_simulated": False
                }

        # Seed initial memory
        self.memories.append({
            "id": "mem_seed_init",
            "session_id": "init_session",
            "agent_role": "SystemSeeder",
            "event_type": "network_initialized",
            "summary": "Seeded Bhote Koshi–Trishuli flood road network.",
            "road_id": "",
            "timestamp": "2026-10-16T07:30:00Z"
        })

    def _get_adj(self, allowed_statuses=None, forbidden_statuses=None):
        adj = {}
        for l_id in self.locations:
            adj[l_id] = []
        for r_id, road in self.road_segments.items():
            st = road["status"]
            if allowed_statuses and st not in allowed_statuses:
                continue
            if forbidden_statuses and st in forbidden_statuses:
                continue
            adj[road["u"]].append((road["v"], road))
            adj[road["v"]].append((road["u"], road))
        return adj

    def _find_paths(self, start_loc: str, allowed_statuses=None, forbidden_statuses=None, must_have_status=None):
        if not start_loc or start_loc not in self.locations:
            return []
        adj = self._get_adj(allowed_statuses, forbidden_statuses)
        fac_locs = {fac["location_id"]: fac for fac in self.facilities.values()}

        results = []
        # BFS / DFS up to 7 hops
        stack = [(start_loc, [start_loc], [])]
        while stack:
            curr, path_locs, path_roads = stack.pop(0)
            if len(path_roads) > 7:
                continue
            if curr in fac_locs and len(path_roads) > 0:
                fac = fac_locs[curr]
                road_statuses = [r["status"] for r in path_roads]
                if must_have_status and not any(s in must_have_status for s in road_statuses):
                    pass
                else:
                    results.append((path_locs, path_roads, fac))

            for nxt, road in adj[curr]:
                if nxt not in path_locs:
                    stack.append((nxt, path_locs + [nxt], path_roads + [road]))
        return results

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        params = params or {}
        q = cypher.strip()

        # RETURN 1
        if q == "RETURN 1":
            return MockQueryResult([[1]])

        # MATCH (n) DETACH DELETE n
        if "MATCH (n) DETACH DELETE n" in q:
            self.reset()
            return MockQueryResult([])

        # MATCH (m:Mission) DETACH DELETE m
        if "MATCH (m:Mission) DETACH DELETE m" in q:
            self.missions.clear()
            return MockQueryResult([])

        # Count missions
        if "MATCH (m:Mission) RETURN count(m)" in q:
            return MockQueryResult([[len(self.missions)]])

        # Count memories
        if "MATCH (e:EpisodicMemory) RETURN count(e)" in q:
            return MockQueryResult([[len(self.memories)]])

        # Fetch all missions
        if "MATCH (m:Mission) " in q and "ORDER BY m.priority_score DESC" in q:
            sorted_m = sorted(self.missions.values(), key=lambda x: x["priority_score"], reverse=True)
            rows = [
                [m["id"], m["road_id"], m["road_name"], m["priority_score"], m["reason"],
                 m["sole_lifeline_count"], m["total_dependent_count"], m["status"], m["timestamp"]]
                for m in sorted_m
            ]
            return MockQueryResult(rows)

        # Create Mission
        if "CREATE (m:Mission {" in q:
            m_id = params["mission_id"]
            self.missions[m_id] = {
                "id": m_id,
                "road_id": params["road_id"],
                "road_name": params["road_name"],
                "priority_score": params["priority_score"],
                "reason": params["reason"],
                "sole_lifeline_count": params["sole_lifeline_count"],
                "total_dependent_count": params["total_dependent_count"],
                "status": "proposed",
                "timestamp": params["timestamp"]
            }
            return MockQueryResult([[m_id]])

        # Fetch settlements
        if "MATCH (s:Location {type: 'settlement'}) RETURN s.id AS id" in q:
            settlements = [loc for loc in self.locations.values() if loc["type"] == "settlement"]
            return MockQueryResult([[s["id"], s["name"], s["households"]] for s in settlements])

        # Candidate roads
        if "MATCH (r:RoadSegment) WHERE r.status IN ['unknown', 'disputed']" in q:
            candidates = [r for r in self.road_segments.values() if r["status"] in ["unknown", "disputed"]]
            return MockQueryResult([[r["id"], r["name"], r["is_bridge"], r["status"]] for r in candidates])

        # Facility weight query
        if "MATCH (f:Facility {id: $fid}) RETURN f.importance_weight" in q:
            fid = params["fid"]
            fac = self.facilities.get(fid)
            w = fac["importance_weight"] if fac else 4.0
            return MockQueryResult([[w]])

        # Verified paths query
        if "WHERE ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable')" in q:
            settlement_id = params.get("settlement_id")
            raw_paths = self._find_paths(settlement_id, allowed_statuses=["verified_passable"])
            rows = []
            for path_locs, path_roads, fac in raw_paths:
                f_nodes = [
                    Node(node_id=i+1, alias=f"n{i}", labels=["Location"], properties=self.locations[lid])
                    for i, lid in enumerate(path_locs)
                ]
                f_edges = [
                    Edge(src_node=f_nodes[i], relation="ROAD_TO", dest_node=f_nodes[i+1], edge_id=i+1, properties={
                        "road_id": r["id"],
                        "road_name": r["name"],
                        "status": r["status"],
                        "length_km": r["length_km"],
                        "is_bridge": r["is_bridge"]
                    })
                    for i, r in enumerate(path_roads)
                ]
                p_obj = FalkorPath(nodes=f_nodes, edges=f_edges)
                rows.append([p_obj, fac["id"], fac["name"], fac["importance_weight"]])
            return MockQueryResult(rows)

        # Affected settlements query (SPECIFIC MATCH FIRST)
        if "rel.road_id = $road_id" in q:
            road_id = params["road_id"]
            affected = []
            for s_id, s in self.locations.items():
                if s["type"] != "settlement":
                    continue
                paths = self._find_paths(s_id, forbidden_statuses=["blocked"])
                for p_locs, p_roads, fac in paths:
                    if any(r["id"] == road_id for r in p_roads):
                        affected.append([s["id"], s["name"], s["households"], s["district"]])
                        break
            return MockQueryResult(affected)

        # Potential paths query
        if "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked')" in q:
            settlement_id = params.get("settlement_id")
            raw_paths = self._find_paths(
                settlement_id,
                forbidden_statuses=["blocked"],
                must_have_status=["unknown", "disputed"]
            )
            rows = []
            for path_locs, path_roads, fac in raw_paths:
                f_nodes = [
                    Node(node_id=i+1, alias=f"n{i}", labels=["Location"], properties=self.locations[lid])
                    for i, lid in enumerate(path_locs)
                ]
                f_edges = [
                    Edge(src_node=f_nodes[i], relation="ROAD_TO", dest_node=f_nodes[i+1], edge_id=i+1, properties={
                        "road_id": r["id"],
                        "road_name": r["name"],
                        "status": r["status"],
                        "length_km": r["length_km"],
                        "is_bridge": r["is_bridge"]
                    })
                    for i, r in enumerate(path_roads)
                ]
                p_obj = FalkorPath(nodes=f_nodes, edges=f_edges)
                rows.append([p_obj, fac["id"], fac["name"], fac["importance_weight"]])
            return MockQueryResult(rows)

        # Create Observation
        if "CREATE (o:Observation {" in q:
            obs_id = params["obs_id"]
            self.observations[obs_id] = {
                "id": obs_id,
                "road_id": params["road_id"],
                "source_id": params["source_id"],
                "status": params["reported_status"],
                "note": params["note"],
                "timestamp": params["timestamp"],
                "is_simulated": params.get("is_simulated", False)
            }
            return MockQueryResult([[obs_id]])

        # Road observations and sources (SPECIFIC MATCH FIRST)
        if "-[:FROM_SOURCE]->(s:Source)" in q:
            road_id = params["road_id"]
            obs_list = [o for o in self.observations.values() if o["road_id"] == road_id]
            rows = []
            for o in sorted(obs_list, key=lambda x: x["timestamp"], reverse=True):
                s = self.sources.get(o["source_id"], {"id": o["source_id"], "name": o["source_id"]})
                o_node = Node(node_id=1, alias="o", labels=["Observation"], properties=o)
                s_node = Node(node_id=2, alias="s", labels=["Source"], properties=s)
                rows.append([o_node, s_node])
            return MockQueryResult(rows)

        # Query all observations for a road (PREFIX MATCH)
        if "MATCH (r:RoadSegment {id: $road_id})-[:HAS_OBSERVATION]->(o:Observation)" in q:
            road_id = params["road_id"]
            matching = [
                [o["status"], o["source_id"]]
                for o in self.observations.values()
                if o["road_id"] == road_id
            ]
            return MockQueryResult(matching)

        # Sync road status
        if "SET r.status = $new_status" in q:
            road_id = params["road_id"]
            new_status = params["new_status"]
            if road_id in self.road_segments:
                self.road_segments[road_id]["status"] = new_status
            return MockQueryResult([[road_id, new_status]])

        # Create Episodic Memory
        if "CREATE (e:EpisodicMemory {" in q:
            m_id = params["memory_id"]
            mem_data = {
                "id": m_id,
                "session_id": params["session_id"],
                "agent_role": params["agent_role"],
                "event_type": params.get("event_type", "observation_recorded"),
                "summary": params["summary"],
                "road_id": params.get("road_id", ""),
                "timestamp": params["timestamp"]
            }
            self.memories.insert(0, mem_data)
            return MockQueryResult([[m_id]])

        # Fetch episodic memories
        if "MATCH (e:EpisodicMemory)" in q:
            limit = params.get("limit", 50)
            rows = []
            for m in self.memories[:limit]:
                node = Node(node_id=1, alias="e", labels=["EpisodicMemory"], properties=m)
                rows.append([node])
            return MockQueryResult(rows)

        if "MATCH (e:EpisodicMemory)-[:AFFECTED_ROAD]->(r:RoadSegment {id: $road_id})" in q:
            road_id = params["road_id"]
            limit = params.get("limit", 50)
            matching = [m for m in self.memories if m.get("road_id") == road_id]
            rows = []
            for m in matching[:limit]:
                node = Node(node_id=1, alias="e", labels=["EpisodicMemory"], properties=m)
                rows.append([node])
            return MockQueryResult(rows)

        # Road Segment with endpoints
        if "MATCH (r:RoadSegment {id: $road_id})" in q and "endpoints" in q:
            road_id = params["road_id"]
            if road_id not in self.road_segments:
                return MockQueryResult([])
            r = self.road_segments[road_id]
            r_node = Node(node_id=1, alias="r", labels=["RoadSegment"], properties=r)
            u_loc = Node(node_id=2, alias="u", labels=["Location"], properties=self.locations[r["u"]])
            v_loc = Node(node_id=3, alias="v", labels=["Location"], properties=self.locations[r["v"]])
            return MockQueryResult([[r_node, [u_loc, v_loc]]])

        # Single location query
        if "MATCH (s:Location {id: $id}) RETURN s" in q:
            loc_id = params["id"]
            if loc_id in self.locations:
                node = Node(node_id=1, alias="s", labels=["Location"], properties=self.locations[loc_id])
                return MockQueryResult([[node]])
            return MockQueryResult([])

        # All locations query
        if "MATCH (l:Location) RETURN l" in q:
            rows = []
            for loc in sorted(self.locations.values(), key=lambda x: x["name"]):
                node = Node(node_id=1, alias="l", labels=["Location"], properties=loc)
                rows.append([node])
            return MockQueryResult(rows)

        # All facilities query
        if "MATCH (f:Facility) RETURN f" in q:
            rows = []
            for fac in self.facilities.values():
                node = Node(node_id=1, alias="f", labels=["Facility"], properties=fac)
                rows.append([node])
            return MockQueryResult(rows)

        # All road segments query
        if "MATCH (r:RoadSegment) RETURN r" in q:
            rows = []
            for road in self.road_segments.values():
                node = Node(node_id=1, alias="r", labels=["RoadSegment"], properties=road)
                rows.append([node])
            return MockQueryResult(rows)

        return MockQueryResult([])

class MockFalkorClient:
    def __init__(self, graph):
        self.graph = graph
        self.connection = self

    def ping(self):
        return True

    def select_graph(self, graph_name=FALKORDB_GRAPH):
        return self.graph

# Fixture setup
@pytest.fixture(autouse=True)
def ensure_graph_backend(monkeypatch):
    """
    If live FalkorDB server on localhost:6379 is reachable, use it.
    Otherwise, route get_graph and ping_falkordb to InMemoryFalkorGraph.
    """
    is_live = False
    try:
        from falkordb import FalkorDB
        client = FalkorDB(host=graph_module.FALKORDB_HOST, port=graph_module.FALKORDB_PORT)
        client.connection.ping()
        is_live = True
    except Exception:
        is_live = False

    if not is_live:
        in_memory_graph = InMemoryFalkorGraph()
        mock_client = MockFalkorClient(in_memory_graph)

        for target in ["backend.graph", "backend.queries", "backend.missions", "backend.agent", "scripts.seed_falkordb"]:
            monkeypatch.setattr(f"{target}.get_graph", lambda graph_name=FALKORDB_GRAPH: in_memory_graph, raising=False)
            monkeypatch.setattr(f"{target}.get_falkor_client", lambda: mock_client, raising=False)
            monkeypatch.setattr(f"{target}.ping_falkordb", lambda: True, raising=False)
        yield in_memory_graph
    else:
        from scripts.seed_falkordb import seed_falkordb
        seed_falkordb()
        yield None
        try:
            seed_falkordb()
        except Exception:
            pass

@pytest.fixture
def client():
    return TestClient(app)
