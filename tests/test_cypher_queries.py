from fastapi.testclient import TestClient

import backend.graph as graph_module
from backend.queries import (
    find_affected_settlements,
    find_potential_paths,
    find_verified_paths,
    get_decision_chain,
    get_road_evidence,
)
from scripts.seed_falkordb import seed_falkordb


def test_find_verified_paths_cypher():
    """Tests multi-hop verified paths query from a connected settlement."""
    paths, cypher = find_verified_paths("loc_syabrubesi")
    assert "MATCH p = (start:Location" in cypher or "MATCH p = (s:Location" in cypher
    assert "ALL(rel IN relationships(p) WHERE rel.status = 'verified_passable')" in cypher
    assert len(paths) >= 1
    primary_path = paths[0]
    assert primary_path.facility_name in [
        "Rasuwa District Hospital",
        "Trishuli Provincial Hospital",
        "Betrawati Emergency Health Post",
    ]
    assert primary_path.total_km > 0
    assert len(primary_path.unverified_road_ids) == 0


def test_find_potential_paths_cypher():
    """Tests potential paths query for an unverified settlement."""
    paths, cypher = find_potential_paths("loc_gatlang")
    assert "NONE(rel IN relationships(p) WHERE rel.status = 'blocked')" in cypher
    assert "ANY(rel IN relationships(p) WHERE rel.status IN ['unknown', 'disputed'])" in cypher
    assert len(paths) >= 1
    for path in paths:
        assert len(path.unverified_road_ids) > 0


def test_find_affected_settlements():
    """Tests finding downstream settlements that traverse a specified corridor."""
    affected, cypher = find_affected_settlements("road_syabru_dhunche")
    assert "WHERE NONE(rel IN relationships(p) WHERE rel.status = 'blocked')" in cypher
    assert any(settlement["id"] == "loc_syabrubesi" for settlement in affected)


def test_get_road_evidence_chain():
    """Tests retrieving full evidence chain for a road: endpoints, observations, memories."""
    evidence = get_road_evidence("road_syabru_dhunche")
    assert evidence["road"].id == "road_syabru_dhunche"
    assert len(evidence["endpoints"]) == 2
    assert len(evidence["observations"]) >= 1
    assert "cypher_executed" in evidence


def test_decision_chain_connects_people_projects_decisions():
    """Tests unified decision chain connecting sources, observations, missions, and infrastructure."""
    chain = get_decision_chain("road_syabru_dhunche")
    assert chain["road_id"] == "road_syabru_dhunche"
    assert "Pasang Lhamu" in chain["road_name"]
    assert len(chain["connected_settlements"]) == 2
    assert "MATCH (r:RoadSegment" in chain["cypher_executed"]
    assert "collect(DISTINCT loc.name)" in chain["cypher_executed"]
    assert "observations" in chain
    assert "people_sources" in chain
    assert "decisions_memories" in chain


def test_falkordb_removal_raises_503(client: TestClient, monkeypatch):
    """Verifies that when FalkorDB is unreachable, protected API endpoints return HTTP 503."""
    def raise_offline(*args, **kwargs):
        raise graph_module.FalkorDBUnavailableError(
            "FalkorDB is unreachable at localhost:6379. Graph database service is required."
        )

    monkeypatch.setattr("backend.main.ping_falkordb", raise_offline)
    monkeypatch.setattr("backend.graph.ping_falkordb", raise_offline)
    monkeypatch.setattr("backend.graph.get_graph", raise_offline)
    monkeypatch.setattr("backend.queries.get_graph", raise_offline)

    endpoints = [
        "/api/health",
        "/api/network",
        "/api/missions",
        "/api/roads/road_syabru_dhunche",
        "/api/settlements/loc_syabrubesi/access",
        "/api/memory",
    ]

    for endpoint in endpoints:
        resp = client.get(endpoint)
        assert resp.status_code == 503, f"Endpoint {endpoint} should return 503 when FalkorDB is offline, got {resp.status_code}"
        assert "FalkorDB is unreachable" in resp.json().get("detail", "")


def test_wipe_and_json_import_lifecycle(client: TestClient):
    """Tests graph wipe and subsequent topology ingestion lifecycle."""
    try:
        wipe_res = client.post("/api/graph/wipe")
        assert wipe_res.status_code == 200
        wipe_data = wipe_res.json()
        assert wipe_data["status"] == "wiped"
        assert wipe_data["is_empty"] is True

        net_res = client.get("/api/network")
        assert net_res.status_code == 200
        net_data = net_res.json()
        assert net_data["is_empty"] is True
        assert len(net_data["locations"]) == 0
        assert len(net_data["road_segments"]) == 0

        import_res = client.post("/api/graph/import-json?sample=true")
        assert import_res.status_code == 200
        import_data = import_res.json()
        assert import_data["status"] == "success"
        counts = import_data["imported_counts"]
        assert counts["locations"] == 12
        assert counts["road_segments"] == 15
        assert counts["bridges"] == 58
        assert counts["active_missions"] >= 1

        net_res2 = client.get("/api/network")
        assert net_res2.status_code == 200
        net_data2 = net_res2.json()
        assert net_data2["is_empty"] is False
        assert len(net_data2["locations"]) == 12
        assert len(net_data2["road_segments"]) == 15

        mem_res = client.get("/api/memory")
        assert mem_res.status_code == 200
        memories = mem_res.json()
        assert any(m["event_type"] == "GRAPH_IMPORT" for m in memories)

        custom_payload = {
            "sources": [{"id": "src_test", "name": "Test Source", "url": "http://test", "license": "ODbL"}],
            "locations": [
                {"id": "loc_alpha", "name": "Alpha Outpost", "type": "settlement", "district": "Rasuwa", "lat": 28.1, "lon": 85.3, "households": 120},
                {"id": "loc_beta", "name": "Beta Hub", "type": "facility_hub", "district": "Rasuwa", "lat": 28.2, "lon": 85.4, "households": 200},
            ],
            "facilities": [
                {"id": "fac_clinic", "name": "Beta Clinic", "location_id": "loc_beta", "type": "health_post", "importance_weight": 1.2},
            ],
            "road_segments": [
                {
                    "id": "road_alpha_beta",
                    "name": "Alpha-Beta Link",
                    "u": "loc_alpha",
                    "v": "loc_beta",
                    "length_km": 5.2,
                    "is_bridge": False,
                    "status": "verified_passable",
                },
            ],
        }
        custom_import_res = client.post("/api/graph/import-json", json=custom_payload)
        assert custom_import_res.status_code == 200
        assert custom_import_res.json()["status"] == "success"

        road_res = client.get("/api/roads/road_alpha_beta")
        assert road_res.status_code == 200
        assert road_res.json()["road"]["status"] == "unknown"

    finally:
        seed_falkordb()


def test_universal_json_normalizer_import(client: TestClient):
    """Tests normalizer handling of Node-Link and GeoJSON payloads."""
    try:
        d3_payload = {
            "nodes": [
                {"id": "camp_1", "label": "Base Camp", "latitude": 28.15, "longitude": 85.32, "population": 450},
                {"id": "camp_2", "label": "Forward Hospital", "type": "hospital", "latitude": 28.18, "longitude": 85.35},
            ],
            "links": [
                {"source": "camp_1", "target": "camp_2", "distance": 4.8},
            ],
        }
        res1 = client.post("/api/graph/import-json", json=d3_payload)
        assert res1.status_code == 200
        counts1 = res1.json()["imported_counts"]
        assert counts1["locations"] == 2
        assert counts1["road_segments"] == 1
        assert counts1["facilities"] >= 1

        geojson_payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Village North", "households": 80},
                    "geometry": {"type": "Point", "coordinates": [85.31, 28.14]},
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Medical Outpost", "type": "hospital"},
                    "geometry": {"type": "Point", "coordinates": [85.33, 28.16]},
                },
                {
                    "type": "Feature",
                    "properties": {"source": "Village North", "target": "Medical Outpost", "length_km": 3.2},
                    "geometry": {"type": "LineString", "coordinates": [[85.31, 28.14], [85.33, 28.16]]},
                },
            ],
        }
        res2 = client.post("/api/graph/import-json", json=geojson_payload)
        assert res2.status_code == 200
        counts2 = res2.json()["imported_counts"]
        assert counts2["locations"] >= 2
        assert counts2["road_segments"] >= 1
    finally:
        seed_falkordb()
