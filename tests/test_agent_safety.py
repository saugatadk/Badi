from backend.agent import (
    FALKORDB_DOMAIN_TOOLS,
    analyze_road_impact,
    execute_agent_query,
    record_road_observation,
    trace_settlement_access,
)
from backend.queries import extract_corridor_endpoints, get_bridge_damage_summary


def test_unresolved_road_never_substitutes_dhunche(monkeypatch):
    """Ensures unknown or unmapped roads report as unresolved without silent substitution."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "")
    response = execute_agent_query("What happens if the road between Narnia and Atlantis is blocked?")
    assert "road_syabru_dhunche" not in response.answer.lower() or "could not" in response.answer.lower()
    assert "could not identify which road" in response.answer.lower()
    assert "candidates" in response.supporting_evidence

    # Direct tool call verification
    result = analyze_road_impact("road_completely_unknown_999")
    assert result["status"] == "unresolved"
    assert "could not be identified" in result["message"]
    assert "candidates" in result


def test_unresolved_settlement_never_substitutes_syabrubesi(monkeypatch):
    """Ensures unknown settlements report as unresolved without silent substitution."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "")
    response = execute_agent_query("Trace hospital access for Wonderland settlement")
    assert "loc_syabrubesi" not in response.answer.lower() or "could not" in response.answer.lower()
    assert "could not identify which settlement" in response.answer.lower()

    # Direct tool call verification
    result = trace_settlement_access("loc_unknown_fairyland")
    assert result["status"] == "unresolved"
    assert "could not be identified" in result["message"]


def test_observation_recording_refuses_unresolved_road():
    """Ensures observations on unrecognized roads are rejected without database mutations."""
    result = record_road_observation(
        road_name_or_id="phantom_corridor_highway",
        reported_status="blocked",
        note="Spotted flood waters",
    )
    assert result["status"] == "unresolved"
    assert "no database changes were made" in result["message"].lower()


def test_casual_report_cannot_promote_to_verified_passable():
    """Ensures casual reports cannot promote road status to verified_passable without ground verification."""
    result = record_road_observation(
        road_name_or_id="road_briddhim_syabru",
        reported_status="verified_passable",
        note="Someone in chat said the road might be open",
        source="community_report",
        scout_verified=False,
    )
    assert result["status"] == "ok"
    assert result["recorded_status"] != "verified_passable"
    assert result["recorded_status"] == "unknown"
    assert "Epistemic Safety Constraint" in result["warning"]


def test_arbitrary_cypher_tool_is_not_exposed_to_agent():
    """Ensures arbitrary raw Cypher tools are not exposed in the agent tool registry."""
    tool_names = [t["function"]["name"] for t in FALKORDB_DOMAIN_TOOLS]
    assert "execute_cypher_read_query" not in tool_names
    assert "get_bridge_damage_summary" in tool_names
    assert "analyze_road_impact" in tool_names
    assert "get_settlement_condition" in tool_names


def test_bridge_damage_summary_returns_live_structure():
    """Ensures bridge damage summary reflects live FalkorDB query result structure."""
    summary = get_bridge_damage_summary()
    assert summary["status"] in ["ok", "unavailable"]
    if summary["status"] == "ok":
        assert isinstance(summary["counts"], dict)
        assert summary["total"] == sum(summary["counts"].values())


def test_population_query_clarifies_households_vs_individuals(monkeypatch):
    """Ensures population answers clarify that data represents surveyed household counts."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "")
    response = execute_agent_query("What is the population of settlements in Rasuwa district?")
    assert "household" in response.answer.lower()


def test_route_between_betrawati_and_syabrubesi():
    """Calculates multi-hop corridor between settlements with friendly names and active missions."""
    response = execute_agent_query("can i reach syabrubesi from betrawati")
    assert response.tool_called == "find_route_between"
    assert "Transit Corridor: Betrawati to Syabrubesi" in response.answer
    assert "road_kispang_betrawati" not in response.answer
    assert "road_syabru_dhunche" not in response.answer
    assert "Kispang–Betrawati Steel Bridge" in response.answer
    assert "Active Recon Mission" in response.answer
    assert "primary_route" in response.supporting_evidence
    assert response.supporting_evidence["origin_id"] == "loc_betrawati"
    assert response.supporting_evidence["destination_id"] == "loc_syabrubesi"


def test_extract_corridor_endpoints_directionality():
    """Extracts origin and destination endpoints accurately from preposition phrasing."""
    orig1, dest1 = extract_corridor_endpoints("can i reach syabrubesi from betrawati")
    assert orig1[0] == "loc_betrawati"
    assert dest1[0] == "loc_syabrubesi"

    orig2, dest2 = extract_corridor_endpoints("how to get to dhunche from bidur")
    assert orig2[0] == "loc_bidur"
    assert dest2[0] == "loc_dhunche"

    orig3, dest3 = extract_corridor_endpoints("route from gatlang to kispang")
    assert orig3[0] == "loc_gatlang"
    assert dest3[0] == "loc_kispang"


def test_out_of_scope_query_scope_enforcement():
    """Refuses out-of-scope domain queries without executing graph traversals."""
    response = execute_agent_query("what is dashain?")
    assert response.agent_role == "ScopeEnforcer"
    assert response.tool_called == "none_out_of_scope"
    assert "out of operational scope" in response.cypher_executed.lower()
    assert response.supporting_evidence.get("status") == "out_of_scope"
    assert "dashain is a" not in response.answer.lower()


def test_observation_update_action_from_corridor_phrase():
    """Routes update observations with corridor phrasing to record_road_observation action."""
    prompt = "helicopter view found the path from syabrubesi bazaar to briddhim village passable. can you update this to reflect the change"
    response = execute_agent_query(prompt)
    assert response.agent_role == "Field Coordinator"
    assert response.tool_called == "record_road_observation"
    assert response.supporting_evidence.get("status") == "ok"
    assert response.supporting_evidence.get("road_id") == "road_briddhim_syabru"
    assert "road_briddhim_syabru" in response.cypher_executed
    assert "Epistemic Safety Rule" in response.answer or "Epistemic" in response.supporting_evidence.get("warning", "")
