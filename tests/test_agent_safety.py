import pytest
from backend.agent import (
    execute_agent_query,
    trace_settlement_access,
    analyze_road_impact,
    record_road_observation,
    _deterministic_agent_router,
    OPENROUTER_TOOLS
)
from backend.queries import get_bridge_damage_summary

def test_unresolved_road_never_substitutes_dhunche(monkeypatch):
    """
    P0 Safety Rule:
    When an unknown or unmapped road is inquired about (e.g. 'between village X and Y'),
    the agent must NEVER silently substitute Syabrubesi–Dhunche.
    It must report the entity as unresolved and present valid options.
    """
    monkeypatch.setattr("backend.agent.OPENROUTER_API_KEY", "")
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
    """
    P0 Safety Rule:
    When an unknown settlement is inquired about, the agent must NEVER silently substitute Syabrubesi.
    It must explicitly report the settlement as unresolved.
    """
    monkeypatch.setattr("backend.agent.OPENROUTER_API_KEY", "")
    response = execute_agent_query("Trace hospital access for Wonderland settlement")
    assert "loc_syabrubesi" not in response.answer.lower() or "could not" in response.answer.lower()
    assert "could not identify which settlement" in response.answer.lower()

    # Direct tool call verification
    result = trace_settlement_access("loc_unknown_fairyland")
    assert result["status"] == "unresolved"
    assert "could not be identified" in result["message"]

def test_observation_recording_refuses_unresolved_road():
    """
    P0 Data Integrity Rule:
    The agent must refuse to record observations on unresolved roads,
    making zero database mutations.
    """
    result = record_road_observation(
        road_name_or_id="phantom_corridor_highway",
        reported_status="blocked",
        note="Spotted flood waters"
    )
    assert result["status"] == "unresolved"
    assert "no database changes were made" in result["message"].lower()

def test_casual_report_cannot_promote_to_verified_passable():
    """
    P0 Epistemic Verification Rule:
    A casual operator or community report claiming a road is passable
    cannot single-handedly promote a road to 'verified_passable' without physical scout survey.
    """
    result = record_road_observation(
        road_name_or_id="road_briddhim_syabru",
        reported_status="verified_passable",
        note="Someone in chat said the road might be open",
        source="community_report",
        scout_verified=False
    )
    assert result["status"] == "ok"
    assert result["recorded_status"] != "verified_passable"
    assert result["recorded_status"] == "unknown"
    assert "Epistemic Safety Constraint" in result["warning"]

def test_arbitrary_cypher_tool_is_not_exposed_to_agent():
    """
    P0 Graph Security Rule:
    Arbitrary Cypher execution tools (e.g. execute_cypher_read_query) must NOT
    be exposed to the agent/LLM. Only parameterized domain tools are permitted.
    """
    tool_names = [t["function"]["name"] for t in OPENROUTER_TOOLS]
    assert "execute_cypher_read_query" not in tool_names
    assert "get_bridge_damage_summary" in tool_names
    assert "analyze_road_impact" in tool_names
    assert "get_settlement_condition" in tool_names

def test_bridge_damage_summary_returns_live_structure():
    """
    P0 Factual Correctness Rule:
    Bridge summary must be derived from FalkorDB query results,
    never relying on hardcoded defaults (39 washed, 4 damaged, 15 intact).
    """
    summary = get_bridge_damage_summary()
    assert summary["status"] in ["ok", "unavailable"]
    if summary["status"] == "ok":
        assert isinstance(summary["counts"], dict)
        assert summary["total"] == sum(summary["counts"].values())

def test_population_query_clarifies_households_vs_individuals(monkeypatch):
    """
    P1 Grounded Communication Rule:
    When asked about population, the agent must clarify that the graph
    stores surveyed household units rather than individual census counts.
    """
    monkeypatch.setattr("backend.agent.OPENROUTER_API_KEY", "")
    response = execute_agent_query("What is the population of settlements in Rasuwa district?")
    assert "household" in response.answer.lower()

def test_route_between_betrawati_and_syabrubesi():
    """
    P0 Point-to-Point Transit Rule:
    The agent must calculate topological paths between two settlements,
    identifying verified passable routes, potential routes with active missions,
    and severed corridors without raw database IDs in user text.
    """
    response = execute_agent_query("can i reach syabrubesi from betrawati")
    assert response.tool_called == "find_route_between"
    assert "Transit Corridor: Betrawati to Syabrubesi" in response.answer
    assert "road_kispang_betrawati" not in response.answer  # No raw road ID in user text
    assert "road_syabru_dhunche" not in response.answer     # No raw road ID in user text
    assert "Kispang–Betrawati Steel Bridge" in response.answer
    assert "Active Recon Mission" in response.answer
    assert "primary_route" in response.supporting_evidence
    assert response.supporting_evidence["origin_id"] == "loc_betrawati"
    assert response.supporting_evidence["destination_id"] == "loc_syabrubesi"

def test_extract_corridor_endpoints_directionality():
    """
    P1 Directionality Rule:
    Extracts origin and destination accurately according to grammatical prepositions.
    """
    from backend.queries import extract_corridor_endpoints

    orig1, dest1 = extract_corridor_endpoints("can i reach syabrubesi from betrawati")
    assert orig1[0] == "loc_betrawati"
    assert dest1[0] == "loc_syabrubesi"

    orig2, dest2 = extract_corridor_endpoints("how to get to dhunche from bidur")
    assert orig2[0] == "loc_bidur"
    assert dest2[0] == "loc_dhunche"

    orig3, dest3 = extract_corridor_endpoints("route from gatlang to kispang")
    assert orig3[0] == "loc_gatlang"
    assert dest3[0] == "loc_kispang"

