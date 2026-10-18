import pytest
from unittest.mock import MagicMock
from google.genai import types
from backend.agent import _execute_gemini_agent_query
from backend.graph import get_graph

def _make_text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        ]
    )

def test_gemini_tool_calling_find_route_between(monkeypatch):
    """
    Track 01 & 02: Validates that when Gemini generates a tool call,
    the agent executes real FalkorDB Cypher, returns real connected graph evidence,
    and logs session-scoped episodic memory.
    """
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    # Mock tool call candidate
    mock_fc = types.FunctionCall(
        name="find_route_between",
        args={"origin": "Betrawati", "destination": "Syabrubesi"}
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_content = types.Content(role="model", parts=[mock_part])
    mock_cand = types.Candidate(content=mock_content)

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])
    resp_turn2 = _make_text_response("Verified route found from Betrawati to Syabrubesi across 3 hops.")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, resp_turn2]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_101"
    res = _execute_gemini_agent_query("can i reach syabrubesi from betrawati?", session_id=session_id)

    assert res is not None
    assert res.tool_called == "find_route_between"
    assert res.agent_role == "Route Analyst"
    # Real FalkorDB Cypher query was executed and returned
    assert "MATCH p = (start:Location" in res.cypher_executed
    # Real connected graph evidence returned
    assert res.supporting_evidence["status"] == "ok"
    assert res.supporting_evidence["origin_id"] == "loc_betrawati"
    assert res.supporting_evidence["destination_id"] == "loc_syabrubesi"
    assert len(res.supporting_evidence["routes"]) > 0

    # Verify session-scoped episodic memory was created in FalkorDB
    g = get_graph()
    q = g.query("MATCH (e:EpisodicMemory {id: $mid}) RETURN e.session_id, e.event_type", {"mid": res.memory_id})
    assert q and q.result_set
    assert q.result_set[0][0] == session_id
    assert q.result_set[0][1] == "gemini_tool_call"

def test_gemini_tool_calling_record_road_observation(monkeypatch):
    """
    Validates Gemini executing record_road_observation on live FalkorDB graph.
    """
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="record_road_observation",
        args={
            "road_name_or_id": "road_syabru_dhunche",
            "status": "blocked",
            "note": "Field scout reports heavy rockfall near Dhunche",
            "source": "scout_field"
        }
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])
    resp_turn2 = _make_text_response("Field observation recorded. Road status updated to blocked.")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, resp_turn2]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_202"
    res = _execute_gemini_agent_query("field scout says syabrubesi to dhunche is blocked", session_id=session_id)

    assert res is not None
    assert res.tool_called == "record_road_observation"
    assert res.agent_role == "Field Coordinator"
    assert "MATCH (r:RoadSegment" in res.cypher_executed
    assert res.supporting_evidence["road_id"] == "road_syabru_dhunche"
    assert res.supporting_evidence["status"] == "ok"

    # Verify memory
    g = get_graph()
    q = g.query("MATCH (e:EpisodicMemory {id: $mid}) RETURN e.session_id", {"mid": res.memory_id})
    assert q and q.result_set
    assert q.result_set[0][0] == session_id

def test_gemini_tool_calling_bridge_damage_summary(monkeypatch):
    """
    Validates Gemini executing query_bridge_damage returning live bridge counts from FalkorDB.
    """
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="get_bridge_damage_summary",
        args={}
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])
    resp_turn2 = _make_text_response("Bridge summary retrieved from live FalkorDB graph.")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, resp_turn2]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_303"
    res = _execute_gemini_agent_query("what is the bridge damage situation?", session_id=session_id)

    assert res is not None
    assert res.tool_called == "get_bridge_damage_summary"
    assert res.agent_role == "Field Coordinator"
    assert "MATCH (b:Bridge)" in res.cypher_executed
    assert "counts" in res.supporting_evidence
    assert res.supporting_evidence["total"] > 0

def test_gemini_tool_calling_fallback_formatting_on_second_turn_error(monkeypatch):
    """
    Validates that when the second LLM turn errors out, clear grounded fallback
    text is created directly from FalkorDB tool output.
    """
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="analyze_road_impact",
        args={"road_name_or_id": "road_syabru_dhunche"}
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])

    mock_client = MagicMock()
    # Turn 2 throws an error
    mock_client.models.generate_content.side_effect = [resp_turn1, RuntimeError("Network timeout on second turn")]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_404"
    res = _execute_gemini_agent_query("what happens if syabrubesi to dhunche is cut off?", session_id=session_id)

    assert res is not None
    assert res.tool_called == "analyze_road_impact"
    assert "Topological impact" in res.answer
    assert res.supporting_evidence["affected_count"] > 0
    assert "road_id" in res.cypher_executed
