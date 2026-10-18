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
    """Validates that Gemini tool calling executes FalkorDB Cypher and logs episodic memory."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="find_route_between",
        args={"origin": "Betrawati", "destination": "Syabrubesi"},
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
    agent_response = _execute_gemini_agent_query("can i reach syabrubesi from betrawati?", session_id=session_id)

    assert agent_response is not None
    assert agent_response.tool_called == "find_route_between"
    assert agent_response.agent_role == "Route Analyst"
    assert "MATCH p = (start:Location" in agent_response.cypher_executed
    assert agent_response.supporting_evidence["status"] == "ok"
    assert agent_response.supporting_evidence["origin_id"] == "loc_betrawati"
    assert agent_response.supporting_evidence["destination_id"] == "loc_syabrubesi"
    assert len(agent_response.supporting_evidence["routes"]) > 0

    graph = get_graph()
    memory_result = graph.query(
        "MATCH (e:EpisodicMemory {id: $mid}) RETURN e.session_id, e.event_type",
        {"mid": agent_response.memory_id},
    )
    assert memory_result and memory_result.result_set
    assert memory_result.result_set[0][0] == session_id
    assert memory_result.result_set[0][1] == "gemini_tool_call"


def test_gemini_tool_calling_record_road_observation(monkeypatch):
    """Validates Gemini executing record_road_observation on live FalkorDB graph."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="record_road_observation",
        args={
            "road_name_or_id": "road_syabru_dhunche",
            "status": "blocked",
            "note": "Field scout reports heavy rockfall near Dhunche",
            "source": "scout_field",
        },
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])
    resp_turn2 = _make_text_response("Field observation recorded. Road status updated to blocked.")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, resp_turn2]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_202"
    agent_response = _execute_gemini_agent_query("field scout says syabrubesi to dhunche is blocked", session_id=session_id)

    assert agent_response is not None
    assert agent_response.tool_called == "record_road_observation"
    assert agent_response.agent_role == "Field Coordinator"
    assert "MATCH (r:RoadSegment" in agent_response.cypher_executed
    assert agent_response.supporting_evidence["road_id"] == "road_syabru_dhunche"
    assert agent_response.supporting_evidence["status"] == "ok"

    graph = get_graph()
    memory_result = graph.query("MATCH (e:EpisodicMemory {id: $mid}) RETURN e.session_id", {"mid": agent_response.memory_id})
    assert memory_result and memory_result.result_set
    assert memory_result.result_set[0][0] == session_id


def test_gemini_tool_calling_bridge_damage_summary(monkeypatch):
    """Validates Gemini executing query_bridge_damage returning live bridge counts."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="get_bridge_damage_summary",
        args={},
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])
    resp_turn2 = _make_text_response("Bridge summary retrieved from live FalkorDB graph.")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, resp_turn2]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_303"
    agent_response = _execute_gemini_agent_query("what is the bridge damage situation?", session_id=session_id)

    assert agent_response is not None
    assert agent_response.tool_called == "get_bridge_damage_summary"
    assert agent_response.agent_role == "Field Coordinator"
    assert "MATCH (b:Bridge)" in agent_response.cypher_executed
    assert "counts" in agent_response.supporting_evidence
    assert agent_response.supporting_evidence["total"] > 0


def test_gemini_tool_calling_fallback_formatting_on_second_turn_error(monkeypatch):
    """Validates fallback text generation directly from tool output when LLM second turn fails."""
    monkeypatch.setattr("backend.agent.GEMINI_API_KEY", "mock_key_for_testing")

    mock_fc = types.FunctionCall(
        name="analyze_road_impact",
        args={"road_name_or_id": "road_syabru_dhunche"},
    )
    mock_part = types.Part(function_call=mock_fc)
    mock_cand = types.Candidate(content=types.Content(role="model", parts=[mock_part]))

    resp_turn1 = types.GenerateContentResponse(candidates=[mock_cand])

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [resp_turn1, RuntimeError("Network timeout on second turn")]
    monkeypatch.setattr("backend.agent.genai.Client", lambda api_key: mock_client)

    session_id = "gemini_tool_session_404"
    agent_response = _execute_gemini_agent_query("what happens if syabrubesi to dhunche is cut off?", session_id=session_id)

    assert agent_response is not None
    assert agent_response.tool_called == "analyze_road_impact"
    assert "Topological impact" in agent_response.answer
    assert agent_response.supporting_evidence["affected_count"] > 0
    assert "road_id" in agent_response.cypher_executed
