import pytest
from backend.agent import record_road_observation, execute_agent_query
from backend.graph import get_graph
from backend.queries import get_episodic_memories
from backend.mcp_server import tool_record_road_observation
import json

def test_record_road_observation_preserves_custom_session_id():
    """
    Track 02: Multi-Agent Episodic Memory Scoping.
    Verifies that record_road_observation scopes EpisodicMemory to the
    explicit session_id provided, eliminating hardcoded 'agent_session'.
    """
    custom_session = "field_coord_session_402"
    result = record_road_observation(
        road_name_or_id="road_syabru_dhunche",
        reported_status="blocked",
        note="Active rockfall near Dhunche bend",
        source="field_scout",
        scout_verified=True,
        session_id=custom_session,
        agent_role="Field Coordinator"
    )

    assert result["status"] == "ok"
    mem_id = result["memory_id"]
    assert mem_id is not None

    # Verify directly in FalkorDB connected graph
    g = get_graph()
    q_res = g.query(
        "MATCH (e:EpisodicMemory {id: $mem_id}) RETURN e.session_id, e.agent_role, e.summary",
        {"mem_id": mem_id}
    )
    assert q_res and q_res.result_set
    row = q_res.result_set[0]
    saved_session_id = row[0]
    saved_agent_role = row[1]

    assert saved_session_id == custom_session
    assert saved_session_id != "agent_session"
    assert saved_agent_role == "Field Coordinator"

def test_execute_agent_query_passes_session_id_to_episodic_memory():
    """
    Verifies execute_agent_query scopes memory to caller session_id across observation updates.
    """
    session_id = "ops_incident_room_777"
    prompt = "scout verified that road_dhunche_kalika is blocked by landslide. update status."
    response = execute_agent_query(prompt, session_id=session_id)

    assert response.tool_called == "record_road_observation"
    assert response.memory_id is not None

    g = get_graph()
    q_res = g.query(
        "MATCH (e:EpisodicMemory {id: $mem_id}) RETURN e.session_id",
        {"mem_id": response.memory_id}
    )
    assert q_res and q_res.result_set
    assert q_res.result_set[0][0] == session_id

def test_mcp_server_record_observation_scopes_memory():
    """
    Verifies that MCP server tool scopes observations to 'mcp_session' and returns valid JSON.
    """
    raw_res = tool_record_road_observation(
        road_name_or_id="road_briddhim_syabru",
        status="blocked",
        note="Trail washed by gully runoff",
        source="mcp_scout",
        override=False
    )
    parsed = json.loads(raw_res)
    assert parsed.get("road_id") == "road_briddhim_syabru"
    assert parsed.get("resolved_network_status") in ["blocked", "unknown", "disputed"]

    # Verify recent memory in FalkorDB has mcp_session
    g = get_graph()
    check = g.query(
        "MATCH (e:EpisodicMemory {session_id: 'mcp_session'}) RETURN count(e)"
    )
    assert check and check.result_set
    assert check.result_set[0][0] > 0
