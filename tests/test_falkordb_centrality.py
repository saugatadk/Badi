import pytest
from backend.queries import (
    classify_settlement_access,
    add_observation_and_sync,
    get_episodic_memories
)
from backend.missions import recompute_missions, get_ranked_missions

def test_centrality_and_landslide_reranking():
    """
    Track 01 & Track 02 Centrality and Dynamic Re-ranking Test:
    1. Initially, Syabrubesi has VERIFIED_ACCESS via road_syabru_dhunche.
    2. A landslide occurs on road_syabru_dhunche (marked as blocked).
    3. Syabrubesi immediately shifts from VERIFIED_ACCESS to NEEDS_RECON.
    4. Alternate bypass corridors (road_syabru_goljung / road_gatlang_dhunche) jump in priority score.
    5. An EpisodicMemory node is recorded in FalkorDB capturing the network mutation event.
    """
    # 1. Baseline state
    recompute_missions()
    init_missions = {m.road_id: m.priority_score for m in get_ranked_missions()}
    state_before, _, v_paths_before, _ = classify_settlement_access("loc_syabrubesi")
    assert state_before == "VERIFIED_ACCESS"
    assert len(v_paths_before) >= 1

    init_goljung_score = init_missions.get("road_syabru_goljung", 0.0)

    # 2. Trigger landslide on road_syabru_dhunche
    result = add_observation_and_sync(
        road_id="road_syabru_dhunche",
        reported_status="blocked",
        note="Heavy monsoon landslide completely buries Pasang Lhamu highway at km 12",
        source_id="src_ndrrma",
        is_simulated=True,
        override=True,
        session_id="test_landslide_session",
        agent_role="DisasterSimulator"
    )

    assert result["new_status"] == "blocked"
    assert result["memory_id"] is not None

    # 3. Syabrubesi shifts to NEEDS_RECON
    state_after, reason_after, v_paths_after, p_paths_after = classify_settlement_access("loc_syabrubesi")
    assert state_after == "NEEDS_RECON"
    assert len(v_paths_after) == 0
    assert len(p_paths_after) >= 1

    # 4. Check that alternate roads have elevated priority score in FalkorDB
    updated_missions = {m.road_id: m.priority_score for m in get_ranked_missions()}
    new_goljung_score = updated_missions.get("road_syabru_goljung", 0.0)

    # road_syabru_goljung is now the lifeline to reach Goljung/Gatlang
    assert new_goljung_score >= init_goljung_score
    top_mission = get_ranked_missions()[0]
    assert top_mission.road_id in ["road_syabru_goljung", "road_gatlang_dhunche", "road_timure_briddhim", "road_briddhim_syabru"]

    # 5. Verify EpisodicMemory in FalkorDB (Track 02)
    memories = get_episodic_memories(limit=10)
    assert len(memories) >= 1
    landslide_mem = next((m for m in memories if m.road_id == "road_syabru_dhunche"), None)
    assert landslide_mem is not None
    assert "observation_recorded" in landslide_mem.event_type
    assert "blocked" in landslide_mem.summary.lower()
