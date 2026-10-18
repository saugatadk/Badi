from backend.missions import get_ranked_missions, recompute_missions
from backend.queries import (
    add_observation_and_sync,
    classify_settlement_access,
    get_episodic_memories,
)


def test_centrality_and_landslide_reranking():
    """Validates dynamic re-ranking when a main corridor is blocked by landslide.

    Verifies state change from VERIFIED_ACCESS to NEEDS_RECON and automatic promotion
    of unverified bypass corridors in the reconnaissance queue.
    """
    recompute_missions()
    initial_missions = {mission.road_id: mission.priority_score for mission in get_ranked_missions()}
    state_before, _, verified_paths_before, _ = classify_settlement_access("loc_syabrubesi")
    assert state_before == "VERIFIED_ACCESS"
    assert len(verified_paths_before) >= 1

    init_goljung_score = initial_missions.get("road_syabru_goljung", 0.0)

    result = add_observation_and_sync(
        road_id="road_syabru_dhunche",
        reported_status="blocked",
        note="Heavy monsoon landslide completely buries Pasang Lhamu highway at km 12",
        source_id="src_ndrrma",
        is_simulated=True,
        override=True,
        session_id="test_landslide_session",
        agent_role="DisasterSimulator",
    )

    assert result["new_status"] == "blocked"
    assert result["memory_id"] is not None

    state_after, _, verified_paths_after, potential_paths_after = classify_settlement_access("loc_syabrubesi")
    assert state_after == "NEEDS_RECON"
    assert len(verified_paths_after) == 0
    assert len(potential_paths_after) >= 1

    updated_missions = {mission.road_id: mission.priority_score for mission in get_ranked_missions()}
    new_goljung_score = updated_missions.get("road_syabru_goljung", 0.0)

    assert new_goljung_score >= init_goljung_score
    top_mission = get_ranked_missions()[0]
    assert top_mission.road_id in [
        "road_syabru_goljung",
        "road_gatlang_dhunche",
        "road_timure_briddhim",
        "road_briddhim_syabru",
    ]

    memories = get_episodic_memories(limit=10)
    assert len(memories) >= 1
    landslide_mem = next((item for item in memories if item.road_id == "road_syabru_dhunche"), None)
    assert landslide_mem is not None
    assert "observation_recorded" in landslide_mem.event_type
    assert "blocked" in landslide_mem.summary.lower()
