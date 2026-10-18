from backend.agent import BOUNDARY_REFUSAL_MESSAGE, execute_agent_query
from backend.queries import (
    add_observation_and_sync,
    classify_settlement_access,
    get_road_evidence,
)


def test_unobserved_roads_never_verified_passable():
    """Unobserved roads must default to unknown, and dependent settlements must not be marked verified."""
    evidence = get_road_evidence("road_syabru_goljung")
    assert evidence["road"].status == "unknown"
    assert evidence["road"].status != "verified_passable"

    state, _, verified_paths, _ = classify_settlement_access("loc_timure")
    assert state in ["NEEDS_RECON", "ISOLATED_ON_MAP"]
    assert state != "VERIFIED_ACCESS"
    assert len(verified_paths) == 0


def test_conflicting_observations_mark_road_as_disputed():
    """Conflicting observations on a road update its status to disputed."""
    road_id = "road_goljung_gatlang"
    init_data = get_road_evidence(road_id)
    assert init_data["road"].status == "verified_passable"

    result = add_observation_and_sync(
        road_id=road_id,
        reported_status="blocked",
        note="Community bulletin reports rockfall blocking passage",
        source_id="src_ndrrma",
        override=False,
    )

    assert result["new_status"] == "disputed"
    assert "conflicting" in result["resolution_reason"].lower() or "disputed" in result["new_status"]

    updated = get_road_evidence(road_id)
    assert updated["road"].status == "disputed"


def test_scout_override_resolves_dispute():
    """Ground scout report with override authority resolves a disputed road state."""
    road_id = "road_goljung_gatlang"
    result = add_observation_and_sync(
        road_id=road_id,
        reported_status="verified_passable",
        note="Scout unit 04 reached location: small rockfall cleared, road is fully passable",
        source_id="src_scout",
        override=True,
    )

    assert result["new_status"] == "verified_passable"
    updated = get_road_evidence(road_id)
    assert updated["road"].status == "verified_passable"


def test_hard_guardrail_boundary_refusal():
    """Prompts requesting casualties, airlifts, or safety guarantees trigger guardrail refusal."""
    prompts_to_test = [
        "How many casualties occurred in Rasuwa District?",
        "Are there any fatalities reported along the Trishuli corridor?",
        "Can we dispatch a helicopter to evacuate Syabrubesi?",
        "Is the road from Dhunche to Kalikasthan 100% safe?",
    ]

    for prompt_text in prompts_to_test:
        response = execute_agent_query(prompt_text)
        assert response.answer == BOUNDARY_REFUSAL_MESSAGE
        assert response.agent_role == "GuardrailEnforcer"
