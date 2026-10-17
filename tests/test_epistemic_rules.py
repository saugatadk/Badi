import pytest
from backend.queries import (
    classify_settlement_access,
    add_observation_and_sync,
    get_road_evidence
)
from backend.agent import execute_agent_query, BOUNDARY_REFUSAL_MESSAGE

def test_unobserved_roads_never_verified_passable():
    """
    Epistemic Safety Rule 1:
    Unobserved roads must default to 'unknown', never 'verified_passable'.
    Settlements with unverified paths must NEVER be classified as VERIFIED_ACCESS.
    """
    # road_syabru_goljung starts as unknown
    evidence = get_road_evidence("road_syabru_goljung")
    assert evidence["road"].status == "unknown"
    assert evidence["road"].status != "verified_passable"

    # loc_timure only has a blocked gorge highway and an unknown trail to Briddhim
    state, reason, v_paths, p_paths = classify_settlement_access("loc_timure")
    assert state in ["NEEDS_RECON", "ISOLATED_ON_MAP"]
    assert state != "VERIFIED_ACCESS"
    assert len(v_paths) == 0

def test_conflicting_observations_mark_road_as_disputed():
    """
    Epistemic Safety Rule 2:
    When conflicting observations (e.g., one 'verified_passable' and one 'blocked')
    are recorded on a road, FalkorDB marks the road status as 'disputed'.
    """
    road_id = "road_goljung_gatlang"
    # Initially verified_passable from local municipality
    init_data = get_road_evidence(road_id)
    assert init_data["road"].status == "verified_passable"

    # Conflicting report arrives: a community bulletin claims a landslide blocked it
    result = add_observation_and_sync(
        road_id=road_id,
        reported_status="blocked",
        note="Community bulletin reports rockfall blocking passage",
        source_id="src_ndrrma",
        override=False
    )

    assert result["new_status"] == "disputed"
    assert "conflicting" in result["resolution_reason"].lower() or "disputed" in result["new_status"]

    # Verify road in FalkorDB reflects 'disputed'
    updated = get_road_evidence(road_id)
    assert updated["road"].status == "disputed"

def test_scout_override_resolves_dispute():
    """
    Epistemic Safety Rule 3:
    A verified physical field scout report with override authority can resolve
    a disputed road state to verified_passable or blocked.
    """
    road_id = "road_goljung_gatlang"
    # Resolve the dispute with an on-ground scout unit
    result = add_observation_and_sync(
        road_id=road_id,
        reported_status="verified_passable",
        note="Scout unit 04 reached location: small rockfall cleared, road is fully passable",
        source_id="src_scout",
        override=True
    )

    assert result["new_status"] == "verified_passable"
    updated = get_road_evidence(road_id)
    assert updated["road"].status == "verified_passable"

def test_hard_guardrail_boundary_refusal():
    """
    Epistemic Boundary Guardrail:
    Prompts mentioning casualties, fatalities, helicopters, or absolute safety guarantees
    MUST immediately return the deterministic boundary refusal message.
    """
    prompts_to_test = [
        "How many casualties occurred in Rasuwa District?",
        "Are there any fatalities reported along the Trishuli corridor?",
        "Can we dispatch a helicopter to evacuate Syabrubesi?",
        "Is the road from Dhunche to Kalikasthan 100% safe?"
    ]

    for p in prompts_to_test:
        response = execute_agent_query(p)
        assert response.answer == BOUNDARY_REFUSAL_MESSAGE
        assert response.agent_role == "GuardrailEnforcer"
