import re
from pathlib import Path
from backend.queries import get_network_snapshot

def test_frontend_glance_capsule_has_no_hardcoded_metrics():
    """
    Problem D Verification:
    Ensures frontend/index.html does not hardcode static numbers (39, 2,886, etc.)
    inside the glance capsule DOM elements.
    """
    html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Check that glance capsule values are placeholders (--), not hardcoded numbers
    assert '<span class="glance-value text-rose" id="val-hot-bridges">--</span>' in content
    assert '<span class="glance-value text-amber" id="val-hot-buildings">--</span>' in content
    assert '<span class="glance-value text-sky" id="val-active-missions">--</span>' in content
    assert '<span class="glance-value text-emerald" id="val-verified-roads">--</span>' in content
    assert '<span class="glance-value text-rose" id="val-blocked-roads">--</span>' in content

    # Ensure hardcoded numbers 39 or 2,886 are not embedded in glance item titles
    assert 'title="39 Washed Out' not in content
    assert 'title="2,886 Destroyed' not in content

def test_frontend_app_js_populates_glance_capsule_from_live_network():
    """
    Verifies frontend/app.js binds glance elements directly to API network state.
    """
    js_path = Path(__file__).resolve().parent.parent / "frontend" / "app.js"
    assert js_path.exists()
    content = js_path.read_text(encoding="utf-8")

    assert 'setEl("val-hot-bridges", washedCount);' in content
    assert 'setEl("val-hot-buildings", bldgDestroyed' in content
    assert 'setEl("val-active-missions", kpi.active_missions' in content
    assert 'setEl("val-verified-roads", kpi.verified_roads' in content
    assert 'setEl("val-blocked-roads", kpi.blocked_roads' in content

def test_network_api_provides_live_kpis_and_summaries():
    """
    Verifies FalkorDB network snapshot computes live KPIs and bridge/building summaries.
    """
    snapshot = get_network_snapshot()
    assert "kpis" in snapshot
    kpi = snapshot["kpis"]
    assert kpi.verified_roads >= 0
    assert kpi.blocked_roads >= 0
    assert kpi.active_missions >= 0
    assert "bridge_summary" in snapshot
    assert "building_summary" in snapshot
