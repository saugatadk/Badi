# Implementation Plan

## Overview

Badi is a FalkorDB-grounded multi-agent flood response system for Nepal's Trishuli corridor. The goal is to ensure the project fully satisfies the hackathon's two target tracks: (1) Agents That Act on Connected Data, and (2) Agent Memory and Coordination. The implementation must ensure the agent genuinely acts on live connected graph data (not static snapshots), remembers what it learns across sessions (persistent episodic memory in FalkorDB), and can show the exact path/Cypher/evidence behind every answer. This plan identifies concrete problems found during investigation and prescribes fixes to make the implementation "totally solid."

## Problems Identified

### Problem A: Gemini Agent Path Does Not Act on Connected Data
**File:** `backend/agent.py`, function `_execute_gemini_agent_query()` (line 1100)

When `GEMINI_API_KEY` is configured, queries route through Gemini 3.1 Flash Lite. The function sends a static text snapshot of the entire graph (`get_falkordb_live_context()`) as system instruction context and receives plain text back. It does **not** execute any Cypher queries, does **not** call any of the defined `FALKORDB_DOMAIN_TOOLS`, and fabricates the `cypher_executed` field (e.g., `"FalkorDB Graph Snapshot Grounding"` instead of an actual query that ran). This violates Track 01 — the agent does not act on connected data when Gemini is active; it merely reasons from a stale text dump.

### Problem B: Hardcoded `session_id="agent_session"` Breaks Session-Scoped Memory
**File:** `backend/agent.py`, function `record_road_observation()` (line ~264)

When the agent records a road observation via `record_road_observation()`, it calls `add_observation_and_sync()` with hardcoded `session_id="agent_session"` and `agent_role="Field Coordinator"` instead of using the actual conversation `session_id` passed to `execute_agent_query()`. This means all agent-recorded observations are attributed to a single fake session, breaking Track 02 — agents cannot remember what they learned in a specific session because memory records are not scoped to the correct session.

### Problem C: Missing `mcp` Package Dependency
**File:** `requirements.txt`; **referenced by:** `backend/mcp_server.py` (line 15)

`mcp_server.py` imports `from mcp.server.mcpserver import MCPServer`, but the `mcp` package is not listed in `requirements.txt`. The MCP server will crash with `ImportError` if attempted to run.

### Problem D: Frontend Hardcoded Metrics
**File:** `frontend/index.html` (lines 41-66)

The glance capsule displays hardcoded static values (39 bridges washed, 2,886 buildings destroyed, 6 active missions, 6/3 verified/blocked roads) directly in HTML. These are not fetched from the FalkorDB API. The `/api/network` endpoint computes these live, but the UI ignores the live data for the headline metrics.

### Problem E: Unused / Misconfigured Environment Variables
**File:** `.env.example`, `backend/config.py`

`.env.example` defines `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`, but neither is referenced anywhere in the codebase. `config.py` only reads `GEMINI_API_KEY` and `GEMINI_MODEL`. This is dead configuration that could confuse judges.

### Problem F: README/Model Version Mismatch
**File:** `README.md` (line 151); `backend/config.py` (line 13)

README says queries route through `gemini-2.5-flash`; `config.py` defaults to `gemini-3.1-flash-lite`. Inconsistency between documentation and code.

### Problem G: Out-of-Scope Test Is Conditionally Skipped
**File:** `tests/test_agent_safety.py`, `test_out_of_scope_query_scope_enforcement()` (line 150)

The test is skipped when `GEMINI_API_KEY` is not configured, meaning the out-of-scope boundary behavior is not validated in CI/test runs without an API key. The deterministic router already handles out-of-scope queries, so the test should run regardless.

### Problem H: N+1 Bridge Queries in Network Snapshot
**File:** `backend/queries.py`, function `get_network_snapshot()` (line ~640)

For every road segment, a separate Cypher query is executed to fetch associated bridges. With 15 roads this is acceptable, but it is an O(N) query pattern that will not scale and is inefficient.

### Problem I: Gemini Path Returns Generic Evidence
**File:** `backend/agent.py`, `_execute_gemini_agent_query()` (line ~1170)

When Gemini generates a response, `supporting_evidence` is hardcoded as `{"model": GEMINI_MODEL, "grounded_in": "FalkorDB", "context_grounded": True}` — no actual graph evidence (paths, nodes, relationships) is included. This violates Track 01's requirement to "show the path behind an answer."

## [Functions]

### New Functions
None required.

### Modified Functions

| Function | File | Required Changes |
|----------|------|-----------------|
| `_execute_gemini_agent_query` | `backend/agent.py` | Implement actual Gemini function calling: pass `FALKORDB_DOMAIN_TOOLS` as tools, handle tool call responses by executing the corresponding FalkorDB queries, set `cypher_executed` to the real query, populate `supporting_evidence` with real graph data. |
| `record_road_observation` | `backend/agent.py` | Add `session_id: str = "ops_session"` and `agent_role: str = "Field Coordinator"` parameters; pass them to `add_observation_and_sync()` instead of hardcoded values. |
| `loadAllData` | `frontend/app.js` | After fetching network data, populate glance capsule DOM elements with live KPI values. |
| `test_out_of_scope_query_scope_enforcement` | `tests/test_agent_safety.py` | Remove `pytest.skip()` conditional so test always runs. |

### Removed Functions
None.

## [Classes]

### New Classes
None required.

### Modified Classes
None required.

### Removed Classes
None.

## [Dependencies]

| Dependency | Change | Reason |
|------------|--------|--------|
| `mcp` | Add to `requirements.txt` | `backend/mcp_server.py` imports `mcp.server.mcpserver` but package is not declared. |
| `google-genai` | No version change | Already present; used for Gemini path. |
| `falkordb` | No version change | Already present; sole database. |

## [Testing]

### Existing Test Modifications
1. **`tests/test_agent_safety.py`**: Remove `pytest.skip()` in `test_out_of_scope_query_scope_enforcement()` so out-of-scope behavior is validated without requiring `GEMINI_API_KEY`.

### New Test Coverage Needed
1. **`tests/test_agent_gemini_tool_calling.py`** (new): Test that when Gemini is configured, it actually executes FalkorDB tools and returns real `cypher_executed` and `supporting_evidence`. Mock the Gemini client to return tool calls and verify the corresponding queries execute.
2. **`tests/test_session_scoped_memory.py`** (new): Test that `record_road_observation()` records episodic memory with the correct `session_id` (not hardcoded `"agent_session"`).
3. **`tests/test_frontend_kpi_population.py`** (conceptual): Frontend tests are not currently part of the suite; consider adding a lightweight check that glance capsule values come from the API.

### Validation Strategy
- Run `pytest -v` after all changes to ensure all 12 existing tests pass.
- Manually verify the agent CLI and web UI show real Cypher queries, real evidence, and session-scoped memory.
- Verify `mcp_server.py` imports without error after adding `mcp` to requirements.

## [Implementation Order]

1. **Fix `requirements.txt`** — Add `mcp` package so the MCP server can import.
2. **Fix `record_road_observation` session_id** — Pass through `session_id` and `agent_role` instead of hardcoding.
3. **Fix Gemini tool calling** — Implement actual function-calling in `_execute_gemini_agent_query` so Gemini executes real FalkorDB queries and returns real `cypher_executed`/`supporting_evidence`.
4. **Fix frontend hardcoded metrics** — Replace static HTML values with API-driven population in `app.js` and `index.html`.
5. **Clean up configuration** — Remove unused `OPENROUTER_*` from `.env.example`; fix README/model version mismatch.
6. **Fix test** — Remove `pytest.skip()` from out-of-scope test.
7. **Run full test suite** — `pytest -v` to validate all changes.
8. **Manual end-to-end verification** — Start FalkorDB, seed graph, launch app, verify chat shows real Cypher paths and memory is session-scoped.

## Summary

The core architecture is sound: FalkorDB is the sole database, Cypher queries are parameterized (no injection risk), epistemic safety rules are enforced, guardrails block out-of-domain queries, and the frontend has a "Show Graph Path & Cypher Behind Answer" drawer. However, three issues directly undermine the hackathon's core objectives:

1. **When Gemini is active, the agent does not act on connected data** — it reasons from a static text snapshot, not live queries.
2. **Memory is not properly session-scoped** — `record_road_observation` hardcodes `session_id="agent_session"`.
3. **The `mcp` dependency is missing** — the MCP server cannot run.

These should be fixed before submission. Refer to `@implementation_plan.md` for the complete breakdown.