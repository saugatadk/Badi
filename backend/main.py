import json
from datetime import datetime, timezone
from pathlib import Path

import redis.exceptions
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent import execute_agent_query
from backend.graph import FalkorDBUnavailableError, get_graph, ping_falkordb
from backend.ingest import import_dataset, resolve_default_file
from backend.missions import get_ranked_missions
from backend.models import (
    AddObservationRequest,
    AgentQueryRequest,
    AgentQueryResponse,
    EpisodicMemoryModel,
    GraphImportResponse,
    GraphWipeResponse,
    MissionModel,
    NetworkResponse,
    RoadEvidenceResponse,
    SettlementAccessResponse,
)
from backend.queries import (
    add_observation_and_sync,
    classify_settlement_access,
    find_potential_paths,
    find_verified_paths,
    get_decision_chain,
    get_episodic_memories,
    get_network_snapshot,
    get_road_evidence,
    wipe_falkordb_graph,
)
from scripts.seed_falkordb import seed_falkordb

DEFAULT_OPS_SESSION = "ops_session"

app = FastAPI(
    title="Badi (बाढी) - Flood Reconnaissance & Graph Triage",
    description="FalkorDB Graph Engine for Disaster Response in Nepal",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers for database connectivity issues
@app.exception_handler(FalkorDBUnavailableError)
async def falkordb_unavailable_handler(request: Request, exc: FalkorDBUnavailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": exc.detail, "error": "FalkorDBUnavailable"},
    )


@app.exception_handler(redis.exceptions.ConnectionError)
async def redis_connection_error_handler(request: Request, exc: redis.exceptions.ConnectionError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "FalkorDB is unreachable at localhost:6379. Graph database service is required.",
            "error": "RedisConnectionError",
            "message": str(exc),
        },
    )


@app.exception_handler(redis.exceptions.TimeoutError)
async def redis_timeout_error_handler(request: Request, exc: redis.exceptions.TimeoutError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "FalkorDB connection timed out at localhost:6379.",
            "error": "RedisTimeoutError",
        },
    )


def require_falkordb() -> bool:
    """Dependency verifying FalkorDB connectivity for protected endpoints."""
    ping_falkordb()
    return True


@app.get("/api/health")
def health_check() -> dict:
    """Pings FalkorDB (RETURN 1). Returns 503 if unreachable."""
    try:
        ping_falkordb()
        return {
            "status": "healthy",
            "falkordb": "connected",
            "graph": "badi_trishuli_graph",
            "mode": "live_falkordb",
        }
    except Exception as e:
        raise FalkorDBUnavailableError(f"FalkorDB health check failed: {str(e)}")


@app.get("/api/network", response_model=NetworkResponse, dependencies=[Depends(require_falkordb)])
def get_network():
    """Returns all Location, Facility, RoadSegment geometries and settlement access classifications."""
    return get_network_snapshot()


@app.get("/api/bridges", dependencies=[Depends(require_falkordb)])
def get_bridges() -> dict:
    """Returns surveyed bridge points with damage assessments from FalkorDB as GeoJSON."""
    graph = get_graph()
    bridge_result = graph.query("MATCH (b:Bridge) RETURN b ORDER BY b.name ASC")
    features = []
    if bridge_result and bridge_result.result_set:
        for row in bridge_result.result_set:
            props = row[0].properties
            features.append({
                "type": "Feature",
                "properties": {
                    "id": props.get("id", ""),
                    "name": props.get("name", ""),
                    "status": props.get("status", "Unknown"),
                    "location": props.get("location", ""),
                    "district": props.get("district", ""),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(props.get("lon", 0.0)), float(props.get("lat", 0.0))],
                },
            })
    return {
        "type": "FeatureCollection",
        "features": features,
    }


@app.get("/api/missions", response_model=list[MissionModel], dependencies=[Depends(require_falkordb)])
def get_missions():
    """Returns prioritized reconnaissance missions from FalkorDB."""
    return get_ranked_missions()


@app.get("/api/roads/{road_id}", response_model=RoadEvidenceResponse, dependencies=[Depends(require_falkordb)])
def get_road_details(road_id: str):
    """Returns evidence chain and downstream dependent settlements for a road."""
    try:
        data = get_road_evidence(road_id)
        return RoadEvidenceResponse(
            road=data["road"],
            endpoints=data["endpoints"],
            observations=data["observations"],
            memories=data["memories"],
            affected_settlements=data["affected_settlements"],
            cypher_executed=data["cypher_executed"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/roads/{road_id}/decision-chain", dependencies=[Depends(require_falkordb)])
def get_road_decision_chain(road_id: str):
    """Returns the unified graph chain connecting sources, observations, missions, and infrastructure."""
    try:
        return get_decision_chain(road_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/settlements/{settlement_id}/access", response_model=SettlementAccessResponse, dependencies=[Depends(require_falkordb)])
def get_settlement_access(settlement_id: str):
    """Returns verified and potential paths to healthcare facilities with Cypher metadata."""
    graph = get_graph()
    settlement_res = graph.query("MATCH (s:Location {id: $id}) RETURN s", {"id": settlement_id})
    if not settlement_res or not settlement_res.result_set:
        raise HTTPException(status_code=404, detail=f"Settlement '{settlement_id}' not found")

    settlement_props = settlement_res.result_set[0][0].properties
    state, reason, verified_paths, potential_paths = classify_settlement_access(settlement_id)
    _, verified_cypher = find_verified_paths(settlement_id)
    _, potential_cypher = find_potential_paths(settlement_id)

    return SettlementAccessResponse(
        settlement_id=settlement_id,
        settlement_name=settlement_props.get("name", settlement_id),
        households=int(settlement_props.get("households", 0)),
        district=settlement_props.get("district", ""),
        access_state=state,
        access_reason=reason,
        verified_paths=verified_paths,
        potential_paths=potential_paths,
        cypher_executed=f"-- Verified Paths:\n{verified_cypher}\n\n-- Potential Paths:\n{potential_cypher}",
    )


@app.post("/api/observations", dependencies=[Depends(require_falkordb)])
def add_observation(payload: AddObservationRequest):
    """Adds field observation, updates graph status, logs episodic memory, and re-ranks missions."""
    return add_observation_and_sync(
        road_id=payload.road_id,
        reported_status=payload.reported_status,
        note=payload.note,
        source_id=payload.source_id,
        is_simulated=payload.is_simulated,
        override=payload.override,
    )


@app.post("/api/demo/trigger-landslide", dependencies=[Depends(require_falkordb)])
def trigger_landslide_demo():
    """Simulates a major landslide on road_syabru_dhunche to demonstrate dynamic graph updates."""
    observation_result = add_observation_and_sync(
        road_id="road_syabru_dhunche",
        reported_status="blocked",
        note="Simulated massive debris flow blocks Syabrubesi-Dhunche main corridor",
        source_id="src_ndrrma",
        is_simulated=True,
        override=False,
        session_id="landslide_demo",
        agent_role="DisasterSimulator",
    )
    return {
        "event": "landslide_simulated",
        "affected_road": "road_syabru_dhunche",
        "result": observation_result,
    }


@app.post("/api/demo/reset", dependencies=[Depends(require_falkordb)])
def reset_graph_demo():
    """Resets FalkorDB graph to baseline seed state."""
    seed_falkordb()
    return {"status": "success", "message": "FalkorDB graph reset to baseline seed state."}


@app.post("/api/graph/wipe", response_model=GraphWipeResponse, dependencies=[Depends(require_falkordb)])
def wipe_graph_endpoint():
    """Resets the graph database by deleting all nodes and relationships."""
    wipe_result = wipe_falkordb_graph()
    return GraphWipeResponse(
        status=wipe_result["status"],
        nodes_deleted=wipe_result["nodes_deleted"],
        is_empty=True,
        timestamp=wipe_result["timestamp"],
    )


@app.post("/api/graph/import-json", response_model=GraphImportResponse, dependencies=[Depends(require_falkordb)])
async def import_topology_json(
    request: Request,
    sample: bool = False,
):
    """Imports disaster topology into FalkorDB from a sample file or upload."""
    try:
        if sample:
            import_summary = import_dataset(clear_existing=True)
        else:
            body = await request.body()
            if body and body.strip():
                try:
                    payload_dict = json.loads(body.decode("utf-8"))
                    import_summary = import_dataset(data_dict=payload_dict, clear_existing=True)
                except json.JSONDecodeError as jde:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid JSON payload: {str(jde)}",
                    )
            else:
                import_summary = import_dataset(clear_existing=True)

        imported_counts = import_summary.get("imported_counts", {})
        return GraphImportResponse(
            status="success",
            message=(
                f"Successfully imported {imported_counts.get('locations', 0)} locations, "
                f"{imported_counts.get('road_segments', 0)} road lifelines, and "
                f"{imported_counts.get('bridges', 0)} surveyed bridges into FalkorDB."
            ),
            imported_counts=imported_counts,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to import topology JSON into FalkorDB: {str(e)}",
        )


@app.get("/api/graph/sample-download")
def download_sample_dataset():
    """Serves the Trishuli flood dataset as an attachment."""
    path = resolve_default_file()
    return FileResponse(
        path=str(path),
        media_type="application/json",
        filename="trishuli_flood_dataset.json",
    )


@app.get("/api/memory", response_model=list[EpisodicMemoryModel], dependencies=[Depends(require_falkordb)])
def get_memory_feed(limit: int = 50):
    """Returns episodic memory timeline stored in FalkorDB."""
    return get_episodic_memories(limit=limit)


@app.post("/api/agent/query", response_model=AgentQueryResponse, dependencies=[Depends(require_falkordb)])
def agent_query(payload: AgentQueryRequest):
    """Executes grounded multi-agent query over FalkorDB with guardrail checks."""
    return execute_agent_query(payload.prompt, session_id=payload.session_id or DEFAULT_OPS_SESSION)


# PMTiles partial content endpoint
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
PMTILES_PATH = FRONTEND_DIR / "vendor" / "hot_flood_npl.pmtiles"


@app.get("/vendor/hot_flood_npl.pmtiles")
async def serve_pmtiles(request: Request):
    """Serves PMTiles with full HTTP 206 Partial Content byte-range support."""
    if not PMTILES_PATH.exists():
        raise HTTPException(status_code=404, detail="PMTiles file not found")

    file_size = PMTILES_PATH.stat().st_size
    range_header = request.headers.get("range")

    if not range_header or not range_header.startswith("bytes="):
        with open(PMTILES_PATH, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # Parse byte range header
    try:
        range_val = range_header.replace("bytes=", "").strip()
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = (end - start) + 1

        with open(PMTILES_PATH, "rb") as f:
            f.seek(start)
            chunk = f.read(chunk_size)

        return Response(
            content=chunk,
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "ETag": f'"{file_size}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid byte range request: {str(e)}")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
