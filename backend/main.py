import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Depends, status, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import redis.exceptions

from backend.graph import ping_falkordb, FalkorDBUnavailableError
from backend.models import (
    AddObservationRequest,
    AgentQueryRequest,
    AgentQueryResponse,
    NetworkResponse,
    RoadEvidenceResponse,
    SettlementAccessResponse,
    MissionModel,
    EpisodicMemoryModel,
    GraphWipeResponse,
    GraphImportResponse
)
from backend.queries import (
    get_network_snapshot,
    get_road_evidence,
    classify_settlement_access,
    find_verified_paths,
    find_potential_paths,
    add_observation_and_sync,
    get_episodic_memories,
    wipe_falkordb_graph
)
from backend.missions import get_ranked_missions
from backend.agent import execute_agent_query
from backend.ingest import import_dataset, resolve_default_file
from scripts.seed_falkordb import seed_falkordb

app = FastAPI(
    title="Badi (बाढी) - Flood Reconnaissance & Graph Triage",
    description="FalkorDB Graph Engine for Disaster Response in Nepal",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FalkorDB Unreachable Exception Handlers (Mandatory Hackathon Removal Test)
@app.exception_handler(FalkorDBUnavailableError)
async def falkordb_unavailable_handler(request: Request, exc: FalkorDBUnavailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": exc.detail, "error": "FalkorDBUnavailable"}
    )

@app.exception_handler(redis.exceptions.ConnectionError)
async def redis_connection_error_handler(request: Request, exc: redis.exceptions.ConnectionError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "FalkorDB is unreachable at localhost:6379. Graph database service is required.",
            "error": "RedisConnectionError",
            "message": str(exc)
        }
    )

@app.exception_handler(redis.exceptions.TimeoutError)
async def redis_timeout_error_handler(request: Request, exc: redis.exceptions.TimeoutError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "FalkorDB connection timed out at localhost:6379.",
            "error": "RedisTimeoutError"
        }
    )

# Dependency to verify FalkorDB is live on all protected endpoints
def require_falkordb():
    ping_falkordb()
    return True

@app.get("/api/health")
def health_check():
    """Pings FalkorDB (RETURN 1). Returns 503 if unreachable."""
    try:
        ping_falkordb()
        return {
            "status": "healthy",
            "falkordb": "connected",
            "graph": "badi_trishuli_graph",
            "mode": "live_falkordb"
        }
    except Exception as e:
        raise FalkorDBUnavailableError(
            f"FalkorDB health check failed: {str(e)}"
        )

@app.get("/api/network", response_model=NetworkResponse, dependencies=[Depends(require_falkordb)])
def get_network():
    """Returns all Location, Facility, RoadSegment geometries and settlement access classifications."""
    snapshot = get_network_snapshot()
    return snapshot

@app.get("/api/bridges", dependencies=[Depends(require_falkordb)])
def get_bridges():
    """Returns all 58 surveyed bridge points with damage assessments from FalkorDB."""
    from backend.graph import get_graph
    graph = get_graph()
    b_res = graph.query("MATCH (b:Bridge) RETURN b ORDER BY b.name ASC")
    features = []
    if b_res and b_res.result_set:
        for row in b_res.result_set:
            b = row[0].properties
            features.append({
                "type": "Feature",
                "properties": {
                    "id": b.get("id", ""),
                    "name": b.get("name", ""),
                    "status": b.get("status", "Unknown"),
                    "location": b.get("location", ""),
                    "district": b.get("district", "")
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(b.get("lon", 0.0)), float(b.get("lat", 0.0))]
                }
            })
    return {
        "type": "FeatureCollection",
        "features": features
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
            cypher_executed=data["cypher_executed"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/roads/{road_id}/decision-chain", dependencies=[Depends(require_falkordb)])
def get_road_decision_chain(road_id: str):
    """
    Returns the unified graph chain connecting People (Scouts/Sources),
    Decisions (Observations & Episodic Memories), Projects (Missions),
    and Infrastructure (Roads, Bridges, Settlements) in FalkorDB.
    """
    from backend.queries import get_decision_chain
    try:
        return get_decision_chain(road_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/settlements/{settlement_id}/access", response_model=SettlementAccessResponse, dependencies=[Depends(require_falkordb)])
def get_settlement_access(settlement_id: str):
    """Returns verified and potential paths to healthcare facilities with Cypher metadata."""
    from backend.graph import get_graph
    graph = get_graph()
    s_res = graph.query("MATCH (s:Location {id: $id}) RETURN s", {"id": settlement_id})
    if not s_res or not s_res.result_set:
        raise HTTPException(status_code=404, detail=f"Settlement '{settlement_id}' not found")
    
    s_props = s_res.result_set[0][0].properties
    state, reason, v_paths, p_paths = classify_settlement_access(settlement_id)
    _, v_cypher = find_verified_paths(settlement_id)
    _, p_cypher = find_potential_paths(settlement_id)

    return SettlementAccessResponse(
        settlement_id=settlement_id,
        settlement_name=s_props.get("name", settlement_id),
        households=int(s_props.get("households", 0)),
        district=s_props.get("district", ""),
        access_state=state,
        access_reason=reason,
        verified_paths=v_paths,
        potential_paths=p_paths,
        cypher_executed=f"-- Verified Paths:\n{v_cypher}\n\n-- Potential Paths:\n{p_cypher}"
    )

@app.post("/api/observations", dependencies=[Depends(require_falkordb)])
def add_observation(payload: AddObservationRequest):
    """Adds field observation, updates graph status, logs episodic memory, and re-ranks missions."""
    result = add_observation_and_sync(
        road_id=payload.road_id,
        reported_status=payload.reported_status,
        note=payload.note,
        source_id=payload.source_id,
        is_simulated=payload.is_simulated,
        override=payload.override
    )
    return result

@app.post("/api/demo/trigger-landslide", dependencies=[Depends(require_falkordb)])
def trigger_landslide_demo():
    """
    Simulates a major landslide on road_syabru_dhunche.
    Demonstrates live graph re-ranking where Syabrubesi loses verified access
    and alternate roads jump to the top of the Recon Queue.
    """
    result = add_observation_and_sync(
        road_id="road_syabru_dhunche",
        reported_status="blocked",
        note="Simulated massive debris flow blocks Syabrubesi-Dhunche main corridor",
        source_id="src_ndrrma",
        is_simulated=True,
        override=False,
        session_id="landslide_demo",
        agent_role="DisasterSimulator"
    )
    return {
        "event": "landslide_simulated",
        "affected_road": "road_syabru_dhunche",
        "result": result
    }

@app.post("/api/demo/reset", dependencies=[Depends(require_falkordb)])
def reset_graph_demo():
    """Resets FalkorDB graph to baseline seed state."""
    seed_falkordb()
    return {"status": "success", "message": "FalkorDB graph reset to baseline seed state."}

@app.post("/api/graph/wipe", response_model=GraphWipeResponse, dependencies=[Depends(require_falkordb)])
def wipe_graph_endpoint():
    """
    Executes a complete graph wipe on FalkorDB (MATCH (n) DETACH DELETE n),
    resetting the database to 0 nodes and 0 edges.
    """
    result = wipe_falkordb_graph()
    return GraphWipeResponse(
        status=result["status"],
        nodes_deleted=result["nodes_deleted"],
        is_empty=True,
        timestamp=result["timestamp"]
    )

@app.post("/api/graph/import-json", response_model=GraphImportResponse, dependencies=[Depends(require_falkordb)])
async def import_topology_json(
    request: Request,
    sample: bool = False
):
    """
    Imports disaster topology into FalkorDB.
    Supports:
    1. sample=True -> Loads server-bundled data/trishuli_flood_dataset.json
    2. Raw JSON body payload (from file drop/upload or direct API)
    """
    try:
        data_dict = None
        if sample:
            summary = import_dataset(clear_existing=True)
        else:
            body = await request.body()
            if body and body.strip():
                try:
                    data_dict = json.loads(body.decode("utf-8"))
                    summary = import_dataset(data_dict=data_dict, clear_existing=True)
                except json.JSONDecodeError as jde:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid JSON payload: {str(jde)}"
                    )
            else:
                summary = import_dataset(clear_existing=True)

        counts = summary.get("imported_counts", {})
        return GraphImportResponse(
            status="success",
            message=f"Successfully imported {counts.get('locations', 0)} locations, {counts.get('road_segments', 0)} road lifelines, and {counts.get('bridges', 0)} surveyed bridges into FalkorDB.",
            imported_counts=counts,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to import topology JSON into FalkorDB: {str(e)}"
        )

@app.get("/api/graph/sample-download")
def download_sample_dataset():
    """Serves the complete Trishuli flood dataset as an attachment for judges."""
    path = resolve_default_file()
    return FileResponse(
        path=str(path),
        media_type="application/json",
        filename="trishuli_flood_dataset.json"
    )

@app.get("/api/memory", response_model=list[EpisodicMemoryModel], dependencies=[Depends(require_falkordb)])
def get_memory_feed(limit: int = 50):
    """Returns episodic memory timeline stored in FalkorDB (Track 02)."""
    return get_episodic_memories(limit=limit)

@app.post("/api/agent/query", response_model=AgentQueryResponse, dependencies=[Depends(require_falkordb)])
def agent_query(payload: AgentQueryRequest):
    """Executes grounded multi-agent query over FalkorDB with guardrail checks."""
    return execute_agent_query(payload.prompt, session_id=payload.session_id or "ops_session")

# Mount PMTiles range-enabled endpoint
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
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
        )

    # Parse range: "bytes=start-end"
    try:
        range_val = range_header.replace("bytes=", "").strip()
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = (end - start) + 1

        with open(PMTILES_PATH, "rb") as f:
            f.seek(start)
            data = f.read(chunk_size)

        return Response(
            content=data,
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "ETag": f'"{file_size}"',
                "Cache-Control": "public, max-age=3600"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid byte range request: {str(e)}")

# Mount static frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
