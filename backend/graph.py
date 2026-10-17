from typing import Optional
from fastapi import HTTPException
from falkordb import FalkorDB
import redis.exceptions
from backend.config import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_GRAPH

class FalkorDBUnavailableError(HTTPException):
    def __init__(self, detail: str = "FalkorDB service unavailable at localhost:6379"):
        super().__init__(status_code=503, detail=detail)

_client: Optional[FalkorDB] = None

def get_falkor_client() -> FalkorDB:
    global _client
    try:
        if _client is None:
            _client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        # Test connection
        _client.connection.ping()
        return _client
    except Exception as e:
        _client = None
        raise FalkorDBUnavailableError(
            f"FalkorDB is unreachable at {FALKORDB_HOST}:{FALKORDB_PORT}. Graph database service is required. Error: {str(e)}"
        )

def get_graph(graph_name: str = FALKORDB_GRAPH):
    client = get_falkor_client()
    try:
        return client.select_graph(graph_name)
    except Exception as e:
        raise FalkorDBUnavailableError(
            f"Failed to access graph '{graph_name}' on FalkorDB: {str(e)}"
        )

def ping_falkordb() -> bool:
    """Verifies that FalkorDB is reachable and responsive."""
    try:
        client = get_falkor_client()
        graph = client.select_graph(FALKORDB_GRAPH)
        res = graph.query("RETURN 1")
        return bool(res.result_set)
    except Exception as e:
        raise FalkorDBUnavailableError(
            f"FalkorDB health check failed on {FALKORDB_HOST}:{FALKORDB_PORT}: {str(e)}"
        )
