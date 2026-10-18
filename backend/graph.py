from typing import Optional

from falkordb import FalkorDB, Graph
from fastapi import HTTPException

from backend.config import FALKORDB_GRAPH, FALKORDB_HOST, FALKORDB_PORT


class FalkorDBUnavailableError(HTTPException):
    def __init__(self, detail: str = "FalkorDB service unavailable at localhost:6379"):
        super().__init__(status_code=503, detail=detail)


_client: Optional[FalkorDB] = None


def get_falkor_client() -> FalkorDB:
    """Returns a singleton FalkorDB client, reconnecting if disconnected."""
    global _client
    try:
        if _client is None:
            _client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        _client.connection.ping()
        return _client
    except Exception as exc:
        _client = None
        raise FalkorDBUnavailableError(
            f"FalkorDB is unreachable at {FALKORDB_HOST}:{FALKORDB_PORT}. "
            f"Graph database service is required. Error: {str(exc)}"
        )


def get_graph(graph_name: str = FALKORDB_GRAPH) -> Graph:
    """Selects and returns the active graph instance from FalkorDB."""
    client = get_falkor_client()
    try:
        return client.select_graph(graph_name)
    except Exception as exc:
        raise FalkorDBUnavailableError(
            f"Failed to access graph '{graph_name}' on FalkorDB: {str(exc)}"
        )


def ping_falkordb() -> bool:
    """Verifies that FalkorDB is reachable and responsive."""
    try:
        client = get_falkor_client()
        graph = client.select_graph(FALKORDB_GRAPH)
        query_result = graph.query("RETURN 1")
        return bool(query_result.result_set)
    except Exception as exc:
        raise FalkorDBUnavailableError(
            f"FalkorDB health check failed on {FALKORDB_HOST}:{FALKORDB_PORT}: {str(exc)}"
        )
