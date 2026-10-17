from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

# Epistemic safety types
RoadStatus = Literal["verified_passable", "blocked", "unknown", "disputed"]
SettlementAccessState = Literal[
    "VERIFIED_ACCESS",
    "NEEDS_RECON",
    "ISOLATED_ON_MAP",
    "NO_MAPPED_ROUTE",
    "UNAVAILABLE",
    "UNKNOWN"
]

class SourceModel(BaseModel):
    id: str
    name: str
    url: str
    license: str

class ObservationModel(BaseModel):
    id: str
    road_id: str
    source_id: str
    source_name: Optional[str] = None
    status: RoadStatus
    note: str
    timestamp: str
    is_simulated: bool = False

class LocationModel(BaseModel):
    id: str
    name: str
    type: str  # settlement, facility_hub, junction
    district: str
    lat: float
    lon: float
    households: int
    access_state: Optional[SettlementAccessState] = None
    access_reason: Optional[str] = None
    verified_paths_count: int = 0
    potential_paths_count: int = 0

class FacilityModel(BaseModel):
    id: str
    name: str
    location_id: str
    type: str  # hospital, health_post
    importance_weight: float

class RoadSegmentModel(BaseModel):
    id: str
    name: str
    u: str
    v: str
    length_km: float
    is_bridge: bool
    status: RoadStatus
    observations: List[ObservationModel] = []
    coordinates: Optional[List[List[float]]] = None
    bridges: List[Dict[str, Any]] = []
    bridge_count: int = 0

class MissionModel(BaseModel):
    id: str
    road_id: str
    road_name: str
    priority_score: float
    reason: str
    sole_lifeline_count: int = 0
    total_dependent_count: int = 0
    status: str = "proposed"
    timestamp: str

class EpisodicMemoryModel(BaseModel):
    id: str
    session_id: str
    agent_role: str
    event_type: str
    summary: str
    road_id: Optional[str] = None
    timestamp: str
    metadata: Dict[str, Any] = {}

class PathInfo(BaseModel):
    locations: List[str]
    location_names: List[str]
    road_ids: List[str]
    road_names: List[str]
    total_km: float
    facility_id: str
    facility_name: str
    unverified_road_ids: List[str] = []

class SettlementAccessResponse(BaseModel):
    settlement_id: str
    settlement_name: str
    households: int
    district: str
    access_state: SettlementAccessState
    access_reason: str
    verified_paths: List[PathInfo]
    potential_paths: List[PathInfo]
    cypher_executed: str

class RoadEvidenceResponse(BaseModel):
    road: RoadSegmentModel
    endpoints: List[LocationModel]
    observations: List[ObservationModel]
    memories: List[EpisodicMemoryModel]
    affected_settlements: List[Dict[str, Any]]
    cypher_executed: str

class AddObservationRequest(BaseModel):
    road_id: str
    reported_status: RoadStatus
    note: str
    source_id: str = "src_scout"
    is_simulated: bool = False
    override: bool = False

class AgentQueryRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default_session"

class AgentQueryResponse(BaseModel):
    answer: str
    agent_role: str
    tool_called: Optional[str] = None
    cypher_executed: Optional[str] = None
    supporting_evidence: Dict[str, Any] = {}
    memory_id: Optional[str] = None

class KPISummary(BaseModel):
    verified_settlements: int = 0
    needs_recon_settlements: int = 0
    isolated_settlements: int = 0
    verified_roads: int = 0
    unknown_roads: int = 0
    blocked_roads: int = 0
    disputed_roads: int = 0
    active_missions: int = 0
    episodic_memories: int = 0

class NetworkResponse(BaseModel):
    locations: List[LocationModel]
    facilities: List[FacilityModel]
    road_segments: List[RoadSegmentModel]
    kpis: KPISummary
    timestamp: str
    is_empty: bool = False
    building_summary: Optional[Dict[str, Any]] = None
    bridge_summary: Optional[Dict[str, Any]] = None

class GraphWipeResponse(BaseModel):
    status: str
    nodes_deleted: int = 0
    is_empty: bool = True
    timestamp: str

class GraphImportResponse(BaseModel):
    status: str
    message: str
    imported_counts: Dict[str, Any]
    timestamp: str
