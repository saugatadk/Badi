const API_BASE = "";
const HEALTH_CHECK_INTERVAL_MS = 15000;
const MAP_RESIZE_DELAY_MS = 500;

// Application state
let networkState = null;
let currentMissions = [];
let selectedElement = null; // { type: 'road' | 'settlement' | 'bridge', id: string, data?: any }
let map = null;
let currentMarkers = [];
let bridgeMarkers = [];
let allBridgesData = [];
let currentBridgeFilter = "ALL";
let currentReconFilter = "ALL";

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupTabs();
  setupEventHandlers();
  loadAllData();

  window.addEventListener("resize", () => {
    if (map) map.resize();
  });
  setTimeout(() => {
    if (map) map.resize();
  }, MAP_RESIZE_DELAY_MS);

  setInterval(checkHealth, HEALTH_CHECK_INTERVAL_MS);
});

// Map initialization and layer configuration
function initMap() {
  if (typeof maplibregl === "undefined" || typeof pmtiles === "undefined") {
    console.warn("MapLibre or PMTiles library not ready yet, retrying in 100ms...");
    setTimeout(initMap, 100);
    return;
  }
  if (map) return; // Already initialized

  const mapContainer = document.getElementById("map");
  if (!mapContainer) return;

  try {
    const protocol = new pmtiles.Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);

    const origin = (typeof window !== "undefined" && window.location) ? window.location.origin : "http://localhost:8000";
    const pmtilesUrl = `pmtiles://${origin}/vendor/hot_flood_npl.pmtiles`;

    // MapLibre Map with pure Satellite basemap + HOT Flood OSM road & bridge network + FalkorDB Lifelines
    map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        sources: {
          satellite: {
            type: "raster",
            tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            tileSize: 256,
            attribution: "Esri Satellite Imagery"
          },
          hot_flood: {
            type: "vector",
            url: pmtilesUrl
          },
          "network-roads": {
            type: "geojson",
            data: { type: "FeatureCollection", features: [] }
          }
        },
        layers: [
          // 1. Satellite Imagery Basemap (Pure satellite, no dark mode or distracting vector tiles)
          {
            id: "satellite-base",
            type: "raster",
            source: "satellite"
          },
          // Background road geometry from HOT dataset
          {
            id: "roads-osm-line",
            type: "line",
            source: "hot_flood",
            "source-layer": "hot_flood_npl",
            filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "category"], "roads"]],
            paint: {
              "line-color": "#64748b",
              "line-width": 1.4,
              "line-opacity": 0.38
            }
          },
          // Bridge lines from HOT dataset
          {
            id: "bridges-osm-line",
            type: "line",
            source: "hot_flood",
            "source-layer": "hot_flood_npl",
            filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "category"], "bridges"]],
            paint: {
              "line-color": "#fbbf24",
              "line-width": 3.5,
              "line-opacity": 0.65
            }
          },
          // Damaged and flooded infrastructure overlays
          {
            id: "destroyed-features-line",
            type: "line",
            source: "hot_flood",
            "source-layer": "hot_flood_npl",
            filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "category"], "destroyed_features"]],
            paint: {
              "line-color": "#ef4444",
              "line-width": 3.5,
              "line-dasharray": [2, 2]
            }
          },
          // Selection halo positioned below casing
          {
            id: "network-roads-halo",
            type: "line",
            source: "network-roads",
            filter: ["==", ["get", "id"], ""],
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": "#ffffff",
              "line-width": [
                "case",
                ["==", ["get", "is_bridge"], true], 18.0,
                15.0
              ],
              "line-opacity": 0.85,
              "line-blur": 2.5
            }
          },
          // Selection edge highlight casing
          {
            id: "network-roads-selection-casing",
            type: "line",
            source: "network-roads",
            filter: ["==", ["get", "id"], ""],
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": "#ffffff",
              "line-width": [
                "case",
                ["==", ["get", "is_bridge"], true], 13.0,
                10.5
              ],
              "line-opacity": 1.0,
              "line-blur": 0
            }
          },
          // High-contrast dark casing for road lifelines
          {
            id: "network-roads-casing",
            type: "line",
            source: "network-roads",
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": "#020617",
              "line-width": [
                "case",
                ["==", ["get", "is_bridge"], true], 8.5,
                7.0
              ],
              "line-opacity": 0.95
            }
          },
          // Verified passable and blocked corridors
          {
            id: "network-roads-line",
            type: "line",
            source: "network-roads",
            filter: ["in", ["get", "status"], ["literal", ["verified_passable", "blocked"]]],
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": [
                "match",
                ["get", "status"],
                "verified_passable", "#10b981",
                "blocked", "#ef4444",
                "#64748b"
              ],
              "line-width": [
                "case",
                ["==", ["get", "is_bridge"], true], 5.2,
                4.0
              ]
            }
          },
          // Unknown and disputed corridors
          {
            id: "network-roads-dashed",
            type: "line",
            source: "network-roads",
            filter: ["in", ["get", "status"], ["literal", ["unknown", "disputed"]]],
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": [
                "match",
                ["get", "status"],
                "unknown", "#f59e0b",
                "disputed", "#ec4899",
                "#e2e8f0"
              ],
              "line-width": [
                "case",
                ["==", ["get", "is_bridge"], true], 5.2,
                4.0
              ],
              "line-dasharray": [2, 2]
            }
          }
        ]
      },
      center: [85.26, 28.08],
      zoom: 10
    });
    window.map = map;

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");
    map.addControl(new maplibregl.FullscreenControl(), "top-right");

    const interactiveRoadLayers = [
      "network-roads-line",
      "network-roads-dashed",
      "network-roads-casing",
      "network-roads-selection-casing",
      "network-roads-halo"
    ];

    // Click on FalkorDB Lifelines
    map.on("click", interactiveRoadLayers, (e) => {
      if (e.features && e.features.length > 0) {
        if (e.originalEvent) e.originalEvent._roadHandled = true;
        const roadId = e.features[0].properties.id;
        selectRoad(roadId);
      }
    });

    map.on("mouseenter", interactiveRoadLayers, () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", interactiveRoadLayers, () => {
      map.getCanvas().style.cursor = "";
    });

    // Click on real road or bridge feature to view attribute popup
    map.on("click", ["bridges-osm-line", "destroyed-features-line", "roads-osm-line"], (e) => {
      if (e.originalEvent && e.originalEvent._roadHandled) return;
      if (e.features && e.features.length > 0) {
        const feat = e.features[0];
        const props = feat.properties;
        const name = props.name || props.name_en || props.name_latin || props.name_ne || props.category || "OSM Survey Feature";
        const status = props.status || "Observed";
        const category = props.category || "road";
        const isBridge = category === "bridges" || (name && name.toLowerCase().includes("bridge"));

        new maplibregl.Popup({ offset: 15, className: "map-popup" })
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="font-family: var(--font-sans); padding: 4px;">
              <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; font-weight: 700; margin-bottom: 2px;">
                <i class="fa-solid fa-map-location-dot"></i> HOT OSM Survey
              </div>
              <strong style="font-size: 0.85rem; color: #f8fafc;">${name}</strong><br>
              <div style="margin-top: 4px; font-size: 0.72rem; color: #cbd5e1;">
                Category: <code>${category}</code> | Status: <b style="color:${status === 'Destroyed' ? '#fb7185' : '#34d399'}">${status}</b>
              </div>
              <button class="popup-inspect-btn" onclick="inspectNearestLifeline('${name.replace(/'/g, "\\'")}', ${e.lngLat.lng}, ${e.lngLat.lat})">
                <i class="fa-solid fa-route"></i> Inspect Nearest Lifeline
              </button>
            </div>
          `)
          .addTo(map);
      }
    });

    map.on("mouseenter", ["bridges-osm-line", "destroyed-features-line", "roads-osm-line"], () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", ["bridges-osm-line", "destroyed-features-line", "roads-osm-line"], () => {
      map.getCanvas().style.cursor = "";
    });

    // When map finishes loading, if networkState is already fetched, push roads
    map.on("load", () => {
      if (networkState && networkState.road_segments) {
        updateMapRoads(networkState.road_segments);
      }
    });

    // Staggered size invalidations
    setTimeout(() => { if (map) map.resize(); }, 150);
    setTimeout(() => { if (map) map.resize(); }, 600);
    setTimeout(() => { if (map) map.resize(); }, 1200);
  } catch (err) {
    console.error("Error initializing MapLibre map:", err);
  }
}

function renderMapNetwork(data) {
  if (!map) {
    initMap();
  }

  // Clear existing markers
  currentMarkers.forEach(m => m.remove());
  currentMarkers = [];

  window.allSettlementsData = data.locations;
  const locationsMap = {};
  data.locations.forEach(loc => {
    locationsMap[loc.id] = loc;
  });

  const facilitiesByLocation = {};
  (data.facilities || []).forEach(fac => {
    facilitiesByLocation[fac.location_id] = fac;
  });

  // Render disaster graph nodes pinned on coordinates
  data.locations.forEach(loc => {
    let markerClass = "marker-verified";
    if (loc.access_state === "NEEDS_RECON") {
      markerClass = "marker-recon";
    } else if (loc.access_state === "ISOLATED_ON_MAP") {
      markerClass = "marker-isolated";
    } else if (loc.type === "facility_hub") {
      markerClass = "marker-hub";
    } else if (loc.type === "junction") {
      markerClass = "marker-junction";
    }

    const facility = facilitiesByLocation[loc.id];
    let innerContent = loc.households ? `${loc.households}` : '<i class="fa-solid fa-location-dot"></i>';
    if (loc.type === "facility_hub" && !loc.households) {
      innerContent = '<i class="fa-solid fa-hospital"></i>';
    }

    const isJunction = loc.type === "junction";
    const el = document.createElement("div");
    el.className = `settlement-marker-wrap ${isJunction ? 'wrap-junction' : ''}`;
    el.setAttribute("data-location-id", loc.id);
    el.setAttribute("data-location-name", loc.name);
    el.innerHTML = `
      <div class="settlement-label">${loc.name}</div>
      <div class="settlement-marker ${markerClass}">
        <span>${innerContent}</span>
      </div>
      ${facility ? `
        <div class="settlement-facility-badge" title="${facility.name} (${facility.type})">
          <i class="fa-solid fa-plus"></i>
        </div>
      ` : ""}
    `;

    el.addEventListener("click", (e) => {
      e.stopPropagation();
      selectSettlement(loc.id);
    });

    // Center anchor locks node circle on junction
    const marker = new maplibregl.Marker({ element: el, anchor: "center" })
      .setLngLat([loc.lon, loc.lat])
      .addTo(map);

    currentMarkers.push(marker);
  });

  // FalkorDB road lifelines
  if (data.road_segments) {
    updateMapRoads(data.road_segments);
  }

  // Surveyed bridge damage markers
  if (allBridgesData.length === 0) {
    loadAndRenderBridges();
  }

  // Automatically fit bounds of active disaster network
  if (!map._hasFittedBounds && data.locations.length > 0) {
    try {
      const bounds = new maplibregl.LngLatBounds();
      data.locations.forEach(loc => {
        if (loc.lat && loc.lon) bounds.extend([loc.lon, loc.lat]);
      });
      map.fitBounds(bounds, { padding: 45, maxZoom: 12 });
      map._hasFittedBounds = true;
    } catch (e) {
      console.warn("Could not fit map bounds:", e);
    }
  }

  if (map) {
    map.resize();
  }
}

function updateMapRoads(roadSegments) {
  if (!map) return;
  const features = (roadSegments || [])
    .filter(r => r.coordinates && r.coordinates.length > 1)
    .map(r => ({
      type: "Feature",
      properties: {
        id: r.id,
        name: r.name,
        status: r.status,
        length_km: r.length_km,
        is_bridge: r.is_bridge,
        bridge_count: r.bridge_count || 0
      },
      geometry: {
        type: "LineString",
        coordinates: r.coordinates
      }
    }));

  const geojson = { type: "FeatureCollection", features };

  const pushRoadData = () => {
    try {
      const src = map.getSource("network-roads");
      if (src) {
        src.setData(geojson);
      }
    } catch (e) {
      console.warn("Could not set network-roads data:", e);
    }
  };

  if (map.isStyleLoaded() || map.loaded()) {
    pushRoadData();
  } else {
    map.once("load", pushRoadData);
    map.once("styledata", pushRoadData);
  }
}

async function loadAndRenderBridges() {
  try {
    const resp = await fetch(`${API_BASE}/api/bridges`);
    if (!resp.ok) return;
    const data = await resp.json();
    allBridgesData = data.features || [];
    renderBridgeMarkers(allBridgesData, currentBridgeFilter);
  } catch (err) {
    console.warn("Could not load bridges:", err);
  }
}

function renderBridgeMarkers(bridges, filterStatus = "ALL") {
  bridgeMarkers.forEach(m => m.remove());
  bridgeMarkers = [];

  bridges.forEach(f => {
    const p = f.properties;
    const coords = f.geometry.coordinates;
    const status = p.status || "Unknown";

    if (filterStatus !== "ALL" && status !== filterStatus) {
      return;
    }

    let statusClass = "bridge-washed-out";
    let icon = '<i class="fa-solid fa-water"></i>';
    if (status === "Damaged") {
      statusClass = "bridge-damaged";
      icon = '<i class="fa-solid fa-triangle-exclamation"></i>';
    } else if (status === "Intact") {
      statusClass = "bridge-intact";
      icon = '<i class="fa-solid fa-check"></i>';
    }

    const el = document.createElement("div");
    el.className = "bridge-marker-wrap";
    el.innerHTML = `
      <div class="bridge-label-hover">${p.name}</div>
      <div class="bridge-marker ${statusClass}" title="${p.name} (${status})">${icon}</div>
    `;

    el.addEventListener("click", (e) => {
      e.stopPropagation();
      selectBridge(f);

      new maplibregl.Popup({ offset: 15, className: "map-popup bridge-popup" })
        .setLngLat(coords)
        .setHTML(`
          <div style="font-family: var(--font-sans); padding: 4px;">
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; font-weight: 700; margin-bottom: 2px;">
              <i class="fa-solid fa-bridge"></i> Surveyed Bridge
            </div>
            <strong style="font-size: 0.85rem; color: #f8fafc;">${p.name}</strong><br>
            <div style="margin-top: 6px; font-size: 0.75rem; display: flex; gap: 6px; align-items: center;">
              <span>Status:</span>
              <span class="status-pill ${status === 'Washed out' ? 'blocked' : (status === 'Damaged' ? 'disputed' : 'verified_passable')}" style="padding: 2px 8px; font-size: 0.7rem;">
                ${status}
              </span>
            </div>
            ${p.location ? `<div style="margin-top: 4px; font-size: 0.72rem; color: #cbd5e1;">Location: <code>${p.location}</code></div>` : ''}
            <div style="margin-top: 2px; font-size: 0.72rem; color: #94a3b8;">District: <b>${p.district || 'Rasuwa'}</b></div>
            <button class="popup-inspect-btn" onclick="selectBridgeByName('${p.name.replace(/'/g, "\\'")}')">
              <i class="fa-solid fa-circle-info"></i> Inspect in Graph &amp; Cypher
            </button>
          </div>
        `)
        .addTo(map);
    });

    const marker = new maplibregl.Marker({ element: el, anchor: "center" })
      .setLngLat(coords)
      .addTo(map);

    bridgeMarkers.push(marker);
  });
}

function fitMapToNetwork() {
  if (!map || !networkState || !networkState.locations) return;
  const bounds = new maplibregl.LngLatBounds();
  networkState.locations.forEach(loc => {
    if (loc.lat && loc.lon) bounds.extend([loc.lon, loc.lat]);
  });
  map.fitBounds(bounds, { padding: 45, maxZoom: 12 });
}


// Data fetching and state synchronization
async function loadAllData() {
  try {
    await Promise.all([
      fetchNetwork(),
      fetchMissions(),
      fetchMemories(),
      checkHealth()
    ]);
    if (networkState && networkState.kpis) {
      updateKPIS(networkState.kpis);
    }
  } catch (err) {
    console.error("Error loading network data:", err);
    showToast("FalkorDB synchronization error", "error");
  }
}

function showZeroState() {
  const overlay = document.getElementById("zero-state-overlay");
  const mapPanel = document.getElementById("map-panel");
  const sideDock = document.getElementById("side-dock");
  if (overlay) overlay.classList.remove("hidden");
  if (mapPanel) mapPanel.classList.add("hidden");
  if (sideDock) sideDock.classList.add("hidden");

  // Zero out the glance capsule counters in header
  const bridgeVal = document.getElementById("val-hot-bridges");
  if (bridgeVal) bridgeVal.innerText = "0";
  const bldgVal = document.getElementById("val-hot-buildings");
  if (bldgVal) bldgVal.innerText = "0";
  const missionVal = document.getElementById("val-active-missions");
  if (missionVal) missionVal.innerText = "0";
  const passVal = document.getElementById("val-verified-roads");
  if (passVal) passVal.innerText = "0";
  const blockVal = document.getElementById("val-blocked-roads");
  if (blockVal) blockVal.innerText = "0";

  const dbStatusText = document.getElementById("db-status-text");
  if (dbStatusText) dbStatusText.innerText = "Connected (0 Nodes)";
}

function hideZeroState() {
  const overlay = document.getElementById("zero-state-overlay");
  const mapPanel = document.getElementById("map-panel");
  const sideDock = document.getElementById("side-dock");
  if (overlay) overlay.classList.add("hidden");
  if (mapPanel) mapPanel.classList.remove("hidden");
  if (sideDock) sideDock.classList.remove("hidden");

  setTimeout(() => {
    if (map) map.resize();
  }, 120);
}

async function fetchNetwork() {
  const resp = await fetch(`${API_BASE}/api/network`);
  if (!resp.ok) {
    if (resp.status === 503) setDbStatus(false);
    throw new Error(`HTTP ${resp.status}`);
  }
  setDbStatus(true);
  networkState = await resp.json();
  if (networkState.is_empty) {
    showZeroState();
  } else {
    hideZeroState();
    updateKPIS(networkState.kpis);
    renderMapNetwork(networkState);
  }
}

async function fetchMissions() {
  const resp = await fetch(`${API_BASE}/api/missions`);
  if (!resp.ok) return;
  currentMissions = await resp.json();
  renderMissions(currentMissions);
  const badge = document.getElementById("missions-count-badge");
  if (badge) badge.innerText = currentMissions.length;
}

async function fetchMemories() {
  const resp = await fetch(`${API_BASE}/api/memory?limit=30`);
  if (!resp.ok) return;
  const memories = await resp.json();
  renderMemories(memories);
}

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    if (resp.ok) {
      setDbStatus(true);
    } else {
      setDbStatus(false);
    }
  } catch (e) {
    setDbStatus(false);
  }
}

function setDbStatus(isOnline) {
  const badge = document.getElementById("db-status");
  const text = document.getElementById("db-status-text");
  if (!badge || !text) return;
  if (isOnline) {
    badge.className = "db-status-badge status-connected";
    text.innerText = "Connected";
  } else {
    badge.className = "db-status-badge status-disconnected";
    text.innerText = "Offline";
  }
}

function updateKPIS(kpi) {
  if (!kpi) return;
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.innerText = val;
  };

  setEl("val-verified-settlements", kpi.verified_settlements ?? "--");
  setEl("val-recon-settlements", kpi.needs_recon_settlements ?? "--");
  setEl("val-isolated-settlements", kpi.isolated_settlements ?? "--");
  setEl("val-verified-roads", kpi.verified_roads ?? "--");
  setEl("val-unknown-roads", (kpi.unknown_roads ?? 0) + (kpi.disputed_roads ? ` (+${kpi.disputed_roads})` : ""));
  setEl("val-blocked-roads", kpi.blocked_roads ?? "--");
  setEl("val-active-missions", kpi.active_missions ?? "--");
  setEl("val-episodic-memories", kpi.episodic_memories ?? "--");

  // HOT Flood Survey Metrics (FalkorDB Grounded)
  if (networkState && networkState.bridge_summary) {
    const bs = networkState.bridge_summary;
    const washedCount = bs["Washed out"] !== undefined ? bs["Washed out"] : 0;
    setEl("val-hot-bridges", washedCount);
  }
  if (networkState && networkState.building_summary) {
    const bldgDestroyed = networkState.building_summary.by_status?.Destroyed !== undefined
      ? networkState.building_summary.by_status.Destroyed
      : 0;
    setEl("val-hot-buildings", bldgDestroyed.toLocaleString());
  }
}

// Mission queue rendering
function renderMissions(missions) {
  const container = document.getElementById("missions-container");
  const countBadge = document.getElementById("missions-count-badge");
  if (countBadge) countBadge.innerText = missions ? missions.length : 0;
  const launcherCount = document.getElementById("launcher-count");
  if (launcherCount) launcherCount.innerText = missions ? missions.length : 0;

  // Update filter counters
  const totalCount = missions ? missions.length : 0;
  const lifelineCount = missions ? missions.filter(m => m.sole_lifeline_count > 0).length : 0;
  const criticalCount = missions ? missions.filter(m => m.priority_score >= 80).length : 0;

  const elAll = document.getElementById("count-recon-all");
  if (elAll) elAll.innerText = totalCount;
  const elLifeline = document.getElementById("count-recon-lifelines");
  if (elLifeline) elLifeline.innerText = lifelineCount;
  const elCritical = document.getElementById("count-recon-critical");
  if (elCritical) elCritical.innerText = criticalCount;

  if (!missions || missions.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i class="fa-solid fa-circle-check empty-icon text-emerald"></i>
        <h4>No Recon Bottlenecks</h4>
        <p>All road lifelines currently have verified access status. No ground reconnaissance needed.</p>
      </div>
    `;
    return;
  }

  // Filter based on currentReconFilter
  let filtered = missions;
  if (currentReconFilter === "LIFELINE") {
    filtered = missions.filter(m => m.sole_lifeline_count > 0);
  } else if (currentReconFilter === "CRITICAL") {
    filtered = missions.filter(m => m.priority_score >= 80);
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: 2.5rem 1rem;">
        <i class="fa-solid fa-filter empty-icon"></i>
        <h4>No Matching Missions</h4>
        <p>No recon targets match the selected filter criteria.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = "";
  filtered.forEach((m, idx) => {
    let pillClass = "priority-medium";
    if (m.priority_score >= 80) pillClass = "priority-critical";
    else if (m.priority_score >= 50) pillClass = "priority-high";

    const isSelected = selectedElement && selectedElement.type === "road" && selectedElement.id === m.road_id;
    const card = document.createElement("div");
    card.className = `mission-card ${isSelected ? "active-mission" : ""}`;
    card.setAttribute("data-road-id", m.road_id);
    card.innerHTML = `
      <div class="mission-card-top">
        <div class="mission-road-title">
          <span class="mission-rank-badge">#${idx + 1}</span>${m.road_name}
        </div>
        <span class="priority-pill ${pillClass}">Priority ${m.priority_score}</span>
      </div>

      ${m.sole_lifeline_count > 0 ? `
        <div class="lifeline-alert-row">
          <span class="lifeline-badge">
            <i class="fa-solid fa-triangle-exclamation"></i> Sole Remaining Lifeline (${m.sole_lifeline_count} Settlement${m.sole_lifeline_count > 1 ? 's' : ''})
          </span>
        </div>
      ` : ""}

      <div class="mission-reason">${m.reason}</div>

      <div class="mission-actions">
        <button class="btn-inspect-mission" onclick="selectRoad('${m.road_id}')">
          <i class="fa-solid fa-clipboard-check"></i> Inspect &amp; Record Field Observation
        </button>
      </div>
    `;

    card.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      selectRoad(m.road_id);
    });

    container.appendChild(card);
  });
}

// Inspector and evidence chain
function inspectNearestLifeline(featureName, lng, lat) {
  if (!networkState || !networkState.road_segments || networkState.road_segments.length === 0) {
    showToast("Road network data loading...", "info");
    return;
  }
  let nearestRoad = null;
  let minDistance = Infinity;
  networkState.road_segments.forEach(r => {
    if (r.coordinates && r.coordinates.length > 0) {
      r.coordinates.forEach(pt => {
        const d = Math.hypot(pt[0] - lng, pt[1] - lat);
        if (d < minDistance) {
          minDistance = d;
          nearestRoad = r;
        }
      });
    }
  });

  if (nearestRoad) {
    showToast(`Mapped ${featureName} to ${nearestRoad.name}`, "info");
    selectRoad(nearestRoad.id);
  } else {
    showToast(`No adjacent FalkorDB lifeline found for ${featureName}`, "warning");
  }
}

const SELECTION_OUTLINE_LAYERS = [
  "network-roads-halo",
  "network-roads-selection-casing"
];

function makeRoadIdFilter(roadIds) {
  const ids = Array.isArray(roadIds) ? roadIds.filter(Boolean) : (roadIds ? [roadIds] : []);
  if (ids.length === 0) {
    return ["==", ["get", "id"], ""];
  }
  if (ids.length === 1) {
    return ["==", ["get", "id"], ids[0]];
  }
  return ["match", ["get", "id"], ids, true, false];
}

function applyRoadFilter(roadIds) {
  if (!map) return;
  const idFilter = makeRoadIdFilter(roadIds);

  // Set white glow and crisp white edge highlight framing the road
  SELECTION_OUTLINE_LAYERS.forEach(layerId => {
    if (map.getLayer(layerId)) {
      map.setFilter(layerId, idFilter);
    }
  });
}

function clearRoadHighlight() {
  if (!map) return;
  applyRoadFilter([]);
  document.querySelectorAll(".mission-card").forEach(c => c.classList.remove("active-mission"));
  document.querySelectorAll(".settlement-marker").forEach(m => m.classList.remove("marker-path-focus"));
}

async function selectRoad(roadId) {
  selectedElement = { type: "road", id: roadId };
  openSideDock();
  switchTab("tab-inspector");

  // Update active mission card highlight in Recon tab
  document.querySelectorAll(".mission-card").forEach(c => {
    c.classList.toggle("active-mission", c.getAttribute("data-road-id") === roadId);
  });

  // Highlight road on map: white edge outline framing its original color
  applyRoadFilter(roadId);

  if (networkState && networkState.road_segments) {
    const road = networkState.road_segments.find(r => r.id === roadId);
    if (road && road.coordinates && road.coordinates.length > 0) {
      const bounds = new maplibregl.LngLatBounds();
      road.coordinates.forEach(c => bounds.extend(c));
      map.fitBounds(bounds, { padding: 80, maxZoom: 13, duration: 800 });
    }
  }

  try {
    const resp = await fetch(`${API_BASE}/api/roads/${roadId}`);
    if (!resp.ok) throw new Error("Failed to load road details");
    const data = await resp.json();
    renderRoadInspector(data);
    if (data.endpoints && data.endpoints.length > 0) {
      highlightSettlementMarkers(data.endpoints.map(e => e.id || e.name));
    }
  } catch (err) {
    showToast(`Error inspecting road ${roadId}`, "error");
  }
}

// Map corridor focusing and highlight overlays
function focusCorridorOnMap(roadIds, options = {}) {
  if (!map) return;
  const ids = Array.isArray(roadIds) ? roadIds.filter(Boolean) : (roadIds ? [roadIds] : []);

  // Highlight road segments with white edge outline preserving original colors
  applyRoadFilter(ids);

  // Compute bounding box across road coordinates and locations
  const bounds = new maplibregl.LngLatBounds();
  let hasCoords = false;

  if (networkState && networkState.road_segments) {
    networkState.road_segments.forEach(r => {
      if (ids.includes(r.id) && r.coordinates && r.coordinates.length > 0) {
        r.coordinates.forEach(c => {
          bounds.extend(c);
          hasCoords = true;
        });
      }
    });
  }

  const locTargets = options.locations || [];
  if (locTargets.length > 0 && networkState && networkState.locations) {
    networkState.locations.forEach(loc => {
      if (locTargets.includes(loc.id) || locTargets.includes(loc.name)) {
        bounds.extend([loc.lon, loc.lat]);
        hasCoords = true;
      }
    });
  }

  // Fit bounds taking into account open side dock margin
  if (hasCoords) {
    const isDockCollapsed = document.getElementById("main-content")?.classList.contains("dock-collapsed");
    const rightPadding = isDockCollapsed ? 80 : 380;

    map.fitBounds(bounds, {
      padding: { top: 90, bottom: 90, left: 90, right: rightPadding },
      maxZoom: options.maxZoom || 13.5,
      duration: options.duration || 1000
    });
  }

  // Highlight endpoint markers with pulse effect
  if (options.settlements && options.settlements.length > 0) {
    highlightSettlementMarkers(options.settlements);
  }
}

function highlightSettlementMarkers(locationIds) {
  const ids = Array.isArray(locationIds) ? locationIds : [locationIds];
  document.querySelectorAll(".settlement-marker-wrap").forEach(wrap => {
    const locId = wrap.getAttribute("data-location-id");
    const locName = wrap.getAttribute("data-location-name");
    const marker = wrap.querySelector(".settlement-marker");
    if (!marker) return;
    if (ids.includes(locId) || ids.includes(locName)) {
      marker.classList.add("marker-path-focus");
      setTimeout(() => marker.classList.remove("marker-path-focus"), 9000);
    } else {
      marker.classList.remove("marker-path-focus");
    }
  });
}

function focusMapFromAgentResponse(data) {
  if (!data || !map) return;
  const ev = data.supporting_evidence || {};

  // 0. Point-to-Point Corridor Routing (find_route_between)
  const route = ev.primary_route || (ev.routes && ev.routes[0]);
  if (route && route.road_ids && route.road_ids.length > 0) {
    const roadIds = route.road_ids || [];
    const locs = (route.locations || []).map(l => (typeof l === 'object' ? (l.id || l.name) : l));
    focusCorridorOnMap(roadIds, {
      locations: locs,
      settlements: locs,
      maxZoom: 13.5
    });
    return;
  }

  // 1. Hospital / Settlement Route Paths
  if (ev.verified_paths && ev.verified_paths.length > 0) {
    const p = ev.verified_paths[0];
    const roadIds = p.road_ids || [];
    const locs = p.locations || [];
    focusCorridorOnMap(roadIds, {
      locations: locs,
      settlements: locs,
      maxZoom: 13.5
    });
    return;
  }
  if (ev.potential_paths && ev.potential_paths.length > 0) {
    const p = ev.potential_paths[0];
    const roadIds = p.road_ids || [];
    const locs = p.locations || [];
    focusCorridorOnMap(roadIds, {
      locations: locs,
      settlements: locs,
      maxZoom: 13.5
    });
    return;
  }

  // 2. Blast Radius / Road Impact
  const rid = ev.road_id || (ev.road && ev.road.id);
  if (rid) {
    const affLocs = (ev.affected_settlements || []).map(s => s.id || s.name);
    focusCorridorOnMap([rid], {
      locations: affLocs,
      settlements: affLocs,
      maxZoom: 13.0
    });
    return;
  }

  // 3. Settlement specific query without paths
  if (ev.settlement_id) {
    focusCorridorOnMap([], {
      locations: [ev.settlement_id],
      settlements: [ev.settlement_id],
      maxZoom: 14.0
    });
    return;
  }

  // 4. Bridge damage overview
  if (data.tool_called && (data.tool_called.includes("bridge") || data.tool_called.includes("Bridge"))) {
    map.flyTo({
      center: [85.27, 28.12],
      zoom: 11.2,
      duration: 1000
    });
  }
}

window.focusCorridorOnMap = focusCorridorOnMap;
window.highlightSettlementMarkers = highlightSettlementMarkers;


function selectBridge(bridgeFeature) {
  if (!bridgeFeature) return;
  clearRoadHighlight();
  const p = bridgeFeature.properties || {};
  const coords = (bridgeFeature.geometry && bridgeFeature.geometry.coordinates) ? bridgeFeature.geometry.coordinates : null;
  const name = p.name || "Surveyed Bridge";

  selectedElement = { type: "bridge", id: name, data: bridgeFeature };
  openSideDock();
  switchTab("tab-inspector");

  if (coords && map) {
    map.flyTo({ center: coords, zoom: 13.5, duration: 800 });
  }

  renderBridgeInspector(bridgeFeature);
}

function selectBridgeByName(bridgeName) {
  const found = allBridgesData.find(b => b.properties && b.properties.name === bridgeName);
  if (found) {
    selectBridge(found);
  } else {
    selectBridge({
      properties: { name: bridgeName, status: "Washed out", district: "Rasuwa" },
      geometry: { coordinates: [85.25, 28.1] }
    });
  }
}

function renderBridgeInspector(bridgeFeature) {
  const p = bridgeFeature.properties || {};
  const coords = (bridgeFeature.geometry && bridgeFeature.geometry.coordinates) ? bridgeFeature.geometry.coordinates : [85.25, 28.1];
  const name = p.name || "Surveyed Bridge";
  const status = p.status || "Washed out";
  const district = p.district || "Rasuwa";

  // Hide empty state and road details, show bridge details
  const emptyState = document.getElementById("inspector-empty-state");
  if (emptyState) emptyState.classList.add("hidden");
  const roadDetails = document.getElementById("inspector-details");
  if (roadDetails) roadDetails.classList.add("hidden");
  const bridgeDetails = document.getElementById("inspector-bridge-details");
  if (!bridgeDetails) return;
  bridgeDetails.classList.remove("hidden");

  // Populate Header
  document.getElementById("bridge-title").innerText = name;
  const statusBadge = document.getElementById("bridge-status-badge");
  statusBadge.className = `status-pill ${status === 'Washed out' ? 'blocked' : (status === 'Damaged' ? 'unknown' : 'verified')}`;
  statusBadge.innerText = status;

  // Find nearest / associated FalkorDB lifeline road
  let nearestRoad = null;
  let minDistance = Infinity;
  if (networkState && networkState.road_segments) {
    for (const r of networkState.road_segments) {
      if (r.bridges && r.bridges.some(b => b.name === name || name.includes(b.name))) {
        nearestRoad = r;
        break;
      }
    }
    if (!nearestRoad) {
      for (const r of networkState.road_segments) {
        if (r.coordinates && r.coordinates.length > 0) {
          for (const pt of r.coordinates) {
            const d = Math.hypot(pt[0] - coords[0], pt[1] - coords[1]);
            if (d < minDistance) {
              minDistance = d;
              nearestRoad = r;
            }
          }
        }
      }
    }
  }

  // Populate Metrics Grid
  document.getElementById("bridge-metrics-grid").innerHTML = `
    <div class="metric-box">
      <div class="m-label">Surveyed Status</div>
      <div class="m-value ${status === 'Washed out' ? 'text-rose' : (status === 'Damaged' ? 'text-amber' : 'text-emerald')}">${status}</div>
    </div>
    <div class="metric-box">
      <div class="m-label">District</div>
      <div class="m-value">${district}</div>
    </div>
    <div class="metric-box">
      <div class="m-label">Coordinates</div>
      <div class="m-value" style="font-size: 0.78rem;">${coords[1].toFixed(4)}°N, ${coords[0].toFixed(4)}°E</div>
    </div>
    <div class="metric-box">
      <div class="m-label">Structure Type</div>
      <div class="m-value" style="font-size: 0.85rem;">${p.type || "Critical River Crossing"}</div>
    </div>
  `;

  // Populate Associated Lifeline Card
  const lifelineCard = document.getElementById("bridge-lifeline-card");
  if (nearestRoad) {
    lifelineCard.innerHTML = `
      <div style="font-size: 0.85rem; font-weight: 700; color: #fff;">
        <i class="fa-solid fa-road"></i> ${nearestRoad.name}
      </div>
      <div style="font-size: 0.75rem; color: #94a3b8; display: flex; gap: 8px; align-items: center; margin-top: 2px;">
        <span>Segment Status:</span>
        <span class="status-pill ${nearestRoad.status}">${nearestRoad.status}</span>
        <span>(${nearestRoad.length_km} km)</span>
      </div>
      <div style="font-size: 0.73rem; color: #cbd5e1; margin-top: 4px;">
        ${nearestRoad.status === 'blocked' ? 'Critical bottleneck: Traffic cut off along this valley corridor due to structural failure.' : 'Monitored bypass lifeline corridor in active use.'}
      </div>
      <button class="btn-inspect-lifeline" style="margin-top: 6px;" onclick="selectRoad('${nearestRoad.id}')">
        <i class="fa-solid fa-arrow-right"></i> Inspect Lifeline Corridor (${nearestRoad.id})
      </button>
    `;
  } else {
    lifelineCard.innerHTML = `<span class="empty-tag">No direct lifeline road mapped</span>`;
  }

  // Cypher executed
  const cypher = `// Live Multi-hop Query for Surveyed Bridge\nMATCH (b:Bridge {name: "${name}"})\nOPTIONAL MATCH (b)-[r:ON_SEGMENT]->(rd:Road)\nOPTIONAL MATCH (rd)-[:LEADS_TO]->(s:Settlement)\nRETURN b.name, b.status, rd.name, rd.status, collect(s.name) AS dependents`;
  const codeBox = document.getElementById("bridge-cypher-code");
  if (codeBox) codeBox.innerText = cypher;
}

async function selectSettlement(settlementId) {
  clearRoadHighlight();
  selectedElement = { type: "settlement", id: settlementId };
  openSideDock();
  switchTab("tab-inspector");

  const bridgeDetails = document.getElementById("inspector-bridge-details");
  if (bridgeDetails) bridgeDetails.classList.add("hidden");

  try {
    const resp = await fetch(`${API_BASE}/api/settlements/${settlementId}/access`);
    if (!resp.ok) throw new Error("Failed to load settlement details");
    const data = await resp.json();
    renderSettlementInspector(data);
  } catch (err) {
    showToast(`Error inspecting settlement ${settlementId}`, "error");
  }
}

function renderRoadInspector(data) {
  document.getElementById("inspector-empty-state").classList.add("hidden");
  const bridgeDetails = document.getElementById("inspector-bridge-details");
  if (bridgeDetails) bridgeDetails.classList.add("hidden");
  const details = document.getElementById("inspector-details");
  details.classList.remove("hidden");

  const road = data.road;
  document.getElementById("insp-type-badge").innerText = road.is_bridge ? "BRIDGE SEGMENT" : "ROAD SEGMENT";
  document.getElementById("insp-title").innerText = road.name;

  const statusBadge = document.getElementById("insp-status-badge");
  statusBadge.className = `status-pill ${road.status}`;
  statusBadge.innerText = road.status;

  // Metrics
  const epNames = data.endpoints.map(e => e.name).join(" ↔ ");
  document.getElementById("insp-metrics-grid").innerHTML = `
    <div class="metric-box">
      <div class="m-label">Length / Span</div>
      <div class="m-value">${road.length_km} km</div>
    </div>
    <div class="metric-box">
      <div class="m-label">Structure Type</div>
      <div class="m-value">${road.is_bridge ? 'Critical Bridge' : 'Highway / Road'}</div>
    </div>
    <div class="metric-box" style="grid-column: span 2;">
      <div class="m-label">Endpoints</div>
      <div class="m-value" style="font-size: 0.8rem;">${epNames}</div>
    </div>
  `;

  // Affected Settlements
  const affList = document.getElementById("insp-affected-list");
  if (data.affected_settlements && data.affected_settlements.length > 0) {
    affList.innerHTML = data.affected_settlements
      .map(s => `<span class="settlement-tag">${s.name} (${s.households} hh)</span>`)
      .join("");
  } else {
    affList.innerHTML = `<span class="empty-tag">No downstream settlements depend on this segment</span>`;
  }

  // Surveyed Bridges along Segment
  const bridgesListEl = document.getElementById("insp-bridges-list");
  if (bridgesListEl) {
    if (road.bridges && road.bridges.length > 0) {
      bridgesListEl.innerHTML = road.bridges.map(b => `
        <div class="bridge-tag-item ${b.status === 'Washed out' ? 'tag-washed' : (b.status === 'Damaged' ? 'tag-damaged' : 'tag-intact')}">
          <span class="bridge-name"><i class="fa-solid fa-bridge"></i> <strong>${b.name}</strong></span>
          <span class="bridge-status-badge">${b.status}</span>
        </div>
      `).join("");
    } else {
      bridgesListEl.innerHTML = `<span class="empty-tag">No individual bridge structures surveyed on this segment</span>`;
    }
  }

  // Observations Timeline
  const obsList = document.getElementById("insp-obs-list");
  if (data.observations && data.observations.length > 0) {
    obsList.innerHTML = data.observations.map(o => `
      <div class="obs-item ${o.status}">
        <div class="obs-meta">
          <span><strong>${o.source_name || o.source_id}</strong> (${o.status})</span>
          <span>${o.timestamp ? o.timestamp.substring(11, 19) + ' UTC' : ''}</span>
        </div>
        <div>${o.note}</div>
      </div>
    `).join("");
  } else {
    obsList.innerHTML = `<span class="empty-tag">No field observations recorded in FalkorDB yet</span>`;
  }

  // Form setup
  document.getElementById("obs-road-id").value = road.id;
  document.getElementById("obs-status").value = road.status === "blocked" ? "verified_passable" : "blocked";

  // Cypher
  document.getElementById("insp-cypher-code").innerText = data.cypher_executed;
}

function renderSettlementInspector(data) {
  document.getElementById("inspector-empty-state").classList.add("hidden");
  const details = document.getElementById("inspector-details");
  details.classList.remove("hidden");

  document.getElementById("insp-type-badge").innerText = "SETTLEMENT NODE";
  document.getElementById("insp-title").innerText = `${data.settlement_name} (${data.district})`;

  const statusBadge = document.getElementById("insp-status-badge");
  let badgeClass = "verified";
  if (data.access_state === "NEEDS_RECON") badgeClass = "unknown";
  if (data.access_state === "ISOLATED_ON_MAP") badgeClass = "blocked";

  statusBadge.className = `status-pill ${badgeClass}`;
  statusBadge.innerText = data.access_state === "NEEDS_RECON" ? "NEEDS VERIFICATION" : data.access_state.replace(/_/g, " ");

  // Metrics
  document.getElementById("insp-metrics-grid").innerHTML = `
    <div class="metric-box">
      <div class="m-label">Population</div>
      <div class="m-value">${data.households} Households</div>
    </div>
    <div class="metric-box">
      <div class="m-label">Verified Paths</div>
      <div class="m-value text-emerald">${data.verified_paths.length}</div>
    </div>
    <div class="metric-box" style="grid-column: span 2;">
      <div class="m-label">Access Assessment</div>
      <div class="m-value" style="font-size: 0.78rem; font-weight: 500;">${data.access_reason}</div>
    </div>
  `;

  // Paths list in affected settlements container
  const affList = document.getElementById("insp-affected-list");
  if (data.verified_paths.length > 0) {
    const p = data.verified_paths[0];
    affList.innerHTML = `
      <div style="font-size: 0.78rem; color: #34d399; width: 100%;">
        <strong>✓ Primary Route:</strong> ${p.location_names.join(" ➔ ")} (${p.total_km} km to ${p.facility_name})
      </div>
    `;
  } else if (data.potential_paths.length > 0) {
    const p = data.potential_paths[0];
    affList.innerHTML = `
      <div style="font-size: 0.78rem; color: #fbbf24; width: 100%;">
        <strong>Recon Path:</strong> ${p.location_names.join(" ➔ ")} (Needs recon on: ${p.unverified_road_ids.join(", ")})
      </div>
    `;
  } else {
    affList.innerHTML = `<span class="empty-tag text-rose">All topological paths severed by flood washouts</span>`;
  }

  // Clear observations list for settlement
  document.getElementById("insp-obs-list").innerHTML = `
    <span class="empty-tag">Settlement access is dynamically evaluated from edge observations.</span>
  `;

  document.getElementById("insp-cypher-code").innerText = data.cypher_executed;
}

// Episodic memory timeline
function renderMemories(memories) {
  const container = document.getElementById("memory-timeline-container");
  if (!memories || memories.length === 0) {
    container.innerHTML = `<div class="empty-state">No episodic memories recorded yet.</div>`;
    return;
  }

  container.innerHTML = memories.map(m => `
    <div class="memory-node">
      <div class="memory-node-header">
        <span class="memory-role"><i class="fa-solid fa-robot"></i> ${m.agent_role}</span>
        <span class="memory-timestamp">${m.timestamp ? m.timestamp.substring(11, 19) + ' UTC' : ''}</span>
      </div>
      <div class="memory-summary">${m.summary}</div>
      ${m.road_id ? `<div style="font-size: 0.65rem; color: #60a5fa;">Referenced Road: <code>${m.road_id}</code></div>` : ''}
    </div>
  `).join("");
}

// Agent chat workflows
async function sendAgentQuery(promptText) {
  if (!promptText || !promptText.trim()) return;

  const chatContainer = document.getElementById("chat-messages");

  // Append User message
  const userBubble = document.createElement("div");
  userBubble.className = "chat-bubble user-bubble selectable-text";
  userBubble.innerHTML = `
    <div class="bubble-header"><span>Operator</span></div>
    <div class="selectable-text">${escapeHtml(promptText)}</div>
  `;
  chatContainer.appendChild(userBubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Append Loading placeholder
  const loadingBubble = document.createElement("div");
  loadingBubble.className = "chat-bubble agent-bubble selectable-text";
  loadingBubble.innerHTML = `
    <div class="bubble-header"><span class="agent-name">Badi Agent</span></div>
    <div><i class="fa-solid fa-spinner fa-spin"></i> Querying FalkorDB graph topology and memory...</div>
  `;
  chatContainer.appendChild(loadingBubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  try {
    const resp = await fetch(`${API_BASE}/api/agent/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText, session_id: "console_session" })
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    const isOutOfScope = data.tool_called === 'none_out_of_scope' || 
                         data.agent_role === 'ScopeEnforcer' || 
                         (data.supporting_evidence && data.supporting_evidence.status === 'out_of_scope');
    const isBoundary = data.tool_called === 'none_boundary_enforced' || 
                       data.agent_role === 'GuardrailEnforcer';

    const statusBadge = isOutOfScope
      ? `<span class="timestamp" style="background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); font-weight: 600;">Out of Scope</span>`
      : isBoundary
        ? `<span class="timestamp" style="background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); font-weight: 600;">Boundary Enforced</span>`
        : `<span class="timestamp">Grounded in FalkorDB</span>`;

    const pathBadge = isOutOfScope
      ? `<span class="path-badge" style="background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.35);">Scope Restricted</span>`
      : isBoundary
        ? `<span class="path-badge" style="background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35);">Epistemic Restriction</span>`
        : `<span class="path-badge">FalkorDB Verified</span>`;

    const pathTitle = isOutOfScope
      ? `<span>Show Scope Evaluation & Graph Audit</span>`
      : isBoundary
        ? `<span>Show Boundary Evaluation & Audit</span>`
        : `<span>Show Graph Path & Cypher Behind Answer</span>`;

    const pathIcon = (isOutOfScope || isBoundary) ? `fa-solid fa-ban` : `fa-solid fa-diagram-project`;
    const pathIconColor = isOutOfScope ? `#f87171` : (isBoundary ? `#fbbf24` : `var(--sky-main)`);

    loadingBubble.innerHTML = `
      <div class="bubble-header">
        <span class="agent-name"><i class="fa-solid fa-shield-halved"></i> ${escapeHtml(data.agent_role)}</span>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          ${statusBadge}
          <button class="copy-code-btn" title="Copy answer" onclick="copyBubbleAnswer(this)"><i class="fa-regular fa-copy"></i> Copy</button>
        </div>
      </div>
      <div class="bubble-body selectable-text">${formatMarkdown(data.answer)}</div>
      ${data.cypher_executed ? `
      <details class="bubble-path-details" ontoggle="if(this.open){ setTimeout(() => this.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 60); }">
        <summary>
          <div class="path-summary-left">
            <i class="fa-solid fa-chevron-right path-chevron"></i>
            <i class="${pathIcon}" style="color: ${pathIconColor};"></i>
            ${pathTitle}
          </div>
          ${pathBadge}
        </summary>
        <div class="bubble-path-body">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.72rem; color: var(--text-muted); border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.35rem;">
            <div><strong>Tool Called:</strong> <code>${escapeHtml(data.tool_called || 'none')}</code></div>
            ${data.memory_id ? `<div><strong>Episodic Memory:</strong> <code>${escapeHtml(data.memory_id)}</code></div>` : ''}
          </div>
          ${renderPathBreadcrumbs(data.supporting_evidence)}
          <div class="path-section-header">
            <span class="path-title"><i class="fa-solid fa-terminal"></i> Live Cypher Query Executed:</span>
            <button class="copy-code-btn" onclick="copyCodeSnippet(this)"><i class="fa-regular fa-copy"></i> Copy Cypher</button>
          </div>
          <pre class="cypher-codebox"><code class="selectable-text">${escapeHtml(data.cypher_executed)}</code></pre>
          ${data.supporting_evidence && Object.keys(data.supporting_evidence).length > 0 ? `
          <div class="path-section-header" style="margin-top: 6px;">
            <span class="path-title"><i class="fa-solid fa-circle-nodes"></i> Supporting Graph Evidence:</span>
            <button class="copy-code-btn" onclick="copyCodeSnippet(this)"><i class="fa-regular fa-copy"></i> Copy JSON</button>
          </div>
          <pre class="cypher-codebox"><code class="selectable-text">${escapeHtml(JSON.stringify(data.supporting_evidence, null, 2))}</code></pre>
          ` : ''}
        </div>
      </details>
      ` : ''}
    `;

    // Automatically shift map visual focus to the queried path / road / corridor
    if (!isOutOfScope && !isBoundary) {
      focusMapFromAgentResponse(data);
    }

    // Refresh memory timeline and network in background
    fetchMemories();
    fetchNetwork();
  } catch (err) {
    loadingBubble.innerHTML = `
      <div class="bubble-header"><span class="agent-name text-rose">Error</span></div>
      <div class="text-rose">Failed to query agent: ${err.message}</div>
    `;
  }
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Ground scout observation actions
async function quickVerifyPassable(roadId, event) {
  if (event) event.stopPropagation();
  try {
    const resp = await fetch(`${API_BASE}/api/observations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        road_id: roadId,
        reported_status: "verified_passable",
        note: "Ground scout recon unit verified road segment is passable",
        source_id: "src_scout",
        override: true
      })
    });
    if (!resp.ok) throw new Error("Update failed");
    showToast(`Road ${roadId} verified passable!`, "success");
    await loadAllData();
    if (selectedElement && selectedElement.id === roadId) {
      selectRoad(roadId);
    }
  } catch (e) {
    showToast(`Error updating road ${roadId}`, "error");
  }
}

async function quickConfirmBlocked(roadId, event) {
  if (event) event.stopPropagation();
  try {
    const resp = await fetch(`${API_BASE}/api/observations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        road_id: roadId,
        reported_status: "blocked",
        note: "Ground scout recon unit confirmed road segment is blocked by landslide",
        source_id: "src_scout",
        override: true
      })
    });
    if (!resp.ok) throw new Error("Update failed");
    showToast(`Road ${roadId} confirmed blocked!`, "warning");
    await loadAllData();
    if (selectedElement && selectedElement.id === roadId) {
      selectRoad(roadId);
    }
  } catch (e) {
    showToast(`Error updating road ${roadId}`, "error");
  }
}

// Side dock navigation and event handlers
let isDockCollapsed = false;

function toggleSideDock() {
  if (isDockCollapsed) {
    openSideDock();
  } else {
    collapseSideDock();
  }
}

function collapseSideDock() {
  isDockCollapsed = true;
  const mainContent = document.getElementById("main-content");
  const launcher = document.getElementById("btn-dock-launcher");
  if (mainContent) mainContent.classList.add("dock-collapsed");
  if (launcher) launcher.classList.remove("hidden");
  triggerMapResize();
}

function openSideDock() {
  isDockCollapsed = false;
  const mainContent = document.getElementById("main-content");
  const launcher = document.getElementById("btn-dock-launcher");
  if (mainContent) mainContent.classList.remove("dock-collapsed");
  if (launcher) launcher.classList.add("hidden");
  triggerMapResize();
}

function triggerMapResize() {
  if (!map) return;
  const start = performance.now();
  const step = () => {
    if (map) map.resize();
    if (performance.now() - start < 380) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}

function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.id === tabId);
  });
}

function setupEventHandlers() {
  // Side Dock Collapse button in dock header
  const btnCollapseDock = document.getElementById("btn-collapse-dock");
  if (btnCollapseDock) {
    btnCollapseDock.addEventListener("click", collapseSideDock);
  }

  // Floating edge launcher on map
  const btnLauncher = document.getElementById("btn-dock-launcher");
  if (btnLauncher) {
    btnLauncher.addEventListener("click", openSideDock);
  }

  // Full-height left border collapse strip on Side Dock
  const dockLeftBar = document.getElementById("dock-left-toggle-bar");
  if (dockLeftBar) {
    dockLeftBar.addEventListener("click", collapseSideDock);
  }

  // Keyboard shortcut: Alt+D or Meta+B
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.altKey) && (e.key.toLowerCase() === "d" || e.key.toLowerCase() === "b")) {
      e.preventDefault();
      toggleSideDock();
    }
  });

  // Toggle OSM Road Survey lines on Map
  const btnToggleOsm = document.getElementById("btn-toggle-osm-roads");
  if (btnToggleOsm) {
    btnToggleOsm.addEventListener("click", () => {
      if (!map) return;
      const curVis = map.getLayoutProperty("roads-osm-line", "visibility");
      const isVisible = curVis !== "none";
      const nextVis = isVisible ? "none" : "visible";
      if (map.getLayer("roads-osm-line")) map.setLayoutProperty("roads-osm-line", "visibility", nextVis);
      if (map.getLayer("bridges-osm-line")) map.setLayoutProperty("bridges-osm-line", "visibility", nextVis);
      if (map.getLayer("destroyed-features-line")) map.setLayoutProperty("destroyed-features-line", "visibility", nextVis);
      btnToggleOsm.classList.toggle("active", !isVisible);
      showToast(`OSM Survey Lines: ${!isVisible ? 'Enabled' : 'Hidden'}`, "info");
    });
  }

  // Recon Filter Pills
  const reconPills = document.querySelectorAll("#recon-filter-group .pill-btn");
  reconPills.forEach(btn => {
    btn.addEventListener("click", () => {
      reconPills.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentReconFilter = btn.dataset.reconFilter;
      renderMissions(currentMissions);
      const cleanLabel = (btn.textContent || btn.innerText || "")
        .replace(/\s+/g, " ")
        .replace(/\(\s+/g, "(")
        .replace(/\s+\)/g, ")")
        .trim();
      showToast(`Filter: ${cleanLabel}`, "info");
    });
  });

  // Preset chips horizontal wheel scroll
  const chipRail = document.getElementById("presets-chip-rail");
  if (chipRail) {
    chipRail.addEventListener("wheel", (e) => {
      if (e.deltaY !== 0) {
        e.preventDefault();
        chipRail.scrollLeft += e.deltaY;
      }
    }, { passive: false });
  }

  // Center / Fit Map button
  const btnFit = document.getElementById("btn-fit-map");
  if (btnFit) {
    btnFit.addEventListener("click", () => {
      fitMapToNetwork();
      showToast("Centered map on Trishuli network", "info");
    });
  }

  // Bridge Quick Filter buttons
  const filterPills = document.querySelectorAll("#bridge-filter-group .pill-btn");
  filterPills.forEach(btn => {
    btn.addEventListener("click", () => {
      filterPills.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentBridgeFilter = btn.dataset.filter;
      renderBridgeMarkers(allBridgesData, currentBridgeFilter);
      const cleanLabel = (btn.textContent || btn.innerText || "")
        .replace(/\s+/g, " ")
        .replace(/\(\s+/g, "(")
        .replace(/\s+\)/g, ")")
        .trim();
      showToast(`Bridge Filter: ${cleanLabel}`, "info");
    });
  });

  // Trigger Landslide button
  document.getElementById("btn-trigger-landslide").addEventListener("click", async () => {
    try {
      showToast("Simulating landslide on road_syabru_dhunche...", "warning");
      const resp = await fetch(`${API_BASE}/api/demo/trigger-landslide`, { method: "POST" });
      if (!resp.ok) throw new Error("Simulation failed");
      showToast("Landslide triggered! Graph re-ranked missions dynamically.", "success");
      await loadAllData();
      selectRoad("road_syabru_dhunche");
    } catch (e) {
      showToast("Failed to simulate landslide", "error");
    }
  });

  // Reset Graph button
  document.getElementById("btn-reset-graph").addEventListener("click", async () => {
    try {
      showToast("Resetting FalkorDB to baseline seed...", "warning");
      const resp = await fetch(`${API_BASE}/api/demo/reset`, { method: "POST" });
      if (!resp.ok) throw new Error("Reset failed");
      showToast("FalkorDB reset completed!", "success");
      await loadAllData();
      document.getElementById("inspector-empty-state").classList.remove("hidden");
      document.getElementById("inspector-details").classList.add("hidden");
      const bDetails = document.getElementById("inspector-bridge-details");
      if (bDetails) bDetails.classList.add("hidden");
    } catch (e) {
      showToast("Failed to reset FalkorDB", "error");
    }
  });

  // Wipe Graph (Zero State) Button & Modal
  const btnWipeGraph = document.getElementById("btn-wipe-graph");
  const wipeModal = document.getElementById("wipe-modal");
  const btnCancelWipe = document.getElementById("btn-cancel-wipe");
  const btnConfirmWipe = document.getElementById("btn-confirm-wipe");

  if (btnWipeGraph && wipeModal) {
    btnWipeGraph.addEventListener("click", () => {
      wipeModal.classList.remove("hidden");
    });
  }

  if (btnCancelWipe && wipeModal) {
    btnCancelWipe.addEventListener("click", () => {
      wipeModal.classList.add("hidden");
    });
  }

  if (btnConfirmWipe && wipeModal) {
    btnConfirmWipe.addEventListener("click", async () => {
      try {
        wipeModal.classList.add("hidden");
        showToast("Wiping FalkorDB to Zero State (0 nodes)...", "warning");
        const resp = await fetch(`${API_BASE}/api/graph/wipe`, { method: "POST" });
        if (!resp.ok) throw new Error("Wipe failed");
        showToast("FalkorDB wiped. 0 nodes in graph.", "info");
        await loadAllData();
      } catch (err) {
        showToast("Failed to wipe FalkorDB graph", "error");
      }
    });
  }

  // Zero-State Ingestion: File Picker & Drag-and-Drop
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("topology-file-input");
  const btnBrowseFile = document.getElementById("btn-browse-file");
  const btnLoadSample = document.getElementById("btn-load-sample-dataset");

  if (btnBrowseFile && fileInput) {
    btnBrowseFile.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => {
      fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        importTopologyFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        importTopologyFile(e.target.files[0]);
      }
    });
  }

  if (btnLoadSample) {
    btnLoadSample.addEventListener("click", () => {
      loadSampleTopology();
    });
  }

  // Refresh missions button
  document.getElementById("btn-refresh-missions").addEventListener("click", () => {
    fetchMissions();
    showToast("Missions refreshed", "success");
  });

  // Refresh memory button
  document.getElementById("btn-refresh-memory").addEventListener("click", () => {
    fetchMemories();
    showToast("Memory timeline refreshed", "success");
  });

  // Manual observation form
  document.getElementById("form-add-observation").addEventListener("submit", async (e) => {
    e.preventDefault();
    const roadId = document.getElementById("obs-road-id").value;
    const status = document.getElementById("obs-status").value;
    const sourceId = document.getElementById("obs-source").value;
    const note = document.getElementById("obs-note").value;
    const override = document.getElementById("obs-override").checked;

    try {
      const resp = await fetch(`${API_BASE}/api/observations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          road_id: roadId,
          reported_status: status,
          source_id: sourceId,
          note: note,
          override: override
        })
      });

      if (!resp.ok) throw new Error("Failed to record observation");
      showToast("Observation recorded & synced in FalkorDB!", "success");
      await loadAllData();
      selectRoad(roadId);
    } catch (err) {
      showToast("Error recording observation", "error");
    }
  });

  // Agent Chat Form
  document.getElementById("agent-input-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("agent-input-text");
    const val = input.value.trim();
    if (val) {
      sendAgentQuery(val);
      input.value = "";
    }
  });

  // Quick Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.query;
      sendAgentQuery(q);
    });
  });

  // Copy Cypher button (Road / Settlement)
  document.getElementById("btn-copy-cypher").addEventListener("click", () => {
    const code = document.getElementById("insp-cypher-code").innerText;
    navigator.clipboard.writeText(code).then(() => {
      showToast("Cypher copied to clipboard!", "success");
    });
  });

  // Copy Cypher button (Bridge)
  const btnCopyBridgeCypher = document.getElementById("btn-copy-bridge-cypher");
  if (btnCopyBridgeCypher) {
    btnCopyBridgeCypher.addEventListener("click", () => {
      const code = document.getElementById("bridge-cypher-code").innerText;
      navigator.clipboard.writeText(code).then(() => {
        showToast("Bridge Cypher copied to clipboard!", "success");
      });
    });
  }
}

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerText = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 4000);
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Markdown formatter for agent responses
function formatMarkdown(raw) {
  if (!raw) return "";
  let text = String(raw).trim();

  // 1. Extract fenced code blocks
  const codeBlocks = [];
  text = text.replace(/```(?:([a-zA-Z0-9_-]+)?\n)?([\s\S]*?)```/g, (match, lang, code) => {
    const idx = codeBlocks.length;
    const cleanLang = (lang || "cypher").toLowerCase();
    codeBlocks.push(`
      <div class="agent-code-block">
        <div class="code-block-header">
          <span><i class="fa-solid fa-code"></i> ${escapeHtml(cleanLang)}</span>
          <button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.closest('.agent-code-block').querySelector('code').innerText); showToast('Code copied to clipboard', 'info');"><i class="fa-regular fa-copy"></i> Copy</button>
        </div>
        <pre><code class="language-${escapeHtml(cleanLang)} selectable-text">${escapeHtml(code.trim())}</code></pre>
      </div>
    `);
    return `__CODE_BLOCK_${idx}__`;
  });

  // Escape HTML entities
  text = escapeHtml(text);

  // Headings
  text = text.replace(/^### (.*$)/gim, '<h4 class="agent-heading">$1</h4>');
  text = text.replace(/^## (.*$)/gim, '<h3 class="agent-heading">$1</h3>');
  text = text.replace(/^# (.*$)/gim, '<h2 class="agent-heading">$1</h2>');

  // Emphasis
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="agent-strong">$1</strong>');
  text = text.replace(/\*([^\*]+)\*/g, '<em class="agent-em">$1</em>');

  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code class="agent-inline-code">$1</code>');

  // Mission badges and priority pills
  text = text.replace(/Rank (\d+):/g, '<span class="agent-rank-badge">Rank $1</span>');
  text = text.replace(/Priority Score: (\d+(?:\.\d+)?)\/100/g, '<span class="priority-pill priority-critical">Priority $1/100</span>');

  // Tables and lists
  const lines = text.split("\n");
  let inList = false;
  let inTable = false;
  let tableRows = [];
  const processed = [];

  const flushTable = () => {
    if (tableRows.length >= 2) {
      const parseRow = r => r.replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
      const headerCells = parseRow(tableRows[0]);
      let bodyRows = tableRows.slice(1);
      const isDivider = r => r.replace(/^\|/, "").replace(/\|$/, "").split("|").every(c => /^[-:\s]+$/.test(c.trim()));
      if (bodyRows.length > 0 && isDivider(bodyRows[0])) {
        bodyRows = bodyRows.slice(1);
      }
      processed.push(`
        <div class="agent-table-wrapper">
          <table class="agent-table">
            <thead><tr>${headerCells.map(c => `<th>${c}</th>`).join("")}</tr></thead>
            <tbody>${bodyRows.map(r => `<tr>${parseRow(r).map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
          </table>
        </div>
      `);
    } else if (tableRows.length > 0) {
      tableRows.forEach(r => processed.push(`<p class="agent-para">${r}</p>`));
    }
    tableRows = [];
    inTable = false;
  };

  for (let line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      if (inList) {
        processed.push('</ul>');
        inList = false;
      }
      inTable = true;
      tableRows.push(trimmed);
      continue;
    } else if (inTable) {
      flushTable();
    }

    const bulletMatch = trimmed.match(/^([•\-\*]|\d+\.)\s+(.*)/);
    if (bulletMatch) {
      if (!inList) {
        processed.push('<ul class="agent-list">');
        inList = true;
      }
      processed.push(`<li>${bulletMatch[2]}</li>`);
    } else {
      if (inList) {
        processed.push('</ul>');
        inList = false;
      }
      if (trimmed.length > 0) {
        if (!trimmed.startsWith('<h') && !trimmed.startsWith('__CODE_BLOCK_') && !trimmed.startsWith('<div')) {
          processed.push(`<p class="agent-para">${trimmed}</p>`);
        } else {
          processed.push(trimmed);
        }
      }
    }
  }

  if (inTable) flushTable();
  if (inList) processed.push('</ul>');

  let html = processed.join("\n");

  // Restore code blocks
  codeBlocks.forEach((block, idx) => {
    html = html.replace(`__CODE_BLOCK_${idx}__`, block);
  });

  return html;
}

function renderPathBreadcrumbs(evidence) {
  if (!evidence || typeof evidence !== "object") return "";

  // 0. Point-to-Point Corridor Route (from find_route_between)
  const primaryP = evidence.primary_route || (evidence.routes && evidence.routes[0]);
  if (primaryP && (primaryP.location_names || primaryP.locations)) {
    const p = primaryP;
    const locs = p.location_names || (p.locations || []).map(l => (typeof l === 'object' ? (l.name || l.id) : l));
    const chips = [];
    locs.forEach((loc, i) => {
      const locId = (p.locations && p.locations[i]) ? (p.locations[i].id || p.locations[i].name || p.locations[i]) : loc;
      chips.push(`<span class="path-chip loc interactive" onclick="focusCorridorOnMap([], { locations: ['${locId}'], settlements: ['${locId}'] })" title="Focus ${escapeHtml(loc)} on map"><i class="fa-solid fa-location-dot"></i> ${escapeHtml(loc)}</span>`);
      if (i < locs.length - 1) {
        const roadId = (p.road_ids && p.road_ids[i]) ? p.road_ids[i] : "";
        const roadName = p.road_names && p.road_names[i] ? p.road_names[i] : "corridor";
        const seg = p.segments && p.segments[i] ? p.segments[i] : {};
        const stClass = seg.status === 'verified_passable' ? 'passable' : (seg.status === 'blocked' ? 'blocked' : '');
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
        chips.push(`<span class="path-chip road ${stClass} interactive" onclick="focusCorridorOnMap(['${roadId}'], { fitBounds: true })" title="Focus ${escapeHtml(roadName)} on map"><i class="fa-solid fa-road"></i> ${escapeHtml(roadName)}</span>`);
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
      }
    });
    const allRoadIds = JSON.stringify(p.road_ids || []);
    const allLocIds = JSON.stringify((p.locations || []).map(l => (typeof l === 'object' ? (l.id || l.name) : l)));
    return `
      <div class="path-section-header">
        <span class="path-title"><i class="fa-solid fa-route"></i> Transit Corridor Route (${escapeHtml(p.tier_label || 'Calculated')} - ${p.total_km} km):</span>
        <button class="copy-code-btn" style="color: #ffffff; border-color: rgba(255,255,255,0.4);" onclick='focusCorridorOnMap(${allRoadIds}, { fitBounds: true, locations: ${allLocIds}, settlements: ${allLocIds} })'><i class="fa-solid fa-crosshairs"></i> Focus Route on Map</button>
      </div>
      <div class="path-breadcrumbs-box">${chips.join("")}</div>
    `;
  }

  // Verified paths to healthcare
  if (evidence.verified_paths && evidence.verified_paths.length > 0) {
    const p = evidence.verified_paths[0];
    const locs = p.location_names || [];
    const chips = [];
    locs.forEach((loc, i) => {
      const locId = (p.locations && p.locations[i]) ? p.locations[i] : loc;
      chips.push(`<span class="path-chip loc interactive" onclick="focusCorridorOnMap([], { locations: ['${locId}'], settlements: ['${locId}'] })" title="Focus ${escapeHtml(loc)} on map"><i class="fa-solid fa-location-dot"></i> ${escapeHtml(loc)}</span>`);
      if (i < locs.length - 1) {
        const roadId = (p.road_ids && p.road_ids[i]) ? p.road_ids[i] : "";
        const roadName = p.road_names && p.road_names[i] ? p.road_names[i] : "corridor";
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
        chips.push(`<span class="path-chip road passable interactive" onclick="focusCorridorOnMap(['${roadId}'], { fitBounds: true })" title="Focus ${escapeHtml(roadName)} on map"><i class="fa-solid fa-road"></i> ${escapeHtml(roadName)}</span>`);
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
      }
    });
    if (p.facility_name) {
      const facId = p.facility_id || "";
      chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
      chips.push(`<span class="path-chip fac interactive" onclick="focusCorridorOnMap([], { locations: ['${facId}'], settlements: ['${facId}'] })" title="Focus ${escapeHtml(p.facility_name)} on map"><i class="fa-solid fa-hospital"></i> ${escapeHtml(p.facility_name)} (${p.total_km} km)</span>`);
    }
    const allRoadIds = JSON.stringify(p.road_ids || []);
    return `
      <div class="path-section-header">
        <span class="path-title"><i class="fa-solid fa-route"></i> Verified Topological Path:</span>
        <button class="copy-code-btn" style="color: #38bdf8; border-color: rgba(56,189,248,0.3);" onclick='focusCorridorOnMap(${allRoadIds}, { fitBounds: true, settlements: ${JSON.stringify(p.locations || [])} })'><i class="fa-solid fa-crosshairs"></i> Focus Route on Map</button>
      </div>
      <div class="path-breadcrumbs-box">${chips.join("")}</div>
    `;
  }

  // Potential paths needing recon
  if (evidence.potential_paths && evidence.potential_paths.length > 0) {
    const p = evidence.potential_paths[0];
    const locs = p.location_names || [];
    const chips = [];
    locs.forEach((loc, i) => {
      const locId = (p.locations && p.locations[i]) ? p.locations[i] : loc;
      chips.push(`<span class="path-chip loc interactive" onclick="focusCorridorOnMap([], { locations: ['${locId}'], settlements: ['${locId}'] })" title="Focus ${escapeHtml(loc)} on map"><i class="fa-solid fa-location-dot"></i> ${escapeHtml(loc)}</span>`);
      if (i < locs.length - 1) {
        const roadId = (p.road_ids && p.road_ids[i]) ? p.road_ids[i] : "";
        const roadName = p.road_names && p.road_names[i] ? p.road_names[i] : "Unverified Segment";
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
        chips.push(`<span class="path-chip road interactive" onclick="focusCorridorOnMap(['${roadId}'], { fitBounds: true })" title="Focus ${escapeHtml(roadName)} on map"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(roadName)}</span>`);
        chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
      }
    });
    if (p.facility_name) {
      const facId = p.facility_id || "";
      chips.push(`<span class="path-arrow"><i class="fa-solid fa-arrow-right"></i></span>`);
      chips.push(`<span class="path-chip fac interactive" onclick="focusCorridorOnMap([], { locations: ['${facId}'], settlements: ['${facId}'] })" title="Focus ${escapeHtml(p.facility_name)} on map"><i class="fa-solid fa-hospital"></i> ${escapeHtml(p.facility_name)}</span>`);
    }
    const allRoadIds = JSON.stringify(p.road_ids || []);
    return `
      <div class="path-section-header">
        <span class="path-title"><i class="fa-solid fa-route"></i> Potential Bypass Route:</span>
        <button class="copy-code-btn" style="color: #fbbf24; border-color: rgba(251,191,36,0.3);" onclick='focusCorridorOnMap(${allRoadIds}, { fitBounds: true, settlements: ${JSON.stringify(p.locations || [])} })'><i class="fa-solid fa-crosshairs"></i> Focus Route on Map</button>
      </div>
      <div class="path-breadcrumbs-box">${chips.join("")}</div>
    `;
  }

  // Downstream settlements affected in blast radius
  if (evidence.affected_settlements && evidence.affected_settlements.length > 0) {
    const targetRoad = evidence.road_id || "";
    const affIds = evidence.affected_settlements.map(s => s.id || s);
    const chips = evidence.affected_settlements.map(s => {
      const sId = s.id || s;
      const sName = s.name || s;
      const hh = s.households ? `(${s.households} HH)` : '';
      return `<span class="path-chip loc interactive" onclick="focusCorridorOnMap(['${targetRoad}'], { locations: ['${sId}'], settlements: ['${sId}'] })" title="Focus ${escapeHtml(sName)} on map"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(sName)} ${hh}</span>`;
    });
    return `
      <div class="path-section-header">
        <span class="path-title"><i class="fa-solid fa-burst"></i> Cut-Edge Isolated Settlements:</span>
        <button class="copy-code-btn" style="color: #f87171; border-color: rgba(248,113,113,0.3);" onclick='focusCorridorOnMap(["${targetRoad}"], { fitBounds: true, settlements: ${JSON.stringify(affIds)} })'><i class="fa-solid fa-crosshairs"></i> Focus Blast Radius on Map</button>
      </div>
      <div class="path-breadcrumbs-box">${chips.join("<span class='path-arrow'>•</span>")}</div>
    `;
  }

  return "";
}

window.copyCodeSnippet = function(btn) {
  const container = btn.closest(".path-section-header") ? btn.closest(".path-section-header").nextElementSibling : null;
  const codeBox = container ? container.querySelector("code") : null;
  const text = codeBox ? codeBox.innerText : "";
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const origHtml = btn.innerHTML;
      btn.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
      btn.classList.add("copied");
      setTimeout(() => {
        btn.innerHTML = origHtml;
        btn.classList.remove("copied");
      }, 1800);
    }).catch(() => {
      showToast("Failed to copy", "error");
    });
  }
};

window.copyBubbleAnswer = function(btn) {
  const bubble = btn.closest(".chat-bubble");
  const body = bubble ? bubble.querySelector(".bubble-body") : null;
  const text = body ? body.innerText : "";
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const origHtml = btn.innerHTML;
      btn.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
      btn.classList.add("copied");
      setTimeout(() => {
        btn.innerHTML = origHtml;
        btn.classList.remove("copied");
      }, 1800);
    }).catch(() => {
      showToast("Failed to copy", "error");
    });
  }
};

// Ingestion handlers for sample and custom topology uploads
function logToIngestTerminal(text) {
  const terminal = document.getElementById("ingest-terminal");
  const output = document.getElementById("ingest-terminal-output");
  if (terminal) terminal.classList.remove("hidden");
  if (output) {
    output.innerText += text + "\n";
    output.scrollTop = output.scrollHeight;
  }
}

async function importTopologyFile(file) {
  if (!file) return;
  const terminal = document.getElementById("ingest-terminal");
  const output = document.getElementById("ingest-terminal-output");
  if (terminal) terminal.classList.remove("hidden");
  if (output) output.innerText = "";

  logToIngestTerminal(`Reading "${file.name}"...`);
  try {
    const text = await file.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (je) {
      logToIngestTerminal(`Error: Invalid JSON format (${je.message})`);
      showToast("Uploaded file is not valid JSON", "error");
      return;
    }

    logToIngestTerminal(`Ingesting nodes and lifelines into FalkorDB...`);

    const resp = await fetch(`${API_BASE}/api/graph/import-json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: text
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      logToIngestTerminal(`Error: ${err.detail || resp.statusText}`);
      showToast("Topology ingestion failed", "error");
      return;
    }

    const data = await resp.json();
    const c = data.imported_counts || {};
    logToIngestTerminal(`Imported: ${c.locations || 0} locations, ${c.road_segments || 0} roads, ${c.bridges || 0} bridges.`);
    logToIngestTerminal(`Generated ${c.active_missions || 0} recon missions.`);
    showToast(`Topology imported (${c.locations || 0} locations)`, "success");

    setTimeout(async () => {
      await loadAllData();
      if (terminal) terminal.classList.add("hidden");
    }, 1000);
  } catch (err) {
    logToIngestTerminal(`Error: ${err.message}`);
    showToast("Error uploading file", "error");
  }
}

async function loadSampleTopology() {
  const terminal = document.getElementById("ingest-terminal");
  const output = document.getElementById("ingest-terminal-output");
  if (terminal) terminal.classList.remove("hidden");
  if (output) output.innerText = "";

  logToIngestTerminal(`Loading Trishuli Flood Corridor...`);

  try {
    const resp = await fetch(`${API_BASE}/api/graph/import-json?sample=true`, {
      method: "POST"
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      logToIngestTerminal(`Error: ${err.detail || resp.statusText}`);
      showToast("Sample loading failed", "error");
      return;
    }

    const data = await resp.json();
    const c = data.imported_counts || {};
    logToIngestTerminal(`Imported: ${c.locations || 0} locations, ${c.road_segments || 0} roads, ${c.bridges || 0} bridges.`);
    logToIngestTerminal(`Generated ${c.active_missions || 0} recon missions.`);
    showToast(`Trishuli Corridor Loaded (${c.locations || 0} locations)`, "success");

    setTimeout(async () => {
      await loadAllData();
      if (terminal) terminal.classList.add("hidden");
    }, 1000);
  } catch (err) {
    logToIngestTerminal(`Error: ${err.message}`);
    showToast("Error loading sample dataset", "error");
  }
}

