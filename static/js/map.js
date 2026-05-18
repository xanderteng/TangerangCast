/* ============================================================
   map.js - Leaflet map logic for current, forecast, historic
   ============================================================ */

const MAP_CENTER = [-6.2, 106.55];
const MAP_ZOOM = 11;
const RAIN_CLR = "#3B82F6";
const DRY_CLR = "#F59E0B";

const MAP_STYLES = {
  dark: {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a>',
    background: "#0b1220",
  },
  light: {
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a>',
    background: "#eef2f6",
  },
  colorful: {
    url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a>',
    background: "#d9edf2",
  },
};

const DATA_SOURCES = {
  current: {
    url: "/api/current",
  },
  forecast: {
    timesUrl: "/api/forecast/times",
    pointUrl: (time) => `/api/forecast/${encodeURIComponent(time)}`,
  },
  historic: {
    timesUrl: "/api/historic/times",
    pointUrl: (time) => `/api/historic/${encodeURIComponent(time)}`,
  },
};

const map = L.map("map", {
  center: MAP_CENTER,
  zoom: MAP_ZOOM,
  zoomControl: false,
});

let tileLayer = null;
let pointLayer = null;
let activeMode = "current";
let activeStyle = "dark";
let activeTimes = [];
let activeTimeIdx = 0;

L.control.zoom({ position: "bottomright" }).addTo(map);
switchMapStyle(activeStyle);
switchMode("current");

function switchMapStyle(styleName) {
  const style = MAP_STYLES[styleName] || MAP_STYLES.dark;
  activeStyle = styleName;

  if (tileLayer) {
    map.removeLayer(tileLayer);
  }

  tileLayer = L.tileLayer(style.url, {
    attribution: style.attribution,
    maxZoom: 19,
  }).addTo(map);

  document.querySelectorAll(".style-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `style-${styleName}`);
  });

  document.documentElement.style.setProperty("--leaflet-bg", style.background);
}

function fmt(value, fallback = "-") {
  return value === null || value === undefined || Number.isNaN(value) ? fallback : value;
}

function fmtCoord(value) {
  return typeof value === "number" ? value.toFixed(3) : fmt(value);
}

function formatTimeLabel(value) {
  if (!value) return "-";
  return value.replace("T", " ");
}

function rainColor(point) {
  if (point.rain_label === 1) return RAIN_CLR;
  if (point.rain_label === 0) return DRY_CLR;
  return point.cloud_cover > 60 ? RAIN_CLR : DRY_CLR;
}

function rainText(point) {
  if (point.rain_label === 1) return "Rain";
  if (point.rain_label === 0) return "Clear";
  return point.cloud_cover > 60 ? "Possible rain" : "Clear";
}

function makeCircle(point, mode) {
  const color = rainColor(point);
  const circle = L.circleMarker([point.latitude, point.longitude], {
    radius: 5,
    fillColor: color,
    color,
    weight: 0,
    fillOpacity: 0.78,
  });

  circle.bindTooltip(`
    <strong>${rainText(point)}</strong><br>
    Time: ${formatTimeLabel(point.timestamp)}<br>
    ${fmt(point.latitude)}, ${fmt(point.longitude)}
  `, { sticky: true });

  circle.on("click", () => openPanel(point, mode));
  return circle;
}

function drawPoints(payload, mode) {
  if (pointLayer) {
    map.removeLayer(pointLayer);
  }

  pointLayer = L.layerGroup();
  payload.points.forEach((point) => makeCircle(point, mode).addTo(pointLayer));
  pointLayer.addTo(map);

  updateStats(payload.points, mode);
  updateDatasetMeta(payload);
}

function updateDatasetMeta(payload) {
  const meta = document.getElementById("fetch-badge");
  if (!meta) return;

  const fetchTime = payload.fetch_time || "-";
  const timestamp = payload.timestamp && payload.timestamp !== payload.fetch_time
    ? ` | Selected: ${formatTimeLabel(payload.timestamp)}`
    : "";
  meta.textContent = `Fetch: ${fetchTime}${timestamp}`;
}

function updateStats(points, mode) {
  const total = points.length || 1;
  const rainCount = points.filter((p) => rainText(p) !== "Clear").length;
  const avgTemp = points.reduce((sum, p) => sum + Number(p.temperature || 0), 0) / total;
  const avgHumid = points.reduce((sum, p) => sum + Number(p.humidity || 0), 0) / total;

  document.getElementById("stat-rain").textContent =
    mode === "forecast" ? `Points ${points.length}` : `Rain ${rainCount}/${points.length}`;
  document.getElementById("stat-temp").textContent = `Temp ${avgTemp.toFixed(1)} C`;
  document.getElementById("stat-humid").textContent = `Hum ${avgHumid.toFixed(0)}%`;
}

function openPanel(point, mode) {
  const panel = document.getElementById("detail-panel");
  const grid = document.getElementById("detail-grid");
  const showRain = mode !== "forecast" || point.rain_label !== null;
  const rainHTML = showRain
    ? `<div class="detail-cell rain-status">
         <div class="detail-cell__label">Rain Status</div>
         <span class="rain-badge ${rainText(point) === "Clear" ? "clear" : "raining"}">
           ${rainText(point)}
         </span>
       </div>`
    : "";

  grid.innerHTML = `
    ${rainHTML}
    <div class="detail-cell">
      <div class="detail-cell__label">Timestamp</div>
      <div class="detail-cell__value compact">${formatTimeLabel(point.timestamp)}</div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Fetch Time</div>
      <div class="detail-cell__value compact">${fmt(point.fetch_time)}</div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Temperature</div>
      <div class="detail-cell__value">${fmt(point.temperature)}<span class="detail-cell__unit"> C</span></div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Humidity</div>
      <div class="detail-cell__value">${fmt(point.humidity)}<span class="detail-cell__unit"> %</span></div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Wind</div>
      <div class="detail-cell__value">${fmt(point.wind_speed)}<span class="detail-cell__unit"> km/h</span></div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Cloud</div>
      <div class="detail-cell__value">${fmt(point.cloud_cover)}<span class="detail-cell__unit"> %</span></div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Pressure</div>
      <div class="detail-cell__value">${fmt(point.pressure)}<span class="detail-cell__unit"> hPa</span></div>
    </div>
    <div class="detail-cell">
      <div class="detail-cell__label">Coordinate</div>
      <div class="detail-cell__value compact">${fmtCoord(point.latitude)},<br>${fmtCoord(point.longitude)}</div>
    </div>
  `;

  panel.classList.remove("hidden");
}

function closePanel() {
  document.getElementById("detail-panel").classList.add("hidden");
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return res.json();
}

async function loadCurrent() {
  const payload = await fetchJson(DATA_SOURCES.current.url);
  drawPoints(payload, "current");
}

async function loadTimes(mode) {
  const payload = await fetchJson(DATA_SOURCES[mode].timesUrl);
  activeTimes = payload.times || [];
  activeTimeIdx = activeTimes.length ? activeTimes.length - 1 : 0;
  renderTimeSelector();
}

function renderTimeSelector() {
  const select = document.getElementById("time-select");
  select.innerHTML = "";

  if (!activeTimes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No timestamps available";
    select.appendChild(option);
  } else {
    activeTimes.forEach((time, idx) => {
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = formatTimeLabel(time);
      option.selected = idx === activeTimeIdx;
      select.appendChild(option);
    });
  }

  updateTimeButtons();
}

function updateTimeButtons() {
  document.getElementById("btn-prev").disabled = activeTimeIdx === 0 || !activeTimes.length;
  document.getElementById("btn-next").disabled = activeTimeIdx === activeTimes.length - 1 || !activeTimes.length;
  document.getElementById("time-select").value = activeTimes.length ? String(activeTimeIdx) : "";
}

async function loadTimedModeAt(idx) {
  if (!activeTimes.length) {
    drawPoints({ fetch_time: null, timestamp: null, points: [] }, activeMode);
    return;
  }

  activeTimeIdx = Math.max(0, Math.min(activeTimes.length - 1, idx));
  updateTimeButtons();

  const time = activeTimes[activeTimeIdx];
  const payload = await fetchJson(DATA_SOURCES[activeMode].pointUrl(time));
  drawPoints(payload, activeMode);
}

async function switchMode(mode) {
  activeMode = mode;
  closePanel();

  document.getElementById("btn-current").classList.toggle("active", mode === "current");
  document.getElementById("btn-forecast").classList.toggle("active", mode === "forecast");
  document.getElementById("btn-historic").classList.toggle("active", mode === "historic");
  document.getElementById("time-controls").classList.toggle("hidden", mode === "current");

  if (mode === "current") {
    activeTimes = [];
    await loadCurrent();
  } else {
    await loadTimes(mode);
    await loadTimedModeAt(activeTimeIdx);
  }
}

function selectTime(indexValue) {
  if (indexValue === "") return;
  loadTimedModeAt(Number(indexValue));
  closePanel();
}

function stepTime(delta) {
  loadTimedModeAt(activeTimeIdx + delta);
  closePanel();
}
