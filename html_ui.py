from __future__ import annotations

import copy
import json
import math
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from main import POSE_OFFSET_X_CM, POSE_OFFSET_Y_CM, POSE_UNITS_TO_CM

DUST_MAP_RAW_LEFT_X = -6600.0
DUST_MAP_RAW_RIGHT_X = -5100.0
DUST_MAP_RAW_TOP_Y = -10200.0
DUST_MAP_RAW_BOTTOM_Y = -11100.0
DUST_MAP_LEFT_X = DUST_MAP_RAW_LEFT_X
DUST_MAP_RIGHT_X = DUST_MAP_RAW_RIGHT_X
DUST_MAP_TOP_Y = DUST_MAP_RAW_TOP_Y
DUST_MAP_BOTTOM_Y = DUST_MAP_RAW_BOTTOM_Y


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rover Viewer</title>
  <style>
    :root {
      --bg: #0b0b0c;
      --panel: #111214;
      --panel-border: #2b2e33;
      --card: #17181a;
      --card-border: #3c4047;
      --text: #f2f3f5;
      --muted: #8f97a3;
      --path: #6eaaff;
      --goal: #f3d04e;
      --target: #2c8fed;
      --obstacle: #db5151;
      --obstacle-low: #d7ad4f;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: "Segoe UI", sans-serif; }
    body { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 18px; padding: 18px; }
    .mapWrap { position: relative; min-width: 0; border: 1px solid #30343a; border-radius: 22px; overflow: hidden; background:
      radial-gradient(circle at top left, rgba(255,255,255,0.05), transparent 30%),
      linear-gradient(180deg, #121416 0%, #0d0f12 100%); }
    canvas { width: 100%; height: 100%; display: block; }
    .sidebar { border: 1px solid var(--panel-border); border-radius: 22px; background: linear-gradient(180deg, #111214 0%, #0d0e10 100%); padding: 18px; }
    .title { font-size: 29px; font-weight: 700; margin: 2px 0 18px; }
    .route { position: relative; padding-left: 38px; margin-bottom: 16px; }
    .rail { position: absolute; left: 11px; top: 40px; width: 3px; height: 190px; background: #4d5158; border-radius: 999px; }
    .dot { position: absolute; left: 0; width: 24px; height: 24px; border-radius: 999px; display: grid; place-items: center; font-size: 12px; font-weight: 700; color: #111; }
    .dot.rover { top: 18px; background: #f5f7fa; color: #111; }
    .dot.goal { top: 201px; background: var(--goal); }
    .card { background: var(--card); border: 1px solid var(--card-border); border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; }
    .cardTitle { font-size: 19px; font-weight: 600; margin-bottom: 2px; }
    .subtle { color: var(--muted); font-size: 13px; }
    .stats { margin-top: 10px; display: grid; gap: 10px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; }
    .row .label { color: var(--muted); }
    .footer { border-top: 1px solid #26292e; margin-top: 18px; padding-top: 16px; display: grid; gap: 14px; }
    .metricLabel { color: var(--muted); font-size: 13px; }
    .metricValue { font-size: 23px; font-weight: 600; line-height: 1.05; }
    .status { color: #9cc2ea; font-size: 14px; }
    .offline { position: absolute; inset: 18px auto auto 18px; background: rgba(20,20,20,0.75); border: 1px solid #444; border-radius: 999px; padding: 8px 12px; font-size: 13px; display: none; }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="title">Route</div>
    <div class="route">
      <div class="rail"></div>
      <div class="dot rover">R</div>
      <div class="dot goal">G</div>
      <div class="card">
        <div class="cardTitle">Rover</div>
        <div class="stats">
          <div class="row"><span class="label">Location</span><span id="roverLoc">--</span></div>
          <div class="row"><span class="label">Heading</span><span id="roverHeading">--</span></div>
          <div class="row"><span class="label">Distance to goal</span><span id="goalDist">--</span></div>
          <div class="row"><span class="label">Navigation time</span><span id="eta">--</span></div>
        </div>
      </div>
      <div class="card">
        <div class="cardTitle">Goal</div>
        <div class="subtle" id="goalLoc">--</div>
      </div>
    </div>
    <div class="footer">
      <div><div class="metricLabel">Run time</div><div id="runtime" class="metricValue">--</div></div>
      <div><div class="metricLabel">Traveled</div><div id="traveled" class="metricValue">--</div></div>
      <div><div class="metricLabel">Obstacles</div><div id="obstacles" class="metricValue">0</div></div>
      <div id="status" class="status">Starting...</div>
    </div>
  </aside>
  <div class="mapWrap">
    <canvas id="map"></canvas>
    <div id="offline" class="offline">Waiting for telemetry...</div>
  </div>
  <script>
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');
    const offline = document.getElementById('offline');
    const backgroundImage = new Image();
    backgroundImage.src = '/moon.jpg';
    let latestState = null;
    let previousState = null;
    let transitionStartMs = performance.now();
    let transitionDurationMs = 160;
    let lastPollMs = null;
    let avgPollIntervalMs = 160;
    let nextExpectedPollMs = null;
    let camera = null;
    let manualZoom = 1.0;
    let lastAutoFitMs = null;
    let dragState = null;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function fmtDuration(sec) {
      if (!isFinite(sec) || sec < 0) return '--';
      sec = Math.round(sec);
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = sec % 60;
      return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    function fmtDist(cm) {
      if (!isFinite(cm)) return '--';
      if (cm >= 100000) return `${(cm / 100000).toFixed(2)} km`;
      if (cm >= 100) return `${(cm / 100).toFixed(1)} m`;
      return `${cm.toFixed(0)} cm`;
    }

    function updateSidebar(state) {
      const goalText = state.goal_label || `${state.goal_xy[0].toFixed(1)}, ${state.goal_xy[1].toFixed(1)} cm`;
      document.getElementById('goalLoc').textContent = goalText;
      document.getElementById('roverLoc').textContent = `${state.raw_rover_xy[0].toFixed(1)}, ${state.raw_rover_xy[1].toFixed(1)}`;
      document.getElementById('roverHeading').textContent = `${state.heading_deg.toFixed(1)} deg`;
      document.getElementById('goalDist').textContent = fmtDist(state.goal_distance_cm);
      document.getElementById('eta').textContent = fmtDuration(state.eta_seconds);
      document.getElementById('runtime').textContent = fmtDuration(state.runtime_elapsed_s);
      document.getElementById('traveled').textContent = fmtDist(state.total_traveled_cm);
      document.getElementById('obstacles').textContent = String(state.obstacle_total);
      document.getElementById('status').textContent = state.status;
    }

    function goalKey(state) {
      return `${state.map_goal_xy[0].toFixed(2)}:${state.map_goal_xy[1].toFixed(2)}`;
    }

    function computeFitBounds(state, w, h) {
      const pts = [state.map_rover_xy, state.map_goal_xy];
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const [x, y] of pts) {
        if (!isFinite(x) || !isFinite(y)) continue;
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
      }
      if (!isFinite(minX)) return {minX: -1000, minY: -1000, maxX: 1000, maxY: 1000};
      const rawSpanX = Math.max(1e-6, maxX - minX);
      const rawSpanY = Math.max(1e-6, maxY - minY);
      const padX = Math.max(20, rawSpanX * 0.14);
      const padY = Math.max(20, rawSpanY * 0.14);
      minX -= padX;
      minY -= padY;
      maxX += padX;
      maxY += padY;
      const spanX = Math.max(1, maxX - minX);
      const spanY = Math.max(1, maxY - minY);
      const targetAspect = w / Math.max(h, 1);
      const spanAspect = spanX / spanY;
      if (spanAspect > targetAspect) {
        const desiredY = spanX / targetAspect;
        const extra = (desiredY - spanY) * 0.5;
        minY -= extra; maxY += extra;
      } else {
        const desiredX = spanY * targetAspect;
        const extra = (desiredX - spanX) * 0.5;
        minX -= extra; maxX += extra;
      }
      return {minX, minY, maxX, maxY};
    }

    function ensureCamera(state, w, h, nowMs) {
      const fitted = computeFitBounds(state, w, h);
      const baseSpanX = Math.max(1, fitted.maxX - fitted.minX);
      const baseSpanY = Math.max(1, fitted.maxY - fitted.minY);
      if (!camera) {
        camera = {
          centerX: (fitted.minX + fitted.maxX) * 0.5,
          centerY: (fitted.minY + fitted.maxY) * 0.5,
          spanX: baseSpanX * manualZoom,
          spanY: baseSpanY * manualZoom,
          lastGoalKey: goalKey(state),
        };
        lastAutoFitMs = nowMs;
        return;
      }

      const currentKey = goalKey(state);
      const shouldRefit = currentKey !== camera.lastGoalKey || lastAutoFitMs === null;
      if (shouldRefit) {
        const fittedCenterX = (fitted.minX + fitted.maxX) * 0.5;
        const fittedCenterY = (fitted.minY + fitted.maxY) * 0.5;
        camera.centerX = fittedCenterX;
        camera.centerY = fittedCenterY;
        camera.spanX = baseSpanX * manualZoom;
        camera.spanY = baseSpanY * manualZoom;
        lastAutoFitMs = nowMs;
      }
      camera.lastGoalKey = currentKey;
    }

    function getView() {
      if (!camera) return {minX: -1000, minY: -1000, maxX: 1000, maxY: 1000};
      return {
        minX: camera.centerX - camera.spanX * 0.5,
        minY: camera.centerY - camera.spanY * 0.5,
        maxX: camera.centerX + camera.spanX * 0.5,
        maxY: camera.centerY + camera.spanY * 0.5,
      };
    }

    function toScreen(view, x, y, w, h) {
      const sx = ((x - view.minX) / (view.maxX - view.minX)) * w;
      const sy = h - ((y - view.minY) / (view.maxY - view.minY)) * h;
      return [sx, sy];
    }

    function toWorld(view, sx, sy, w, h) {
      const worldX = view.minX + (sx / Math.max(w, 1)) * (view.maxX - view.minX);
      const worldY = view.minY + ((h - sy) / Math.max(h, 1)) * (view.maxY - view.minY);
      return [worldX, worldY];
    }

    function canvasPointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      return [event.clientX - rect.left, event.clientY - rect.top];
    }

    function drawBackgroundImage(w, h) {
      ctx.fillStyle = '#1e2023';
      ctx.fillRect(0, 0, w, h);
      if (!backgroundImage.complete || !backgroundImage.naturalWidth) return;
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.drawImage(backgroundImage, 0, 0, w, h);
      ctx.restore();
    }

    function drawDangerMarker(x, y, size = 22) {
      const h = size;
      const w = size * 0.94;
      const pts = [
        [x, y - h],
        [x + w, y + h * 0.72],
        [x - w, y + h * 0.72],
      ];
      ctx.beginPath();
      const radius = Math.max(4, size * 0.22);
      for (let i = 0; i < pts.length; i++) {
        const prev = pts[(i + pts.length - 1) % pts.length];
        const cur = pts[i];
        const next = pts[(i + 1) % pts.length];
        const toPrevX = prev[0] - cur[0];
        const toPrevY = prev[1] - cur[1];
        const toNextX = next[0] - cur[0];
        const toNextY = next[1] - cur[1];
        const prevLen = Math.max(1e-6, Math.hypot(toPrevX, toPrevY));
        const nextLen = Math.max(1e-6, Math.hypot(toNextX, toNextY));
        const startX = cur[0] + (toPrevX / prevLen) * radius;
        const startY = cur[1] + (toPrevY / prevLen) * radius;
        const endX = cur[0] + (toNextX / nextLen) * radius;
        const endY = cur[1] + (toNextY / nextLen) * radius;
        if (i === 0) ctx.moveTo(startX, startY);
        else ctx.lineTo(startX, startY);
        ctx.quadraticCurveTo(cur[0], cur[1], endX, endY);
      }
      ctx.closePath();
      ctx.fillStyle = '#e05c5c';
      ctx.fill();
      ctx.strokeStyle = '#ff8b8b';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.fillStyle = '#111214';
      ctx.font = '700 20px "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      const text = 'D';
      const metrics = ctx.measureText(text);
      const textX = x - ((metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight) * 0.5);
      const textY = y + (((metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) * 0.5)) + 2;
      ctx.fillText(text, textX, textY);
    }

    function applySoftShadow() {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.28)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 2;
    }

    function rotateBodyPoint(localX, localY, headingDeg) {
      const rad = headingDeg * Math.PI / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      const wx = localX * cos - localY * sin;
      const wy = localX * sin + localY * cos;
      return [wx, wy];
    }

    function mixColor(a, b, t) {
      const clampT = Math.max(0, Math.min(1, t));
      return [
        Math.round(a[0] + (b[0] - a[0]) * clampT),
        Math.round(a[1] + (b[1] - a[1]) * clampT),
        Math.round(a[2] + (b[2] - a[2]) * clampT),
      ];
    }

    function drawRoundedTriangle(x, y, headingDeg, stuckHeat) {
      ctx.save();
      applySoftShadow();
      const pts = [[28, 0], [-18, 22], [-18, -22]];
      const rotated = pts.map(([px, py]) => {
        const [wx, wy] = rotateBodyPoint(px, py, headingDeg);
        return [x + wx, y - wy];
      });
      ctx.beginPath();
      const radius = 7;
      for (let i = 0; i < rotated.length; i++) {
        const prev = rotated[(i + rotated.length - 1) % rotated.length];
        const cur = rotated[i];
        const next = rotated[(i + 1) % rotated.length];
        const v1x = prev[0] - cur[0], v1y = prev[1] - cur[1];
        const v2x = next[0] - cur[0], v2y = next[1] - cur[1];
        const l1 = Math.hypot(v1x, v1y), l2 = Math.hypot(v2x, v2y);
        const d = Math.min(radius, l1 * 0.45, l2 * 0.45);
        const p1 = [cur[0] + (v1x / l1) * d, cur[1] + (v1y / l1) * d];
        const p2 = [cur[0] + (v2x / l2) * d, cur[1] + (v2y / l2) * d];
        if (i === 0) ctx.moveTo(p1[0], p1[1]); else ctx.lineTo(p1[0], p1[1]);
        ctx.quadraticCurveTo(cur[0], cur[1], p2[0], p2[1]);
      }
      ctx.closePath();
      ctx.fillStyle = '#f5f7fa';
      ctx.fill();
      ctx.strokeStyle = '#5d636b';
      ctx.lineWidth = 2;
      ctx.stroke();

      const dotColor = mixColor([245, 247, 250], [230, 68, 68], stuckHeat);
      ctx.fillStyle = `rgb(${dotColor[0]}, ${dotColor[1]}, ${dotColor[2]})`;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(93, 99, 107, 0.18)';
      ctx.lineWidth = 1.0;
      ctx.stroke();
      ctx.restore();
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function lerpPoint(a, b, t) {
      return [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
    }

    function pathLengths(path) {
      const lengths = [0];
      for (let i = 1; i < path.length; i++) {
        const dx = path[i][0] - path[i - 1][0];
        const dy = path[i][1] - path[i - 1][1];
        lengths.push(lengths[lengths.length - 1] + Math.hypot(dx, dy));
      }
      return lengths;
    }

    function samplePath(path, lengths, distance) {
      if (path.length === 0) return [0, 0];
      if (path.length === 1) return path[0].slice();
      if (distance <= 0) return path[0].slice();
      const total = lengths[lengths.length - 1];
      if (distance >= total) return path[path.length - 1].slice();
      for (let i = 1; i < lengths.length; i++) {
        if (distance <= lengths[i]) {
          const span = Math.max(1e-6, lengths[i] - lengths[i - 1]);
          const t = (distance - lengths[i - 1]) / span;
          return lerpPoint(path[i - 1], path[i], t);
        }
      }
      return path[path.length - 1].slice();
    }

    function interpolatePath(a, b, t) {
      if (!a || a.length === 0) return (b || []).map((pt) => pt.slice());
      if (!b || b.length === 0) return a.map((pt) => pt.slice());
      const samples = Math.max(a.length, b.length, 24);
      const aLengths = pathLengths(a);
      const bLengths = pathLengths(b);
      const aTotal = aLengths[aLengths.length - 1];
      const bTotal = bLengths[bLengths.length - 1];
      const out = [];
      for (let i = 0; i < samples; i++) {
        const u = samples === 1 ? 0 : i / (samples - 1);
        const pa = samplePath(a, aLengths, aTotal * u);
        const pb = samplePath(b, bLengths, bTotal * u);
        out.push(lerpPoint(pa, pb, t));
      }
      return out;
    }

    function drawSmoothPath(points) {
      if (!points || points.length < 2) return;
      ctx.save();
      applySoftShadow();
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      if (points.length === 2) {
        ctx.lineTo(points[1][0], points[1][1]);
      } else {
        for (let i = 0; i < points.length - 1; i++) {
          const current = points[i];
          const next = points[i + 1];
          const midX = (current[0] + next[0]) * 0.5;
          const midY = (current[1] + next[1]) * 0.5;
          if (i === 0) {
            ctx.quadraticCurveTo(current[0], current[1], midX, midY);
          } else {
            ctx.quadraticCurveTo(current[0], current[1], midX, midY);
          }
        }
        const penultimate = points[points.length - 2];
        const last = points[points.length - 1];
        ctx.quadraticCurveTo(penultimate[0], penultimate[1], last[0], last[1]);
      }
      ctx.strokeStyle = '#6eaaff';
      ctx.lineWidth = 5;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();
    }

    function interpolateAngleDeg(a, b, t) {
      let delta = ((b - a + 540) % 360) - 180;
      return a + delta * t;
    }

    function updatePollTiming(nowMs) {
      if (lastPollMs !== null) {
        const sampleMs = Math.max(40, Math.min(1000, nowMs - lastPollMs));
        avgPollIntervalMs = (avgPollIntervalMs * 0.82) + (sampleMs * 0.18);
      }
      lastPollMs = nowMs;
      transitionDurationMs = Math.max(80, Math.min(320, avgPollIntervalMs));
      nextExpectedPollMs = nowMs + transitionDurationMs;
    }

    function getInterpolatedState(nowMs) {
      if (!latestState) return null;
      if (!previousState) return latestState;
      const elapsedMs = nowMs - transitionStartMs;
      const plannedDurationMs = nextExpectedPollMs === null
        ? transitionDurationMs
        : Math.max(1, nextExpectedPollMs - transitionStartMs);
      if (elapsedMs >= Math.max(plannedDurationMs * 1.25, 420)) {
        previousState = latestState;
        return latestState;
      }
      const t = Math.max(0, Math.min(1, elapsedMs / plannedDurationMs));
      return {
        ...latestState,
        rover_xy: lerpPoint(previousState.rover_xy, latestState.rover_xy, t),
        goal_xy: lerpPoint(previousState.goal_xy, latestState.goal_xy, t),
        target_xy: lerpPoint(previousState.target_xy, latestState.target_xy, t),
        heading_deg: interpolateAngleDeg(previousState.heading_deg, latestState.heading_deg, t),
        path_world: interpolatePath(previousState.path_world, latestState.path_world, t),
      };
    }

    function render(nowMs) {
      const w = canvas.getBoundingClientRect().width || 300;
      const h = canvas.getBoundingClientRect().height || 300;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#101215';
      ctx.fillRect(0, 0, w, h);
      const state = getInterpolatedState(nowMs);
      if (!state) {
        requestAnimationFrame(render);
        return;
      }

      ensureCamera(latestState || state, w, h, nowMs);
      updateSidebar(latestState || state);
      const view = getView();
      drawBackgroundImage(w, h);

      for (const [ox, oy, low] of (latestState || state).map_obstacles_world) {
        const [sx, sy] = toScreen(view, ox, oy, w, h);
        const cellPx = Math.max(3, Math.min(14, state.cell_size_cm * (w / (view.maxX - view.minX))));
        ctx.fillStyle = low ? 'rgba(215, 173, 79, 0.45)' : 'rgba(219, 81, 81, 0.28)';
        ctx.fillRect(sx - cellPx * 0.5, sy - cellPx * 0.5, cellPx, cellPx);
      }

      for (const [cx, cy] of (latestState || state).map_obstacle_chunks || []) {
        const [sx, sy] = toScreen(view, cx, cy, w, h);
        drawDangerMarker(sx, sy, 22);
      }

      if (state.map_path_world.length > 1) {
        const screenPath = state.map_path_world.map((pt) => toScreen(view, pt[0], pt[1], w, h));
        drawSmoothPath(screenPath);
      }

      const [gx, gy] = toScreen(view, state.map_goal_xy[0], state.map_goal_xy[1], w, h);
      const [rx, ry] = toScreen(view, state.map_rover_xy[0], state.map_rover_xy[1], w, h);

      ctx.save();
      applySoftShadow();
      ctx.fillStyle = '#f3d04e';
      ctx.beginPath();
      ctx.arc(gx, gy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      drawRoundedTriangle(rx, ry, state.heading_deg, state.stuck_heat);
      requestAnimationFrame(render);
    }

    async function poll() {
      try {
        const res = await fetch('/state', { cache: 'no-store' });
        if (!res.ok) throw new Error(String(res.status));
        const state = await res.json();
        const nowMs = performance.now();
        offline.style.display = 'none';
        updatePollTiming(nowMs);
        previousState = latestState;
        latestState = state;
        transitionStartMs = nowMs;
      } catch (err) {
        offline.style.display = 'block';
      }
    }

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      if (!camera) return;
      const rect = canvas.getBoundingClientRect();
      const canvasW = Math.max(rect.width, 1);
      const canvasH = Math.max(rect.height, 1);
      const beforeView = getView();
      const cursorX = event.clientX - rect.left;
      const cursorY = event.clientY - rect.top;
      const [worldXBefore, worldYBefore] = toWorld(beforeView, cursorX, cursorY, canvasW, canvasH);
      const zoomFactor = event.deltaY < 0 ? 0.88 : 1.14;
      manualZoom = Math.max(0.35, Math.min(6.0, manualZoom * zoomFactor));
      camera.spanX *= zoomFactor;
      camera.spanY *= zoomFactor;
      camera.centerX = worldXBefore - (cursorX / canvasW - 0.5) * camera.spanX;
      camera.centerY = worldYBefore - (0.5 - cursorY / canvasH) * camera.spanY;
      lastAutoFitMs = performance.now();
    }, { passive: false });

    canvas.addEventListener('pointerdown', (event) => {
      if (!camera) return;
      const [startX, startY] = canvasPointFromEvent(event);
      canvas.setPointerCapture(event.pointerId);
      dragState = {
        pointerId: event.pointerId,
        startX,
        startY,
        startCenterX: camera.centerX,
        startCenterY: camera.centerY,
      };
    });

    canvas.addEventListener('pointermove', (event) => {
      if (!camera || !dragState || dragState.pointerId !== event.pointerId) return;
      const [x, y] = canvasPointFromEvent(event);
      const rect = canvas.getBoundingClientRect();
      const canvasW = Math.max(rect.width, 1);
      const canvasH = Math.max(rect.height, 1);
      const dxPx = x - dragState.startX;
      const dyPx = y - dragState.startY;
      camera.centerX = dragState.startCenterX - (dxPx / canvasW) * camera.spanX;
      camera.centerY = dragState.startCenterY + (dyPx / canvasH) * camera.spanY;
      lastAutoFitMs = performance.now();
    });

    function endDrag(event) {
      if (!dragState || dragState.pointerId !== event.pointerId) return;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      dragState = null;
    }

    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    window.addEventListener('resize', resize);
    resize();
    poll();
    setInterval(poll, 150);
    requestAnimationFrame(render);
  </script>
</body>
</html>
"""


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class HtmlCanvasWindow:
    def __init__(self, planner) -> None:
        self._state_lock = threading.Lock()
        self._state: dict[str, object] = {
            "goal_xy": [0.0, 0.0],
            "target_xy": [0.0, 0.0],
            "rover_xy": [0.0, 0.0],
            "raw_rover_xy": [0.0, 0.0],
            "map_goal_xy": [0.0, 0.0],
            "map_rover_xy": [0.0, 0.0],
            "heading_deg": 0.0,
            "path_world": [],
            "map_path_world": [],
            "map_obstacles_world": [],
            "map_obstacle_chunks": [],
            "cell_size_cm": float(planner.config.cell_size_cm),
            "runtime_elapsed_s": 0.0,
            "total_traveled_cm": 0.0,
            "waypoint_distance_cm": 0.0,
            "goal_distance_cm": 0.0,
            "eta_seconds": float("nan"),
            "obstacle_total": 0,
            "throttle_cmd": 0.0,
            "steering_cmd": 0.0,
            "stuck_heat": 0.0,
            "reverse_active": False,
            "status": "Starting...",
        }
        self._cached_planner_id: int | None = None
        self._cached_obstacle_version: int = -1
        self._cached_map_obstacles_world: list[list[float | int]] = []
        self._cached_map_obstacle_chunks: list[list[float]] = []
        self._cached_obstacle_chunk_count: int = 0
        self.view_center_x_cm = 0.0
        self.view_center_y_cm = 0.0
        self._closed = False
        self._port = _pick_free_port()
        self.url = f"http://127.0.0.1:{self._port}/"
        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Frontend URL: {self.url}")
        webbrowser.open_new(self.url)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/moon.jpg":
                    with open("moon.jpg", "rb") as f:
                        body = f.read()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/state":
                    with outer._state_lock:
                        body = json.dumps(outer._state, allow_nan=True).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler

    @staticmethod
    def _estimate_eta_seconds(goal_distance_cm: float, runtime_elapsed_s: float, total_traveled_cm: float) -> float:
        if runtime_elapsed_s <= 1e-6 or total_traveled_cm <= 1e-6:
            return float("nan")
        avg_speed_cm_s = total_traveled_cm / runtime_elapsed_s
        if avg_speed_cm_s <= 1e-6:
            return float("nan")
        return goal_distance_cm / avg_speed_cm_s

    @staticmethod
    def _serialize_obstacles(planner) -> list[list[float | int]]:
        obstacles: list[list[float | int]] = []
        for y in range(planner.height_cells):
            for x in range(planner.width_cells):
                cell_value = planner.grid[y][x]
                if cell_value == 0:
                    continue
                wx, wy = planner.cell_to_world_center((x, y))
                obstacles.append([float(wx), float(wy), 1 if int(cell_value) == 2 else 0])
        return obstacles

    @staticmethod
    def _local_x_cm_to_raw(local_x_cm: float) -> float:
        return (float(local_x_cm) + float(POSE_OFFSET_X_CM)) / float(POSE_UNITS_TO_CM)

    @staticmethod
    def _local_y_cm_to_raw(local_y_cm: float) -> float:
        return (float(local_y_cm) + float(POSE_OFFSET_Y_CM)) / float(POSE_UNITS_TO_CM)

    @classmethod
    def _local_xy_to_raw(cls, xy: tuple[float, float]) -> list[float]:
        return [cls._local_x_cm_to_raw(xy[0]), cls._local_y_cm_to_raw(xy[1])]

    @classmethod
    def _serialize_path_raw(cls, path_world: list[tuple[float, float]]) -> list[list[float]]:
        return [cls._local_xy_to_raw(point) for point in path_world]

    @classmethod
    def _serialize_obstacles_raw(cls, planner) -> list[list[float | int]]:
        obstacles: list[list[float | int]] = []
        for obstacle_x_cm, obstacle_y_cm, low_flag in cls._serialize_obstacles(planner):
            obstacles.append(
                [
                    cls._local_x_cm_to_raw(float(obstacle_x_cm)),
                    cls._local_y_cm_to_raw(float(obstacle_y_cm)),
                    int(low_flag),
                ]
            )
        return obstacles

    @classmethod
    def _serialize_chunk_centers_raw(cls, chunk_centers: list[tuple[float, float]]) -> list[list[float]]:
        return [cls._local_xy_to_raw(center) for center in chunk_centers]

    def _get_cached_obstacle_payload(self, planner) -> tuple[list[list[float | int]], list[list[float]], int]:
        planner_id = id(planner)
        obstacle_version = int(getattr(planner, "_obstacle_version", -1))
        if planner_id != self._cached_planner_id or obstacle_version != self._cached_obstacle_version:
            obstacle_chunks = self._compute_obstacle_chunks(planner)
            self._cached_map_obstacles_world = self._serialize_obstacles_raw(planner)
            self._cached_map_obstacle_chunks = self._serialize_chunk_centers_raw(obstacle_chunks)
            self._cached_obstacle_chunk_count = len(obstacle_chunks)
            self._cached_planner_id = planner_id
            self._cached_obstacle_version = obstacle_version
        return (
            list(self._cached_map_obstacles_world),
            list(self._cached_map_obstacle_chunks),
            int(self._cached_obstacle_chunk_count),
        )

    @staticmethod
    def _count_obstacle_chunks(planner, merge_distance_cm: float = 700.0) -> int:
        return len(HtmlCanvasWindow._compute_obstacle_chunks(planner, merge_distance_cm=merge_distance_cm))

    @staticmethod
    def _compute_obstacle_chunks(planner, merge_distance_cm: float = 700.0) -> list[tuple[float, float]]:
        occupied: list[tuple[int, int]] = []
        for y in range(planner.height_cells):
            for x in range(planner.width_cells):
                if int(planner.grid[y][x]) != 0:
                    occupied.append((x, y))

        if not occupied:
            return []

        merge_radius_cells = max(
            1.0,
            float(merge_distance_cm) / max(1.0, float(planner.config.cell_size_cm)),
        )
        merge_radius_sq = merge_radius_cells * merge_radius_cells
        visited = [False] * len(occupied)
        chunk_centers: list[tuple[float, float]] = []

        for start_idx, _start_cell in enumerate(occupied):
            if visited[start_idx]:
                continue
            stack = [start_idx]
            visited[start_idx] = True
            chunk_cells: list[tuple[int, int]] = []
            while stack:
                current_idx = stack.pop()
                cx, cy = occupied[current_idx]
                chunk_cells.append((cx, cy))
                for neighbor_idx, (nx, ny) in enumerate(occupied):
                    if visited[neighbor_idx]:
                        continue
                    dx = float(nx - cx)
                    dy = float(ny - cy)
                    if (dx * dx) + (dy * dy) <= merge_radius_sq:
                        visited[neighbor_idx] = True
                        stack.append(neighbor_idx)
            if chunk_cells:
                world_points = [planner.cell_to_world_center(cell) for cell in chunk_cells]
                avg_x = sum(point[0] for point in world_points) / len(world_points)
                avg_y = sum(point[1] for point in world_points) / len(world_points)
                chunk_centers.append((float(avg_x), float(avg_y)))
        return chunk_centers

    def _update_scale(self, planner) -> None:
        return

    def get_state_snapshot(self) -> dict[str, object]:
        with self._state_lock:
            return copy.deepcopy(self._state)

    def set_state_snapshot(self, state: dict[str, object]) -> None:
        with self._state_lock:
            self._state = copy.deepcopy(state)

    def draw(
        self,
        planner,
        rover_xy: tuple[float, float],
        heading_deg: float,
        goal_xy: tuple[float, float],
        target_xy: tuple[float, float],
        path_world: list[tuple[float, float]],
        status: str,
        goal_distance_cm: float,
        throttle_cmd: float,
        steering_cmd: float,
        waypoint_idx: int,
        waypoint_distance_cm: float,
        waypoint_distance_avg_cm: float,
        obstacle_total: int,
        lidar_cm,
        lidar_debug_rows=None,
        goal_label: str | None = None,
        runtime_elapsed_s: float = 0.0,
        total_traveled_cm: float = 0.0,
        stationary_elapsed_s: float = 0.0,
        reverse_active: bool = False,
        raw_rover_xy: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        eta_seconds = self._estimate_eta_seconds(goal_distance_cm, runtime_elapsed_s, total_traveled_cm)
        map_obstacles_world, map_obstacle_chunks, obstacle_chunk_count = self._get_cached_obstacle_payload(planner)
        stuck_heat = 0.0 if reverse_active else max(0.0, min(1.0, float(stationary_elapsed_s) / 7.0))
        with self._state_lock:
            map_goal_xy = self._local_xy_to_raw(goal_xy)
            map_rover_xy = [float(raw_rover_xy[0]), float(raw_rover_xy[1])]
            map_path_world = self._serialize_path_raw(path_world)
            self._state = {
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "target_xy": [float(target_xy[0]), float(target_xy[1])],
                "rover_xy": [float(rover_xy[0]), float(rover_xy[1])],
                "raw_rover_xy": [float(raw_rover_xy[0]), float(raw_rover_xy[1])],
                "map_goal_xy": map_goal_xy,
                "map_rover_xy": map_rover_xy,
                "heading_deg": float(heading_deg),
                "path_world": [[float(x), float(y)] for x, y in path_world],
                "map_path_world": map_path_world,
                "map_obstacles_world": map_obstacles_world,
                "map_obstacle_chunks": map_obstacle_chunks,
                "cell_size_cm": float(planner.config.cell_size_cm),
                "runtime_elapsed_s": float(runtime_elapsed_s),
                "total_traveled_cm": float(total_traveled_cm),
                "waypoint_distance_cm": float(waypoint_distance_cm),
                "goal_distance_cm": float(goal_distance_cm),
                "eta_seconds": float(eta_seconds),
                "obstacle_total": int(obstacle_chunk_count),
                "throttle_cmd": float(throttle_cmd),
                "steering_cmd": float(steering_cmd),
                "goal_label": None if goal_label is None else str(goal_label),
                "stuck_heat": float(stuck_heat),
                "reverse_active": bool(reverse_active),
                "status": str(status),
            }
        return not self._closed

    def close(self) -> None:
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)
