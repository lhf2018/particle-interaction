const canvas = document.getElementById("sim");
const hud = document.getElementById("hud");
const ctx =
  canvas.getContext("2d", { alpha: true, desynchronized: true }) ||
  canvas.getContext("2d");

const state = {
  width: 0,
  height: 0,
  count: 0,
  forceStrength: 76000,
  paused: false,
  positions: new Float32Array(0),
  mouse: { x: 0, y: 0, left: false, right: false },
  ws: null,
  connected: false,
  lastTs: performance.now(),
  lastRecvTs: performance.now(),
  fps: 0,
  streamFps: 0,
  gpuMemUsed: 0,
  gpuMemTotal: 0,
  backend: "unknown",
  wsError: "",
  simWidth: 1,
  simHeight: 1,
  streamPoints: 0,
  targetFps: 20,
  obstacles: [],
  obstaclesScreenSpace: false,
  trailCanvas: null,
  trailCtx: null,
  mode: "2d",
  enableTrail: false,
  stars: [],
  phase: 0,
};

function resize() {
  state.width = canvas.width = window.innerWidth;
  state.height = canvas.height = window.innerHeight;
  buildStars();
  sendConfig({ width: state.width, height: state.height });
}

function buildStars() {
  const n = state.mode === "3d" ? 180 : 90;
  state.stars = Array.from({ length: n }, () => ({
    x: Math.random() * state.width,
    y: Math.random() * state.height * 0.8,
    b: 120 + Math.random() * 120,
    s: 0.4 + Math.random() * 1.2,
  }));
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const wsPath = state.mode === "3d" ? "/ws/sim3d" : "/ws/sim";
  state.ws = new WebSocket(`${protocol}://${location.host}${wsPath}`);
  state.ws.binaryType = "arraybuffer";

  state.ws.onopen = () => {
    state.connected = true;
    sendConfig({
      width: state.width,
      height: state.height,
      force: state.forceStrength,
      paused: state.paused,
      stream_points: state.mode === "3d" ? 14000 : 22000,
      target_fps: state.mode === "3d" ? 20 : 24,
    });
  };

  state.ws.onmessage = (event) => {
    try {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        if (msg.type === "ready" || msg.type === "stats") {
          const s = msg.stats || {};
          state.count = s.count ?? state.count;
          state.forceStrength = s.force ?? state.forceStrength;
          state.paused = s.paused ?? state.paused;
          state.gpuMemUsed = s.gpu_mem_used_mb ?? state.gpuMemUsed;
          state.gpuMemTotal = s.gpu_mem_total_mb ?? state.gpuMemTotal;
          state.backend = s.backend ?? state.backend;
          state.streamPoints = s.stream_points ?? state.streamPoints;
          state.targetFps = s.target_fps ?? state.targetFps;
          state.obstacles = s.obstacles ?? state.obstacles;
          state.obstaclesScreenSpace = s.obstacles_screen_space ?? false;
          state.wsError = "";
        }
        return;
      }

      const buffer = event.data;
      if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) return;
      const view = new DataView(buffer);
      const count = view.getUint32(0, true);
      const simW = view.getUint16(4, true);
      const simH = view.getUint16(6, true);
      const expectedBytes = 8 + count * 2 * 2;
      if (expectedBytes !== buffer.byteLength || count <= 0 || count > 100000) {
        return;
      }
      const packed = new Uint16Array(buffer, 8, count * 2);
      const arr = new Float32Array(count * 2);
      const scaleX = state.width / Math.max(1, simW - 1);
      const scaleY = state.height / Math.max(1, simH - 1);
      for (let i = 0; i < count; i++) {
        const base = i * 2;
        arr[base] = (packed[base] / 65535) * (simW - 1) * scaleX;
        arr[base + 1] = (packed[base + 1] / 65535) * (simH - 1) * scaleY;
      }
      state.count = count;
      state.simWidth = simW;
      state.simHeight = simH;
      state.positions = arr;
      const now = performance.now();
      const dt = (now - state.lastRecvTs) / 1000;
      if (dt > 0) state.streamFps = 0.9 * state.streamFps + 0.1 * (1 / dt);
      state.lastRecvTs = now;
      state.wsError = "";
    } catch (err) {
      state.wsError = `Frame parse error: ${err?.message || err}`;
    }
  };

  state.ws.onclose = () => {
    state.connected = false;
    setTimeout(connect, 1200);
  };
  state.ws.onerror = () => {
    state.wsError = "WebSocket failed, check backend logs";
  };
}

function sendInput() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const mode = state.mouse.left ? -1 : state.mouse.right ? 1 : 0;
  state.ws.send(
    JSON.stringify({
      type: "input",
      x: state.mouse.x,
      y: state.mouse.y,
      mode,
    }),
  );
}

function sendConfig(extra = {}) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(
    JSON.stringify({
      type: "config",
      ...extra,
    }),
  );
}

function drawBackground() {
  if (state.mode === "3d") {
    const g = ctx.createLinearGradient(0, 0, 0, state.height);
    g.addColorStop(0, "#060816");
    g.addColorStop(0.42, "#120a2f");
    g.addColorStop(1, "#1f123e");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, state.width, state.height);

    const horizon = state.height * 0.5;
    for (let i = 0; i < state.stars.length; i++) {
      const st = state.stars[i];
      const tw = 0.6 + 0.4 * Math.sin(state.phase * 2.2 + i * 0.13);
      const c = Math.min(255, st.b * tw);
      ctx.fillStyle = `rgba(${c},${Math.min(255, c * 0.8)},255,0.92)`;
      ctx.fillRect(st.x, st.y, st.s, st.s);
    }

    // Perspective neon grid helps depth perception.
    ctx.strokeStyle = "rgba(130,100,255,0.34)";
    ctx.lineWidth = 1;
    for (let i = -14; i <= 14; i++) {
      ctx.beginPath();
      ctx.moveTo(state.width * 0.5 + i * 85, state.height);
      ctx.lineTo(state.width * 0.5 + i * 12, horizon);
      ctx.stroke();
    }
    for (let j = 1; j <= 9; j++) {
      const y = horizon + ((state.height - horizon) * (j / 10) ** 1.6);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(state.width, y);
      ctx.stroke();
    }
    return;
  }

  const g = ctx.createLinearGradient(0, 0, 0, state.height);
  g.addColorStop(0, "#d6ecff");
  g.addColorStop(0.45, "#9dcbff");
  g.addColorStop(1, "#6fa7e5");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, state.width, state.height);

  const horizon = state.height * 0.52;
  ctx.strokeStyle = "rgba(207,230,255,0.22)";
  ctx.lineWidth = 1;
  for (let i = -12; i <= 12; i++) {
    ctx.beginPath();
    ctx.moveTo(state.width * 0.5 + i * 72, state.height);
    ctx.lineTo(state.width * 0.5 + i * 10, horizon);
    ctx.stroke();
  }
}

function drawObstacles() {
  for (const o of state.obstacles) {
    const x = state.obstaclesScreenSpace ? o.x : o.x * (state.width / Math.max(1, state.simWidth));
    const y = state.obstaclesScreenSpace ? o.y : o.y * (state.height / Math.max(1, state.simHeight));
    const r = state.obstaclesScreenSpace ? o.r : o.r * (state.width / Math.max(1, state.simWidth));
    const pulse = 0.75 + 0.25 * Math.sin(state.phase * 3.0 + x * 0.01);
    const grd = ctx.createRadialGradient(x - r * 0.25, y - r * 0.25, r * 0.15, x, y, r);
    grd.addColorStop(0, state.mode === "3d" ? `rgba(230,245,255,${0.42 * pulse})` : "rgba(210,245,255,0.35)");
    grd.addColorStop(1, state.mode === "3d" ? "rgba(90,120,255,0.10)" : "rgba(70,140,220,0.10)");
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = state.mode === "3d" ? `rgba(145,205,255,${0.95 * pulse})` : "rgba(150,220,255,0.85)";
    ctx.lineWidth = state.mode === "3d" ? 2.4 : 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
    if (state.mode === "3d") {
      ctx.strokeStyle = `rgba(120,135,255,${0.6 * pulse})`;
      ctx.beginPath();
      ctx.arc(x, y, r * 1.18, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function drawParticles() {
  const px = state.positions;
  const drawCount = Math.min(state.count, px.length / 2);
  const tctx = state.enableTrail ? state.trailCtx || ctx : ctx;
  if (state.enableTrail && state.trailCtx) {
    // Fade old trails with dark alpha to avoid full-screen white washout.
    tctx.globalCompositeOperation = "source-over";
    tctx.fillStyle = "rgba(0,0,0,0.14)";
    tctx.fillRect(0, 0, state.width, state.height);
  }
  for (let i = 0; i < drawCount; i++) {
    const idx = i * 2;
    const x = px[idx];
    const y = px[idx + 1];
    const t = Math.min(Math.abs((y / Math.max(1, state.height)) - 0.5) * 1.2, 1);
    if (state.mode === "3d") {
      const hueWave = 0.5 + 0.5 * Math.sin(state.phase * 2.1 + x * 0.014);
      const rr = Math.floor(180 + (1 - t) * 70 + hueWave * 35);
      const gg = Math.floor(70 + (1 - t) * 90);
      const bb = Math.floor(255 - t * 60 + (1 - hueWave) * 35);
      tctx.fillStyle = `rgba(${rr},${gg},${bb},0.9)`;
      tctx.beginPath();
      tctx.arc(x, y, 2.0, 0, Math.PI * 2);
      tctx.fill();
    } else {
      const rr = Math.floor(90 + (1 - t) * 140);
      const gg = Math.floor(130 + (1 - t) * 70);
      const bb = Math.floor(255 - t * 120);
      tctx.fillStyle = `rgba(${rr},${gg},${bb},0.78)`;
      tctx.fillRect(x, y, 2, 2);
    }
  }
  if (state.enableTrail && state.trailCanvas) {
    ctx.drawImage(state.trailCanvas, 0, 0);
    if (state.mode === "3d") {
      ctx.globalCompositeOperation = "lighter";
      ctx.filter = "blur(2.1px)";
      ctx.drawImage(state.trailCanvas, 0, 0);
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.filter = "none";
  }
}

function updateHUD() {
  hud.textContent =
    `FluidVibe Web (CUDA backend)\n` +
    `Connection: ${state.connected ? "WS connected" : "reconnecting..."}\n` +
    `Mode: ${state.mode.toUpperCase()}\n` +
    `Backend: ${state.backend}\n` +
    `FPS: ${state.fps.toFixed(1)}\n` +
    `Stream FPS: ${state.streamFps.toFixed(1)}\n` +
    `Particles: ${state.count} (stream ${state.streamPoints || state.count})\n` +
    `Target stream: ${state.targetFps} fps\n` +
    `Obstacles: ${state.obstacles.length}\n` +
    `GPU Mem: ${state.gpuMemUsed.toFixed(0)}/${state.gpuMemTotal.toFixed(0)} MB\n` +
    `Force: ${state.forceStrength.toFixed(0)}\n` +
    `Mouse: L repel | R attract | Wheel force | Space pause | R reset` +
    (state.wsError ? `\nError: ${state.wsError}` : "");
}

function frame(ts) {
  if (!ctx) {
    hud.textContent = "Canvas 2D context unavailable in this browser session.";
    return;
  }
  const dt = Math.min((ts - state.lastTs) / 1000, 1 / 30);
  state.lastTs = ts;
  state.phase += dt;
  state.fps = 0.9 * state.fps + 0.1 * (1 / Math.max(dt, 1e-6));
  try {
    sendInput();
    drawBackground();
    drawObstacles();
    drawParticles();
    updateHUD();
  } catch (err) {
    state.wsError = `Render error: ${err?.message || err}`;
    updateHUD();
  }
  requestAnimationFrame(frame);
}

canvas.addEventListener("mousemove", (e) => {
  state.mouse.x = e.clientX;
  state.mouse.y = e.clientY;
});
canvas.addEventListener("mousedown", (e) => {
  if (e.button === 0) state.mouse.left = true;
  if (e.button === 2) state.mouse.right = true;
});
canvas.addEventListener("mouseup", (e) => {
  if (e.button === 0) state.mouse.left = false;
  if (e.button === 2) state.mouse.right = false;
});
canvas.addEventListener("wheel", (e) => {
  state.forceStrength *= e.deltaY < 0 ? 1.1 : 1 / 1.1;
  state.forceStrength = Math.max(6000, Math.min(220000, state.forceStrength));
  sendConfig({ force: state.forceStrength });
  e.preventDefault();
}, { passive: false });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    state.paused = !state.paused;
    sendConfig({ paused: state.paused });
  }
  if (e.key.toLowerCase() === "r") sendConfig({ reset: true });
});
window.addEventListener("resize", () => {
  resize();
  state.trailCanvas = document.createElement("canvas");
  state.trailCanvas.width = state.width;
  state.trailCanvas.height = state.height;
  state.trailCtx =
    state.trailCanvas.getContext("2d", { alpha: true, desynchronized: true }) ||
    state.trailCanvas.getContext("2d");
});

const params = new URLSearchParams(location.search);
state.mode = params.get("mode") === "3d" ? "3d" : "2d";
state.enableTrail = state.mode === "3d";

resize();
state.trailCanvas = document.createElement("canvas");
state.trailCanvas.width = state.width;
state.trailCanvas.height = state.height;
state.trailCtx =
  state.trailCanvas.getContext("2d", { alpha: true, desynchronized: true }) ||
  state.trailCanvas.getContext("2d");
connect();
requestAnimationFrame(frame);
