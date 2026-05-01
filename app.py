import asyncio
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except Exception:
    cp = None
    CUDA_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="FluidVibe Web Service", version="1.0.0")

app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web/index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class GPUSimulator2D:
    def __init__(self, width: int, height: int, count: int = 60_000) -> None:
        self.width = float(width)
        self.height = float(height)
        self.count = int(max(5_000, min(120_000, count)))
        self.target_fps = 30 if CUDA_AVAILABLE else 20
        self.max_stream_points = 24_000 if CUDA_AVAILABLE else 12_000
        self.gravity = 580.0
        self.damping = 0.996
        self.bounce = 0.68
        self.max_speed = 1200.0
        self.floor_band = 90.0
        self.floor_push = 900.0
        self.mouse_strength = 76_000.0
        self.paused = False
        self.mouse_x = self.width * 0.5
        self.mouse_y = self.height * 0.5
        self.mouse_mode = 0
        self.obstacle_bounce = 0.82
        self._setup_obstacles()
        self._init_particles()

    def _setup_obstacles(self) -> None:
        xp = cp if CUDA_AVAILABLE else np
        self.obstacle_centers = xp.array(
            [
                [self.width * 0.32, self.height * 0.62],
                [self.width * 0.56, self.height * 0.45],
                [self.width * 0.76, self.height * 0.68],
            ],
            dtype=xp.float32,
        )
        self.obstacle_radii = xp.array(
            [min(self.width, self.height) * 0.08, min(self.width, self.height) * 0.065, min(self.width, self.height) * 0.055],
            dtype=xp.float32,
        )

    def _init_particles(self) -> None:
        n = self.count
        cx = self.width * 0.5
        cy = self.height * 0.34
        radius = min(self.width, self.height) * 0.26
        if CUDA_AVAILABLE:
            r = cp.sqrt(cp.random.random(n, dtype=cp.float32)) * cp.float32(radius)
            t = cp.random.random(n, dtype=cp.float32) * cp.float32(2.0 * np.pi)
            self.pos = cp.empty((n, 2), dtype=cp.float32)
            self.vel = cp.random.normal(0.0, 20.0, size=(n, 2)).astype(cp.float32)
            self.pos[:, 0] = cp.float32(cx) + cp.cos(t) * r
            self.pos[:, 1] = cp.float32(cy) + cp.sin(t) * r
        else:
            r = np.sqrt(np.random.random(n).astype(np.float32)) * np.float32(radius)
            t = np.random.random(n).astype(np.float32) * np.float32(2.0 * np.pi)
            self.pos = np.empty((n, 2), dtype=np.float32)
            self.vel = np.random.normal(0.0, 20.0, size=(n, 2)).astype(np.float32)
            self.pos[:, 0] = np.float32(cx) + np.cos(t) * r
            self.pos[:, 1] = np.float32(cy) + np.sin(t) * r

    def configure(self, payload: dict) -> None:
        xp = cp if CUDA_AVAILABLE else np
        if "width" in payload and "height" in payload:
            new_w = max(200, int(payload["width"]))
            new_h = max(200, int(payload["height"]))
            sx = new_w / max(1.0, self.width)
            sy = new_h / max(1.0, self.height)
            self.pos[:, 0] *= xp.float32(sx)
            self.pos[:, 1] *= xp.float32(sy)
            self.width = float(new_w)
            self.height = float(new_h)
            self._setup_obstacles()
        if "force" in payload:
            self.mouse_strength = float(max(6_000, min(220_000, float(payload["force"]))))
        if "stream_points" in payload:
            self.max_stream_points = int(max(3_000, min(50_000, int(payload["stream_points"]))))
        if "target_fps" in payload:
            self.target_fps = int(max(12, min(50, int(payload["target_fps"]))))
        if "paused" in payload:
            self.paused = bool(payload["paused"])
        if payload.get("reset"):
            self._init_particles()

    def update_mouse(self, payload: dict) -> None:
        self.mouse_x = float(payload.get("x", self.mouse_x))
        self.mouse_y = float(payload.get("y", self.mouse_y))
        self.mouse_mode = int(payload.get("mode", self.mouse_mode))

    def _apply_mouse(self, dt: float) -> None:
        if self.mouse_mode == 0:
            return
        xp = cp if CUDA_AVAILABLE else np
        mouse = xp.array([self.mouse_x, self.mouse_y], dtype=xp.float32)
        diff = mouse - self.pos
        dist2 = xp.sum(diff * diff, axis=1) + xp.float32(42.0)
        inv = xp.reciprocal(xp.sqrt(dist2))
        direction = diff * inv[:, None]
        force = xp.float32(self.mouse_strength * self.mouse_mode) / dist2
        self.vel += direction * force[:, None] * xp.float32(dt * 120.0)

    def step(self, dt: float) -> None:
        if self.paused:
            return
        xp = cp if CUDA_AVAILABLE else np
        self._apply_mouse(dt)
        self.vel[:, 1] += xp.float32(self.gravity * dt)
        floor_depth = self.pos[:, 1] - xp.float32(self.height - self.floor_band)
        in_floor = floor_depth > 0.0
        if xp.any(in_floor):
            self.vel[:, 1] -= xp.where(
                in_floor, floor_depth / xp.float32(self.floor_band), xp.float32(0.0)
            ) * xp.float32(self.floor_push * dt)
            wave = xp.sin(self.pos[:, 0] * xp.float32(0.01) + xp.float32(time.time() * 2.0))
            self.vel[:, 0] += xp.where(in_floor, wave, xp.float32(0.0)) * xp.float32(70.0 * dt)
        self.vel *= xp.float32(self.damping)

        speed = xp.sqrt(xp.sum(self.vel * self.vel, axis=1)) + xp.float32(1e-6)
        over = speed > xp.float32(self.max_speed)
        if xp.any(over):
            scale = xp.where(over, xp.float32(self.max_speed) / speed, xp.float32(1.0))
            self.vel *= scale[:, None]

        self.pos += self.vel * xp.float32(dt)
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        vx = self.vel[:, 0]
        vy = self.vel[:, 1]

        left = x < 0.0
        right = x > (self.width - 1)
        top = y < 0.0
        bottom = y > (self.height - 1)
        x[left] = 0.0
        x[right] = self.width - 1
        y[top] = 0.0
        y[bottom] = self.height - 1
        vx[left | right] *= xp.float32(-self.bounce)
        vy[top | bottom] *= xp.float32(-self.bounce)

        # Obstacle collisions (circle structures).
        for i in range(int(self.obstacle_centers.shape[0])):
            center = self.obstacle_centers[i]
            radius = self.obstacle_radii[i]
            diff = self.pos - center
            dist = xp.sqrt(xp.sum(diff * diff, axis=1)) + xp.float32(1e-6)
            inside = dist < radius
            if not xp.any(inside):
                continue
            normal = diff / dist[:, None]
            self.pos[inside] = center + normal[inside] * radius
            vdot = xp.sum(self.vel * normal, axis=1)
            reflected = self.vel - (xp.float32(1.0) + xp.float32(self.obstacle_bounce)) * vdot[:, None] * normal
            self.vel[inside] = reflected[inside]

    def payload_binary(self) -> bytes:
        # Stream a sampled, quantized frame (uint16) to reduce bandwidth and stutter.
        if self.count <= self.max_stream_points:
            sampled = self.pos
            out_count = self.count
        else:
            stride = max(1, self.count // self.max_stream_points)
            sampled = self.pos[::stride]
            out_count = int(sampled.shape[0])

        if CUDA_AVAILABLE:
            arr = cp.asnumpy(sampled).astype(np.float32, copy=False)
        else:
            arr = sampled.astype(np.float32, copy=False)

        arr = np.ascontiguousarray(arr)
        x = np.clip(arr[:, 0], 0.0, max(1.0, self.width - 1.0))
        y = np.clip(arr[:, 1], 0.0, max(1.0, self.height - 1.0))
        sx = np.float32(65535.0 / max(1.0, self.width - 1.0))
        sy = np.float32(65535.0 / max(1.0, self.height - 1.0))
        packed = np.empty((out_count, 2), dtype=np.uint16)
        packed[:, 0] = (x * sx).astype(np.uint16)
        packed[:, 1] = (y * sy).astype(np.uint16)

        # Header: [count:uint32, sim_w:uint16, sim_h:uint16]
        header = np.array(
            [out_count, int(self.width), int(self.height)],
            dtype=np.uint32,
        )
        header_bytes = (
            np.array([header[0]], dtype=np.uint32).tobytes()
            + np.array([header[1], header[2]], dtype=np.uint16).tobytes()
        )
        return header_bytes + packed.tobytes()

    def stats(self) -> dict:
        if CUDA_AVAILABLE:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mb = round((total_mem - free_mem) / (1024 * 1024), 1)
            total_mb = round(total_mem / (1024 * 1024), 1)
            backend = "cuda"
        else:
            used_mb = 0.0
            total_mb = 0.0
            backend = "cpu"
        return {
            "count": self.count,
            "stream_points": int(min(self.count, self.max_stream_points)),
            "target_fps": self.target_fps,
            "force": self.mouse_strength,
            "paused": self.paused,
            "gpu_mem_used_mb": used_mb,
            "gpu_mem_total_mb": total_mb,
            "backend": backend,
            "obstacles": self._obstacle_payload(),
        }

    def _obstacle_payload(self) -> list[dict]:
        if CUDA_AVAILABLE:
            centers = cp.asnumpy(self.obstacle_centers)
            radii = cp.asnumpy(self.obstacle_radii)
        else:
            centers = self.obstacle_centers
            radii = self.obstacle_radii
        out = []
        for i in range(len(radii)):
            out.append(
                {
                    "x": float(centers[i, 0]),
                    "y": float(centers[i, 1]),
                    "r": float(radii[i]),
                }
            )
        return out


class GPUSimulator3D:
    def __init__(self, width: int, height: int, count: int = 70_000) -> None:
        self.width = float(width)
        self.height = float(height)
        self.count = int(max(6_000, min(140_000, count)))
        self.target_fps = 28 if CUDA_AVAILABLE else 18
        self.max_stream_points = 22_000 if CUDA_AVAILABLE else 10_000
        self.mouse_strength = 5200.0
        self.damping = 0.994
        self.max_speed = 760.0
        self.gravity = 150.0
        self.world_radius = 300.0
        self.paused = False
        self.mouse_x = self.width * 0.5
        self.mouse_y = self.height * 0.5
        self.mouse_mode = 0
        self.time_acc = 0.0
        self.obstacle_bounce = 0.82
        self._setup_obstacles()
        self._init_particles()

    def _setup_obstacles(self) -> None:
        xp = cp if CUDA_AVAILABLE else np
        self.obstacle_centers = xp.array(
            [[-100.0, 30.0, 60.0], [90.0, -40.0, -80.0], [10.0, 90.0, 20.0]],
            dtype=xp.float32,
        )
        self.obstacle_radii = xp.array([56.0, 50.0, 44.0], dtype=xp.float32)

    def _init_particles(self) -> None:
        xp = cp if CUDA_AVAILABLE else np
        n = self.count
        u = xp.random.random(n).astype(xp.float32)
        v = xp.random.random(n).astype(xp.float32)
        w = xp.random.random(n).astype(xp.float32)
        theta = xp.float32(2.0 * np.pi) * u
        phi = xp.arccos(xp.float32(2.0) * v - xp.float32(1.0))
        r = xp.cbrt(w) * xp.float32(self.world_radius * 0.72)
        x = r * xp.sin(phi) * xp.cos(theta)
        y = r * xp.cos(phi) + xp.float32(50.0)
        z = r * xp.sin(phi) * xp.sin(theta)
        self.pos = xp.stack([x, y, z], axis=1).astype(xp.float32)
        self.vel = xp.random.normal(0.0, 12.0, size=(n, 3)).astype(xp.float32)

    def configure(self, payload: dict) -> None:
        if "width" in payload and "height" in payload:
            self.width = float(max(200, int(payload["width"])))
            self.height = float(max(200, int(payload["height"])))
        if "force" in payload:
            self.mouse_strength = float(max(500, min(8000, float(payload["force"]) / 28.0)))
        if "stream_points" in payload:
            self.max_stream_points = int(max(3_000, min(50_000, int(payload["stream_points"]))))
        if "target_fps" in payload:
            self.target_fps = int(max(12, min(40, int(payload["target_fps"]))))
        if "paused" in payload:
            self.paused = bool(payload["paused"])
        if payload.get("reset"):
            self._init_particles()

    def update_mouse(self, payload: dict) -> None:
        self.mouse_x = float(payload.get("x", self.mouse_x))
        self.mouse_y = float(payload.get("y", self.mouse_y))
        self.mouse_mode = int(payload.get("mode", self.mouse_mode))

    def step(self, dt: float) -> None:
        if self.paused:
            return
        xp = cp if CUDA_AVAILABLE else np
        self.time_acc += dt
        if self.mouse_mode != 0:
            nx = (self.mouse_x / max(1.0, self.width) - 0.5) * 2.0
            ny = -(self.mouse_y / max(1.0, self.height) - 0.5) * 2.0
            target = xp.array([nx * self.world_radius * 0.85, ny * self.world_radius * 0.7, 0.0], dtype=xp.float32)
            diff = target - self.pos
            dist2 = xp.sum(diff * diff, axis=1) + xp.float32(40.0)
            inv = xp.reciprocal(xp.sqrt(dist2))
            direction = diff * inv[:, None]
            strength = xp.float32(self.mouse_strength * self.mouse_mode) / dist2
            self.vel += direction * strength[:, None] * xp.float32(dt * 2600.0)

        tx = xp.sin(self.pos[:, 1] * xp.float32(0.016) + xp.float32(self.time_acc * 1.6))
        tz = xp.cos(self.pos[:, 0] * xp.float32(0.014) + xp.float32(self.time_acc * 1.35))
        self.vel[:, 0] += tx * xp.float32(26.0 * dt)
        self.vel[:, 2] += tz * xp.float32(24.0 * dt)
        self.vel[:, 1] += xp.float32(self.gravity * dt)
        self.vel *= xp.float32(self.damping)

        speed = xp.sqrt(xp.sum(self.vel * self.vel, axis=1)) + xp.float32(1e-6)
        over = speed > xp.float32(self.max_speed)
        if xp.any(over):
            scale = xp.where(over, xp.float32(self.max_speed) / speed, xp.float32(1.0))
            self.vel *= scale[:, None]

        self.pos += self.vel * xp.float32(dt)
        dist = xp.sqrt(xp.sum(self.pos * self.pos, axis=1)) + xp.float32(1e-6)
        outside = dist > xp.float32(self.world_radius)
        if xp.any(outside):
            normal = self.pos / dist[:, None]
            self.pos[outside] = normal[outside] * xp.float32(self.world_radius)
            vdot = xp.sum(self.vel * normal, axis=1)
            self.vel[outside] = (self.vel - xp.float32(1.75) * vdot[:, None] * normal)[outside]

        for i in range(int(self.obstacle_centers.shape[0])):
            center = self.obstacle_centers[i]
            radius = self.obstacle_radii[i]
            diff = self.pos - center
            d = xp.sqrt(xp.sum(diff * diff, axis=1)) + xp.float32(1e-6)
            inside = d < radius
            if not xp.any(inside):
                continue
            n = diff / d[:, None]
            self.pos[inside] = center + n[inside] * radius
            vdot = xp.sum(self.vel * n, axis=1)
            reflected = self.vel - xp.float32(1.0 + self.obstacle_bounce) * vdot[:, None] * n
            self.vel[inside] = reflected[inside]

    def _project(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yaw = self.time_acc * 0.35
        pitch = 0.26 + 0.06 * np.sin(self.time_acc * 0.6)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cpit, spit = np.cos(pitch), np.sin(pitch)
        x = arr[:, 0] * cy - arr[:, 2] * sy
        z = arr[:, 0] * sy + arr[:, 2] * cy
        y = arr[:, 1] * cpit - z * spit
        z = arr[:, 1] * spit + z * cpit
        cam_z = -760.0
        focal = 560.0
        zc = np.maximum(z - cam_z, 12.0)
        sx = x * (focal / zc) + self.width * 0.5
        sy2 = -y * (focal / zc) + self.height * 0.5
        return np.stack([sx, sy2], axis=1), zc

    def payload_binary(self) -> bytes:
        if self.count <= self.max_stream_points:
            sampled = self.pos
        else:
            stride = max(1, self.count // self.max_stream_points)
            sampled = self.pos[::stride]
        arr = cp.asnumpy(sampled).astype(np.float32, copy=False) if CUDA_AVAILABLE else sampled.astype(np.float32, copy=False)
        proj, _ = self._project(np.ascontiguousarray(arr))
        x = np.clip(proj[:, 0], 0.0, max(1.0, self.width - 1.0))
        y = np.clip(proj[:, 1], 0.0, max(1.0, self.height - 1.0))
        out_count = proj.shape[0]
        sx = np.float32(65535.0 / max(1.0, self.width - 1.0))
        sy = np.float32(65535.0 / max(1.0, self.height - 1.0))
        packed = np.empty((out_count, 2), dtype=np.uint16)
        packed[:, 0] = (x * sx).astype(np.uint16)
        packed[:, 1] = (y * sy).astype(np.uint16)
        header_bytes = (
            np.array([out_count], dtype=np.uint32).tobytes()
            + np.array([int(self.width), int(self.height)], dtype=np.uint16).tobytes()
        )
        return header_bytes + packed.tobytes()

    def _obstacle_payload(self) -> list[dict]:
        centers = cp.asnumpy(self.obstacle_centers) if CUDA_AVAILABLE else self.obstacle_centers
        radii = cp.asnumpy(self.obstacle_radii) if CUDA_AVAILABLE else self.obstacle_radii
        proj, zc = self._project(np.asarray(centers, dtype=np.float32))
        focal = 560.0
        out = []
        for i in range(len(radii)):
            r2d = max(8.0, min(120.0, float(focal * (float(radii[i]) / float(zc[i])))))
            out.append({"x": float(proj[i, 0]), "y": float(proj[i, 1]), "r": r2d})
        return out

    def stats(self) -> dict:
        if CUDA_AVAILABLE:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mb = round((total_mem - free_mem) / (1024 * 1024), 1)
            total_mb = round(total_mem / (1024 * 1024), 1)
            backend = "cuda-3d"
        else:
            used_mb = 0.0
            total_mb = 0.0
            backend = "cpu-3d"
        return {
            "count": self.count,
            "stream_points": int(min(self.count, self.max_stream_points)),
            "target_fps": self.target_fps,
            "force": self.mouse_strength * 28.0,
            "paused": self.paused,
            "gpu_mem_used_mb": used_mb,
            "gpu_mem_total_mb": total_mb,
            "backend": backend,
            "obstacles": self._obstacle_payload(),
            "obstacles_screen_space": True,
        }


async def _run_sim(websocket: WebSocket, sim) -> None:
    await websocket.accept()
    stop_event = asyncio.Event()

    async def receiver() -> None:
        try:
            while True:
                msg = await websocket.receive_text()
                payload = json.loads(msg)
                typ = payload.get("type")
                if typ == "input":
                    sim.update_mouse(payload)
                elif typ == "config":
                    sim.configure(payload)
        except (WebSocketDisconnect, RuntimeError):
            stop_event.set()
        except Exception:
            stop_event.set()

    recv_task = asyncio.create_task(receiver())
    await websocket.send_text(json.dumps({"type": "ready", "stats": sim.stats()}))
    last_stats = time.perf_counter()
    last_tick = time.perf_counter()

    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            dt = min(now - last_tick, 1.0 / 30.0)
            last_tick = now
            sim.step(dt)
            await websocket.send_bytes(sim.payload_binary())
            if now - last_stats > 1.0:
                await websocket.send_text(json.dumps({"type": "stats", "stats": sim.stats()}))
                last_stats = now
            await asyncio.sleep(1 / sim.target_fps)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop_event.set()
        recv_task.cancel()


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket) -> None:
    sim = GPUSimulator2D(width=1280, height=720, count=60_000 if CUDA_AVAILABLE else 18_000)
    await _run_sim(websocket, sim)


@app.websocket("/ws/sim3d")
async def ws_sim3d(websocket: WebSocket) -> None:
    sim = GPUSimulator3D(width=1280, height=720, count=80_000 if CUDA_AVAILABLE else 16_000)
    await _run_sim(websocket, sim)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
