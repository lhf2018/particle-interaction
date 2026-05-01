import argparse
import math
import subprocess
from dataclasses import dataclass

import cupy as cp
import numpy as np
import pygame

try:
    import pynvml
except Exception:
    pynvml = None


@dataclass
class GPUStats:
    name: str = "Unknown"
    util_gpu: int = -1
    util_mem: int = -1
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    temperature: int = -1
    power_w: float = -1.0


class GPUStatsMonitor:
    def __init__(self) -> None:
        self.available = False
        self.handle = None
        self.name = "Unavailable"
        self.smi_available = False
        if pynvml is None:
            self._probe_smi()
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self.handle).decode(errors="ignore")
            self.available = True
        except Exception:
            self.available = False
        self._probe_smi()

    def _probe_smi(self) -> None:
        try:
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True, timeout=1.5)
            self.smi_available = True
            if self.name == "Unavailable":
                line = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=1.5,
                ).strip()
                if line:
                    self.name = line
        except Exception:
            self.smi_available = False

    def read(self) -> GPUStats:
        stats = GPUStats(name=self.name)
        if self.available and self.handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                stats.util_gpu = int(util.gpu)
                stats.util_mem = int(util.memory)
                stats.mem_used_mb = mem.used / (1024 * 1024)
                stats.mem_total_mb = mem.total / (1024 * 1024)
                stats.temperature = int(temp)
                stats.power_w = float(power)
                return stats
            except Exception:
                pass

        # Fallback when NVML Python bindings fail but nvidia-smi exists.
        if self.smi_available:
            try:
                line = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,name",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=1.5,
                ).strip()
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    stats.util_gpu = int(float(parts[0]))
                    stats.util_mem = int(float(parts[1]))
                    stats.mem_used_mb = float(parts[2])
                    stats.mem_total_mb = float(parts[3])
                    stats.temperature = int(float(parts[4]))
                    if parts[5].lower() not in ("n/a", "[n/a]"):
                        stats.power_w = float(parts[5])
                    if parts[6]:
                        stats.name = parts[6]
            except Exception:
                pass
        return stats


class ParticleSystem3D:
    def __init__(self, count: int, width: int, height: int) -> None:
        self.count = count
        self.width = width
        self.height = height
        self.center = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
        self.gravity = cp.array([0.0, -120.0, 0.0], dtype=cp.float32)
        self.gravity_enabled = True
        self.damping = 0.993
        self.bounce = 0.72
        self.max_speed = 650.0
        self.world_radius = 430.0
        self.mouse_strength = 2900.0
        self.mouse_radius = 180.0
        self.mouse_linear_gain = 1300.0
        self.noise_strength = 35.0
        self.floor_band = 85.0
        self.floor_push = 520.0
        self.t = 0.0
        self.shock_pos = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
        self.shock_time = 999.0
        self.shock_mode = 0
        # Internal obstacles removed: keep clean spherical volume only.
        self.obstacle_bounce = 0.82
        self.obstacle_centers = cp.empty((0, 3), dtype=cp.float32)
        self.obstacle_radii = cp.empty((0,), dtype=cp.float32)
        self.reset(self.count)

    def _set_obstacles(self, centers: list[list[float]], radii: list[float]) -> None:
        self.obstacle_centers = cp.array(centers, dtype=cp.float32)
        self.obstacle_radii = cp.array(radii, dtype=cp.float32)

    def _build_triad(self) -> tuple[list[list[float]], list[float]]:
        centers = [[-105.0, -35.0, 35.0], [120.0, -25.0, -55.0], [0.0, 75.0, 100.0]]
        radii = [58.0, 52.0, 46.0]
        return centers, radii

    def _build_funnel(self) -> tuple[list[list[float]], list[float]]:
        centers: list[list[float]] = []
        radii: list[float] = []
        levels = 8
        for i in range(levels):
            t = i / (levels - 1)
            y = 145.0 - 34.0 * i
            z = -40.0 + 18.0 * i
            x_offset = self.funnel_gap * (1.0 - t) * 0.5 + 22.0
            centers.append([-x_offset, y, z])
            centers.append([x_offset, y, z])
            radii.extend([22.0, 22.0])
        # Bottom pegs to break symmetry and create richer splashes.
        centers.extend([[-32.0, -120.0, 18.0], [26.0, -136.0, -22.0], [2.0, -96.0, 44.0]])
        radii.extend([28.0, 24.0, 20.0])
        return centers, radii

    def _build_ring(self) -> tuple[list[list[float]], list[float]]:
        centers: list[list[float]] = []
        radii: list[float] = []
        n = 14
        ring_r = 145.0
        for i in range(n):
            a = 2.0 * math.pi * i / n
            centers.append([math.cos(a) * ring_r, -25.0 + 22.0 * math.sin(a * 2.0), math.sin(a) * ring_r])
            radii.append(22.0)
        centers.append([0.0, 78.0, 0.0])
        radii.append(34.0)
        return centers, radii

    def _build_galaxy(self) -> tuple[list[list[float]], list[float]]:
        centers = [
            [0.0, 0.0, 0.0],
            [85.0, 48.0, -52.0],
            [-98.0, -34.0, 42.0],
            [126.0, -86.0, 24.0],
            [-122.0, 92.0, -30.0],
            [0.0, 132.0, 88.0],
            [0.0, -132.0, -82.0],
        ]
        radii = [44.0, 28.0, 30.0, 24.0, 26.0, 22.0, 22.0]
        return centers, radii

    def _rebuild_obstacles(self) -> None:
        self.obstacle_centers = cp.empty((0, 3), dtype=cp.float32)
        self.obstacle_radii = cp.empty((0,), dtype=cp.float32)

    def cycle_style(self, step: int = 1) -> None:
        return

    def set_style(self, idx: int) -> None:
        return

    def adjust_funnel_gap(self, scale: float) -> None:
        return

    @staticmethod
    def recommend_count(target_count: int) -> int:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        estimated = target_count * (3 * 4 + 3 * 4 + 4 + 4)
        budget = int(free_mem * 0.50)
        if estimated <= budget:
            return target_count
        safe = max(30_000, int(budget / 24))
        return min(target_count, safe)

    def reset(self, count: int | None = None) -> None:
        if count is not None:
            self.count = count
        n = self.count
        u = cp.random.random(n, dtype=cp.float32)
        v = cp.random.random(n, dtype=cp.float32)
        w = cp.random.random(n, dtype=cp.float32)
        theta = cp.float32(2.0 * math.pi) * u
        phi = cp.arccos(cp.float32(2.0) * v - cp.float32(1.0))
        r = cp.cbrt(w) * cp.float32(self.world_radius * 0.55)
        sx = r * cp.sin(phi) * cp.cos(theta)
        sy = r * cp.cos(phi) + cp.float32(80.0)
        sz = r * cp.sin(phi) * cp.sin(theta)
        self.pos = cp.stack([sx, sy, sz], axis=1).astype(cp.float32)
        self.vel = cp.random.normal(0.0, 9.0, (n, 3)).astype(cp.float32)

    def _apply_mouse_force(self, dt: float, mouse_pos: tuple[int, int], mode: int) -> None:
        if mode == 0:
            return
        mx, my = mouse_pos
        nx = (mx / max(1.0, self.width) - 0.5) * 2.0
        ny = -(my / max(1.0, self.height) - 0.5) * 2.0
        target = cp.array(
            [nx * self.world_radius * 0.75, ny * self.world_radius * 0.6, 0.0], dtype=cp.float32
        )
        diff = target - self.pos
        dist2 = cp.sum(diff * diff, axis=1) + cp.float32(30.0)
        dist = cp.sqrt(dist2)
        inv_dist = cp.reciprocal(cp.sqrt(dist2))
        direction = diff * inv_dist[:, None]
        mag = cp.float32(self.mouse_strength * mode) / dist2
        self.vel += direction * mag[:, None] * cp.float32(dt * 1200.0)
        near = dist < cp.float32(self.mouse_radius)
        near_gain = cp.where(
            near,
            cp.float32(1.0) - dist / cp.float32(self.mouse_radius),
            cp.float32(0.0),
        )
        self.vel += direction * near_gain[:, None] * cp.float32(self.mouse_linear_gain * mode * dt)

        # Swirl around Z axis to make interaction feel less rigid.
        tangent = cp.stack([-direction[:, 1], direction[:, 0], cp.zeros(self.count, dtype=cp.float32)], axis=1)
        swirl = cp.float32(self.mouse_strength * 0.12 * mode) / (dist2 + cp.float32(120.0))
        self.vel += tangent * swirl[:, None] * cp.float32(dt * 900.0)

    def trigger_shock(self, mouse_pos: tuple[int, int], mode: int) -> None:
        mx, my = mouse_pos
        nx = (mx / max(1.0, self.width) - 0.5) * 2.0
        ny = -(my / max(1.0, self.height) - 0.5) * 2.0
        self.shock_pos = cp.array(
            [nx * self.world_radius * 0.75, ny * self.world_radius * 0.6, 0.0], dtype=cp.float32
        )
        self.shock_time = 0.0
        self.shock_mode = mode

    def _apply_shock(self, dt: float) -> None:
        if self.shock_time > 0.45 or self.shock_mode == 0:
            return
        self.shock_time += dt
        age = cp.float32(self.shock_time)
        diff = self.pos - self.shock_pos
        dist = cp.sqrt(cp.sum(diff * diff, axis=1)) + cp.float32(1e-5)
        wave_front = cp.float32(35.0 + 420.0 * self.shock_time)
        band = cp.abs(dist - wave_front)
        active = band < cp.float32(70.0)
        normal = diff / dist[:, None]
        impulse = cp.where(active, cp.float32(1.0) - band / cp.float32(70.0), cp.float32(0.0))
        strength = cp.float32(780.0 * (1.0 - self.shock_time / 0.45) * self.shock_mode)
        self.vel += normal * impulse[:, None] * strength * cp.float32(dt)

    def _apply_noise(self, dt: float) -> None:
        nx = cp.sin(self.pos[:, 1] * cp.float32(0.017) + cp.float32(self.t * 1.31))
        ny = cp.cos(self.pos[:, 2] * cp.float32(0.015) + cp.float32(self.t * 1.17))
        nz = cp.sin(self.pos[:, 0] * cp.float32(0.013) + cp.float32(self.t * 1.47))
        self.vel[:, 0] += nx * cp.float32(self.noise_strength * dt)
        self.vel[:, 1] += ny * cp.float32(self.noise_strength * 0.8 * dt)
        self.vel[:, 2] += nz * cp.float32(self.noise_strength * 0.7 * dt)

    def _apply_floor_convection(self, dt: float) -> None:
        floor_y = -self.world_radius + self.floor_band
        depth = floor_y - self.pos[:, 1]
        in_band = depth > 0.0
        push = cp.where(in_band, depth / cp.float32(self.floor_band), cp.float32(0.0))
        self.vel[:, 1] += push * cp.float32(self.floor_push * dt)
        swirl = cp.sin(self.pos[:, 0] * cp.float32(0.015) + cp.float32(self.t * 2.1))
        swirl2 = cp.cos(self.pos[:, 2] * cp.float32(0.014) + cp.float32(self.t * 1.8))
        self.vel[:, 0] += cp.where(in_band, swirl, cp.float32(0.0)) * cp.float32(70.0 * dt)
        self.vel[:, 2] += cp.where(in_band, swirl2, cp.float32(0.0)) * cp.float32(70.0 * dt)

    def _apply_container(self) -> None:
        diff = self.pos - self.center
        dist = cp.sqrt(cp.sum(diff * diff, axis=1)) + cp.float32(1e-6)
        outside = dist > cp.float32(self.world_radius)
        normal = diff / dist[:, None]
        self.pos[outside] = self.center + normal[outside] * cp.float32(self.world_radius)
        vdotn = cp.sum(self.vel * normal, axis=1)
        reflected = self.vel - (cp.float32(1.0) + cp.float32(self.bounce)) * vdotn[:, None] * normal
        self.vel[outside] = reflected[outside]

    def _apply_obstacle_collisions(self) -> None:
        return

    def step(self, dt: float, mouse_pos: tuple[int, int], mouse_mode: int, paused: bool) -> None:
        if paused:
            return
        self.t += dt
        self._apply_mouse_force(dt, mouse_pos, mouse_mode)
        self._apply_shock(dt)
        self._apply_noise(dt)
        self._apply_floor_convection(dt)
        if self.gravity_enabled:
            self.vel += self.gravity * cp.float32(dt)

        speed = cp.sqrt(cp.sum(self.vel * self.vel, axis=1)) + cp.float32(1e-6)
        over = speed > cp.float32(self.max_speed)
        if cp.any(over):
            scale = cp.where(over, cp.float32(self.max_speed) / speed, cp.float32(1.0))
            self.vel *= scale[:, None]

        self.vel *= cp.float32(self.damping)
        self.pos += self.vel * cp.float32(dt)
        self._apply_obstacle_collisions()
        self._apply_container()


class Renderer3D:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.focal = 520.0
        self.camera_z = -520.0
        self.show_hud = True
        self.color_mode = 1
        self.font = pygame.font.SysFont("Consolas", 18)
        self.trail = pygame.Surface((width, height), pygame.SRCALPHA)
        self.use_trail = False
        self.phase = 0.0
        self.orbit = 0.0
        self.stars = self._build_stars(280)
        self.max_render_particles = 140_000
        self.gpu_raster = True
        self._gpu_canvas: cp.ndarray | None = None

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.trail = pygame.Surface((width, height), pygame.SRCALPHA)
        self.stars = self._build_stars(280)

    def _build_stars(self, n: int) -> np.ndarray:
        x = np.random.randint(0, max(1, self.width), size=n)
        y = np.random.randint(0, max(1, self.height), size=n)
        b = np.random.randint(70, 180, size=n)
        return np.stack([x, y, b], axis=1)

    def _draw_reference_grid(self, screen: pygame.Surface, yaw: float, pitch: float) -> None:
        # Perspective-ish floor grid as a stable reference object for camera motion.
        horizon = int(self.height * 0.48 + math.sin(pitch) * 28.0)
        pygame.draw.line(screen, (120, 140, 165), (0, horizon), (self.width, horizon), 1)
        center_x = self.width * 0.5
        spread = 0.85 + abs(math.sin(yaw)) * 0.5
        for i in range(-12, 13):
            x0 = center_x + i * 60 * spread
            x1 = center_x + i * 12 * (0.6 + abs(math.cos(yaw)))
            pygame.draw.line(
                screen,
                (58, 76, 98),
                (int(x0), self.height),
                (int(x1), horizon),
                1,
            )
        for j in range(1, 10):
            y = int(horizon + (self.height - horizon) * (j / 10.0) ** 1.7)
            pygame.draw.line(screen, (50, 68, 92), (0, y), (self.width, y), 1)

        # Light tech stripes for stronger visual guidance.
        for k in range(0, self.width, 80):
            alpha_col = (65, 105, 145)
            pygame.draw.line(screen, alpha_col, (k, horizon), (k // 3, self.height), 1)

    def _draw_axes(self, screen: pygame.Surface, yaw: float, pitch: float) -> None:
        # Screen-space axis gizmo, like DCC tools.
        ox, oy = 90, self.height - 90
        l = 46
        x_end = (int(ox + l * math.cos(yaw)), int(oy - l * math.sin(yaw)))
        y_end = (int(ox), int(oy - l))
        z_end = (int(ox - l * math.sin(yaw)), int(oy - l * math.cos(yaw) * 0.8))
        pygame.draw.line(screen, (255, 95, 95), (ox, oy), x_end, 3)
        pygame.draw.line(screen, (95, 255, 120), (ox, oy), y_end, 3)
        pygame.draw.line(screen, (110, 165, 255), (ox, oy), z_end, 3)
        pygame.draw.circle(screen, (220, 230, 245), (ox, oy), 5, 1)

    def _draw_wire_sphere_hint(self, screen: pygame.Surface, phase: float, yaw: float) -> None:
        # Subtle container cue helps depth perception.
        cx = int(self.width * 0.83)
        cy = int(self.height * 0.24)
        r = 64
        pygame.draw.circle(screen, (120, 150, 185), (cx, cy), r, 1)
        ry = int(r * (0.28 + 0.18 * (0.5 + 0.5 * math.sin(phase * 1.2 + yaw))))
        pygame.draw.ellipse(screen, (95, 130, 170), (cx - r, cy - ry, 2 * r, 2 * ry), 1)
        rx = int(r * (0.35 + 0.2 * (0.5 + 0.5 * math.cos(phase * 1.1 + yaw))))
        pygame.draw.ellipse(screen, (95, 130, 170), (cx - rx, cy - r, 2 * rx, 2 * r), 1)

    def _project_points_with_camera(
        self, points_3d: np.ndarray, yaw: float, pitch: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        x = points_3d[:, 0] * cos_y - points_3d[:, 2] * sin_y
        z = points_3d[:, 0] * sin_y + points_3d[:, 2] * cos_y
        y = points_3d[:, 1] * cos_p - z * sin_p
        z = points_3d[:, 1] * sin_p + z * cos_p
        transformed = np.stack([x, y, z], axis=1)
        points, radii = self._project(transformed)
        return points, radii

    def _draw_obstacles(self, screen: pygame.Surface, ps: ParticleSystem3D, yaw: float, pitch: float) -> None:
        return

    def _project(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = pos[:, 2] - self.camera_z
        z = np.maximum(z, 10.0)
        scale = self.focal / z
        x = (pos[:, 0] * scale + self.width * 0.5).astype(np.int32)
        y = (-pos[:, 1] * scale + self.height * 0.5).astype(np.int32)
        radius = np.clip((scale * 2.4).astype(np.int32), 1, 4)
        return np.stack([x, y], axis=1), radius

    def _render_particles_gpu(self, ps: ParticleSystem3D, yaw: float, pitch: float, render_stride: int) -> np.ndarray:
        pos = ps.pos[::render_stride]
        vel = ps.vel[::render_stride]
        cos_y = cp.float32(math.cos(yaw))
        sin_y = cp.float32(math.sin(yaw))
        cos_p = cp.float32(math.cos(pitch))
        sin_p = cp.float32(math.sin(pitch))

        x = pos[:, 0] * cos_y - pos[:, 2] * sin_y
        z = pos[:, 0] * sin_y + pos[:, 2] * cos_y
        y = pos[:, 1] * cos_p - z * sin_p
        z = pos[:, 1] * sin_p + z * cos_p

        z_cam = cp.maximum(z - cp.float32(self.camera_z), cp.float32(10.0))
        scale = cp.float32(self.focal) / z_cam
        sx = cp.rint(x * scale + cp.float32(self.width * 0.5)).astype(cp.int32)
        sy = cp.rint(-y * scale + cp.float32(self.height * 0.5)).astype(cp.int32)
        inside = (sx >= 0) & (sx < self.width) & (sy >= 0) & (sy < self.height)

        if not bool(cp.any(inside)):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        sx = sx[inside]
        sy = sy[inside]
        speed = cp.sqrt(cp.sum(vel * vel, axis=1) + cp.float32(1e-6))[inside]
        z_in = z[inside]
        z_min = cp.min(z_in)
        z_ptp = cp.maximum(cp.ptp(z_in), cp.float32(1e-6))
        depth_norm = (z_in - z_min) / z_ptp
        s = cp.clip(speed / cp.float32(380.0), cp.float32(0.0), cp.float32(1.0))
        hue_wave = cp.float32(0.5) + cp.float32(0.5) * cp.sin(cp.float32(self.phase * 2.4) + sx.astype(cp.float32) * cp.float32(0.012))
        fog = cp.float32(0.45) + cp.float32(0.55) * (cp.float32(1.0) - depth_norm)

        colors = cp.empty((sx.shape[0], 3), dtype=cp.uint8)
        colors[:, 0] = cp.clip((cp.float32(120.0) + s * cp.float32(120.0) + hue_wave * cp.float32(90.0)) * fog, 0, 255).astype(cp.uint8)
        colors[:, 1] = cp.clip((cp.float32(70.0) + (cp.float32(1.0) - s) * cp.float32(110.0) + hue_wave * cp.float32(35.0)) * fog, 0, 255).astype(cp.uint8)
        colors[:, 2] = cp.clip((cp.float32(255.0) - s * cp.float32(170.0) + (cp.float32(1.0) - hue_wave) * cp.float32(80.0)) * fog, 0, 255).astype(cp.uint8)

        if self._gpu_canvas is None or self._gpu_canvas.shape[:2] != (self.height, self.width):
            self._gpu_canvas = cp.zeros((self.height, self.width, 3), dtype=cp.uint8)
        else:
            self._gpu_canvas.fill(0)

        # Core point (keep particles visually small/crisp).
        self._gpu_canvas[sy, sx] = colors
        # Very light halo only, so particle points stay small.
        c_soft = (colors.astype(cp.float32) * cp.float32(0.45)).astype(cp.uint8)
        sx_p = cp.clip(sx + 1, 0, self.width - 1)
        sx_m = cp.clip(sx - 1, 0, self.width - 1)
        sy_p = cp.clip(sy + 1, 0, self.height - 1)
        sy_m = cp.clip(sy - 1, 0, self.height - 1)
        self._gpu_canvas[sy, sx_p] = cp.maximum(self._gpu_canvas[sy, sx_p], c_soft)
        self._gpu_canvas[sy, sx_m] = cp.maximum(self._gpu_canvas[sy, sx_m], c_soft)
        self._gpu_canvas[sy_p, sx] = cp.maximum(self._gpu_canvas[sy_p, sx], c_soft)
        self._gpu_canvas[sy_m, sx] = cp.maximum(self._gpu_canvas[sy_m, sx], c_soft)
        return cp.asnumpy(self._gpu_canvas)

    def draw(
        self,
        screen: pygame.Surface,
        ps: ParticleSystem3D,
        fps: float,
        paused: bool,
        gpu: GPUStats,
        mouse_pos: tuple[int, int],
        dt: float,
    ) -> None:
        self.phase += dt
        self.orbit += dt * 0.42

        # Auto orbit + mouse-driven camera tilt for stronger 3D feeling.
        mx, my = mouse_pos
        yaw = self.orbit + (mx / max(1, self.width) - 0.5) * 0.9
        pitch = (my / max(1, self.height) - 0.5) * 0.55

        render_stride = max(1, int(ps.count / max(1, self.max_render_particles)))
        if self.gpu_raster:
            canvas = self._render_particles_gpu(ps, yaw, pitch, render_stride)
        else:
            # Legacy CPU path can be restored later if needed.
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        frame = pygame.surfarray.make_surface(np.swapaxes(canvas, 0, 1))
        if self.use_trail:
            self.trail.fill((180, 205, 235, 8))
            self.trail.blit(frame, (0, 0))

        bg_top = (
            int(138 + 34 * (0.5 + 0.5 * math.sin(self.phase * 0.7))),
            int(176 + 34 * (0.5 + 0.5 * math.sin(self.phase * 0.9 + 1.1))),
            int(230 + 20 * (0.5 + 0.5 * math.sin(self.phase * 0.8 + 2.1))),
        )
        bg_bottom = (70, 106, 156)
        screen.fill(bg_bottom)
        for yi in range(0, self.height, 6):
            t = yi / max(1, self.height - 1)
            col = (
                int(bg_top[0] * (1 - t) + bg_bottom[0] * t),
                int(bg_top[1] * (1 - t) + bg_bottom[1] * t),
                int(bg_top[2] * (1 - t) + bg_bottom[2] * t),
            )
            pygame.draw.line(screen, col, (0, yi), (self.width, yi))

        self._draw_reference_grid(screen, yaw, pitch)

        # Twinkling stars.
        twinkle = 0.6 + 0.4 * math.sin(self.phase * 3.0)
        for sx, sy, b in self.stars:
            # Parallax star drift with camera yaw.
            drift_x = int((yaw * 26.0) % max(1, self.width))
            x = (int(sx) + drift_x) % max(1, self.width)
            c = int(min(255, b * twinkle))
            screen.set_at((x, int(sy)), (c, c, min(255, c + 25)))

        if self.use_trail:
            screen.blit(self.trail, (0, 0))
        else:
            screen.blit(frame, (0, 0))

        # Additive glow pass for cinematic punch.
        glow = pygame.transform.smoothscale(frame, (max(1, self.width // 2), max(1, self.height // 2)))
        glow = pygame.transform.smoothscale(glow, (self.width, self.height))
        glow.set_alpha(34)
        screen.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        # Draw obstacles on top so they stay visible as scene references.
        self._draw_obstacles(screen, ps, yaw, pitch)
        self._draw_wire_sphere_hint(screen, self.phase, yaw)
        self._draw_axes(screen, yaw, pitch)

        if self.show_hud:
            pool_used_mb = cp.get_default_memory_pool().used_bytes() / (1024 * 1024)
            hud_lines = [
                f"FPS: {fps:5.1f} | N: {ps.count} | {'PAUSED' if paused else 'RUN'}",
                f"Mouse L=Attract R=Repel M=Pause/Reset | Wheel=Force({ps.mouse_strength:.0f})",
                f"Render stride: {render_stride} | GPU raster: {'ON' if self.gpu_raster else 'OFF'}",
                f"GPU: {gpu.name}",
                (
                    f"Util GPU: {gpu.util_gpu if gpu.util_gpu >= 0 else 'N/A'}% | "
                    f"MemCtl: {gpu.util_mem if gpu.util_mem >= 0 else 'N/A'}% | "
                    f"VRAM: {gpu.mem_used_mb:.0f}/{gpu.mem_total_mb:.0f} MB"
                ),
                f"Temp: {gpu.temperature if gpu.temperature >= 0 else 'N/A'} C | Power: {gpu.power_w:.1f} W"
                if gpu.power_w >= 0
                else f"Temp: {gpu.temperature if gpu.temperature >= 0 else 'N/A'} C | Power: N/A",
                f"CuPy Pool: {pool_used_mb:.0f} MB",
            ]
            y = 8
            for line in hud_lines:
                text = self.font.render(line, True, (190, 255, 210))
                screen.blit(text, (8, y))
                y += 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FluidVibe 3D - GPU realtime particle fluid")
    parser.add_argument("--particles", type=int, default=120_000, help="Target particle count, default=120000")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("FluidVibe 3D (GPU) v2.4 Bright")
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    gpu_monitor = GPUStatsMonitor()

    count = ParticleSystem3D.recommend_count(args.particles)
    ps = ParticleSystem3D(count, args.width, args.height)
    renderer = Renderer3D(args.width, args.height)

    running = True
    paused = False
    frame_idx = 0
    cached_stats = gpu_monitor.read()
    last_middle_click_ms = 0
    slider_dragging = False
    ui_hover = False
    slider_min = 20_000
    slider_max = 320_000
    pending_particles = ps.count

    def slider_rects() -> tuple[pygame.Rect, pygame.Rect]:
        panel_w = 360
        panel_h = 56
        panel_x = max(10, renderer.width - panel_w - 10)
        panel_y = 10
        bar = pygame.Rect(panel_x + 14, panel_y + 30, panel_w - 28, 10)
        panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        return panel, bar

    def slider_value_from_mouse(mx: int, bar: pygame.Rect) -> int:
        t = (mx - bar.left) / max(1, bar.width)
        t = max(0.0, min(1.0, t))
        return int(slider_min + t * (slider_max - slider_min))

    def apply_particle_target(target: int) -> None:
        nonlocal pending_particles
        clamped = max(slider_min, min(slider_max, int(target)))
        recommended = ParticleSystem3D.recommend_count(clamped)
        if recommended != ps.count:
            ps.reset(recommended)
        pending_particles = clamped

    while running:
        dt = min(clock.tick(120) / 1000.0, 1.0 / 30.0)
        frame_idx += 1
        panel_rect, bar_rect = slider_rects()
        mx, my = pygame.mouse.get_pos()
        ui_hover = panel_rect.collidepoint(mx, my)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                ps.width, ps.height = e.w, e.h
                renderer.resize(e.w, e.h)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1 and panel_rect.collidepoint(e.pos):
                    slider_dragging = True
                    pending_particles = slider_value_from_mouse(e.pos[0], bar_rect)
                elif e.button == 4:
                    ps.mouse_strength *= 1.12
                elif e.button == 5:
                    ps.mouse_strength /= 1.12
                elif e.button == 2:
                    now = pygame.time.get_ticks()
                    if now - last_middle_click_ms < 280:
                        ps.reset()
                    else:
                        paused = not paused
                    last_middle_click_ms = now
                elif e.button == 1 and not panel_rect.collidepoint(e.pos):
                    # Left click is attract mode; inward shock feels more "follow mouse".
                    ps.trigger_shock(pygame.mouse.get_pos(), mode=-1)
                elif e.button == 3:
                    # Right click is repel mode.
                    ps.trigger_shock(pygame.mouse.get_pos(), mode=1)
            elif e.type == pygame.MOUSEMOTION and slider_dragging:
                pending_particles = slider_value_from_mouse(e.pos[0], bar_rect)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and slider_dragging:
                slider_dragging = False
                apply_particle_target(pending_particles)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_5, pygame.K_KP5):
                    renderer.max_render_particles = max(40_000, renderer.max_render_particles - 20_000)
                elif e.key in (pygame.K_6, pygame.K_KP6):
                    renderer.max_render_particles = min(240_000, renderer.max_render_particles + 20_000)

        left, middle, right = pygame.mouse.get_pressed(3)
        if slider_dragging or ui_hover:
            left = False
            right = False
        mouse_mode = 0
        if left:
            mouse_mode = 1
        elif right:
            mouse_mode = -1
        if middle:
            ps.noise_strength = 65.0
        else:
            ps.noise_strength = 35.0

        ps.step(dt, pygame.mouse.get_pos(), mouse_mode, paused)

        if frame_idx % 20 == 0:
            cached_stats = gpu_monitor.read()

        renderer.draw(screen, ps, clock.get_fps(), paused, cached_stats, pygame.mouse.get_pos(), dt)

        # Particle count slider UI (top-right).
        panel_rect, bar_rect = slider_rects()
        pygame.draw.rect(screen, (8, 14, 24, 185), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (95, 130, 170), panel_rect, width=1, border_radius=8)
        pygame.draw.rect(screen, (70, 92, 118), bar_rect, border_radius=5)
        ratio = (pending_particles - slider_min) / max(1, slider_max - slider_min)
        ratio = max(0.0, min(1.0, ratio))
        knob_x = int(bar_rect.left + ratio * bar_rect.width)
        pygame.draw.rect(screen, (120, 190, 255), pygame.Rect(bar_rect.left, bar_rect.top, knob_x - bar_rect.left, bar_rect.height), border_radius=5)
        pygame.draw.circle(screen, (220, 245, 255), (knob_x, bar_rect.centery), 8)
        slider_text = renderer.font.render(
            f"Particles: {ps.count}  ->  Target: {pending_particles}",
            True,
            (210, 240, 255),
        )
        hint_text = renderer.font.render("Drag slider to adjust, release to apply", True, (160, 205, 235))
        screen.blit(slider_text, (panel_rect.x + 12, panel_rect.y + 6))
        screen.blit(hint_text, (panel_rect.x + 12, panel_rect.y + 42))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
