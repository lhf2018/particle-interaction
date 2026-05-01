import argparse
import math

import cupy as cp
import numpy as np
import pygame


class ParticleSystem:
    def __init__(self, count: int, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.count = count
        self.gravity_enabled = True
        self.gravity = 180.0
        self.mouse_strength = 3200.0
        self.damping = 0.995
        self.bounce = 0.75
        self.max_speed = 900.0
        self.floor_band = 90.0
        self.floor_spring = 520.0
        self.floor_drag = 0.965
        self.turbulence = 24.0
        self.sim_time = 0.0
        self.reset(count)

    @staticmethod
    def recommend_count(target_count: int) -> int:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        estimated_bytes = target_count * (2 * 4 + 2 * 4 + 2 * 4 + 4 + 4)
        # Keep a conservative headroom to avoid OOM spikes.
        budget = int(free_mem * 0.55)
        if estimated_bytes <= budget:
            return target_count
        safe_count = max(20_000, int(budget / 32))
        return min(target_count, safe_count)

    def reset(self, count: int | None = None) -> None:
        if count is not None:
            self.count = count
        n = self.count
        cx = self.width * 0.5
        cy = self.height * 0.42
        radius = min(self.width, self.height) * 0.24

        r = cp.sqrt(cp.random.random(n, dtype=cp.float32)) * radius
        theta = cp.random.random(n, dtype=cp.float32) * cp.float32(2.0 * math.pi)
        self.pos = cp.empty((n, 2), dtype=cp.float32)
        self.vel = cp.random.normal(0.0, 8.0, size=(n, 2)).astype(cp.float32)
        self.pos[:, 0] = cp.float32(cx) + r * cp.cos(theta)
        self.pos[:, 1] = cp.float32(cy) + r * cp.sin(theta)

    def resize(self, width: int, height: int) -> None:
        sx = width / max(1, self.width)
        sy = height / max(1, self.height)
        self.pos[:, 0] *= cp.float32(sx)
        self.pos[:, 1] *= cp.float32(sy)
        self.width = width
        self.height = height

    def _apply_mouse_force(self, dt: float, mouse_pos: tuple[int, int], mode: int) -> None:
        # mode: 1 attract (right mouse), -1 repel (left mouse), 0 off
        if mode == 0:
            return
        mx, my = mouse_pos
        mouse = cp.array([mx, my], dtype=cp.float32)
        diff = mouse - self.pos
        dist2 = cp.sum(diff * diff, axis=1) + cp.float32(36.0)
        inv_dist = cp.reciprocal(cp.sqrt(dist2))
        direction = diff * inv_dist[:, None]
        mag = cp.float32(self.mouse_strength * mode) / dist2
        self.vel += direction * mag[:, None] * cp.float32(dt * 1300.0)
        # Add a tangential component so dragging creates whirlpool-like flow.
        tangent = cp.stack((-direction[:, 1], direction[:, 0]), axis=1)
        swirl = cp.float32(self.mouse_strength * 0.15 * mode) / (dist2 + cp.float32(120.0))
        self.vel += tangent * swirl[:, None] * cp.float32(dt * 900.0)

    def _apply_floor_effects(self, dt: float) -> None:
        floor_line = self.height - 1.0
        depth = self.pos[:, 1] - cp.float32(floor_line - self.floor_band)
        in_band = depth > 0.0
        if not cp.any(in_band):
            return

        # Soft spring near floor: particles compress then rebound, like dense liquid.
        push = cp.where(in_band, depth / cp.float32(self.floor_band), cp.float32(0.0))
        self.vel[:, 1] -= push * cp.float32(self.floor_spring * dt)

        # Lateral drift keeps bottom particles slowly moving instead of freezing.
        wave = cp.sin(self.pos[:, 0] * cp.float32(0.011) + cp.float32(self.sim_time * 1.9))
        self.vel[:, 0] += cp.where(in_band, wave, cp.float32(0.0)) * cp.float32(40.0 * dt)
        self.vel[:, 0] = cp.where(in_band, self.vel[:, 0] * cp.float32(self.floor_drag), self.vel[:, 0])

    def _apply_turbulence(self, dt: float) -> None:
        # Low-cost deterministic turbulence to avoid dead-looking motion.
        tx = cp.sin(self.pos[:, 1] * cp.float32(0.018) + cp.float32(self.sim_time * 1.3))
        ty = cp.cos(self.pos[:, 0] * cp.float32(0.016) + cp.float32(self.sim_time * 1.6))
        self.vel[:, 0] += tx * cp.float32(self.turbulence * dt)
        self.vel[:, 1] += ty * cp.float32(self.turbulence * 0.7 * dt)

    def clear_around(self, center: tuple[int, int], radius: float = 50.0) -> None:
        cx, cy = center
        diff = self.pos - cp.array([cx, cy], dtype=cp.float32)
        dist2 = cp.sum(diff * diff, axis=1)
        keep = dist2 > cp.float32(radius * radius)
        kept = int(cp.count_nonzero(keep).item())
        if kept < 20_000:
            return
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.count = kept

    def step(self, dt: float, mouse_pos: tuple[int, int], mouse_mode: int, paused: bool) -> None:
        if paused:
            return
        self.sim_time += dt

        self._apply_mouse_force(dt, mouse_pos, mouse_mode)
        self._apply_turbulence(dt)

        if self.gravity_enabled:
            self.vel[:, 1] += cp.float32(self.gravity * dt)

        self._apply_floor_effects(dt)

        speed = cp.sqrt(cp.sum(self.vel * self.vel, axis=1)) + cp.float32(1e-6)
        too_fast = speed > cp.float32(self.max_speed)
        if cp.any(too_fast):
            scale = cp.where(too_fast, cp.float32(self.max_speed) / speed, cp.float32(1.0))
            self.vel *= scale[:, None]

        self.vel *= cp.float32(self.damping)
        self.pos += self.vel * cp.float32(dt)

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

        vx[left | right] *= cp.float32(-self.bounce)
        vy[top | bottom] *= cp.float32(-self.bounce)


class Renderer:
    def __init__(self, width: int, height: int, show_fps: bool = True) -> None:
        self.width = width
        self.height = height
        self.show_fps = show_fps
        self.mode = 0  # 0=particles, 1=velocity color
        self.font = pygame.font.SysFont("Consolas", 18)
        self.trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def draw(self, screen: pygame.Surface, ps: ParticleSystem, fps: float, paused: bool) -> None:
        pos_cpu = cp.asnumpy(ps.pos)
        vel_cpu = cp.asnumpy(ps.vel)

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        xy = np.rint(pos_cpu).astype(np.int32)
        np.clip(xy[:, 0], 0, self.width - 1, out=xy[:, 0])
        np.clip(xy[:, 1], 0, self.height - 1, out=xy[:, 1])

        if self.mode == 0:
            canvas[xy[:, 1], xy[:, 0]] = (235, 235, 240)
        else:
            speed = np.linalg.norm(vel_cpu, axis=1)
            s = np.clip(speed / 420.0, 0.0, 1.0)
            colors = np.zeros((ps.count, 3), dtype=np.uint8)
            colors[:, 0] = (s * 255).astype(np.uint8)
            colors[:, 1] = (70 + 80 * (1.0 - s)).astype(np.uint8)
            colors[:, 2] = ((1.0 - s) * 255).astype(np.uint8)
            canvas[xy[:, 1], xy[:, 0]] = colors

        surface = pygame.surfarray.make_surface(np.swapaxes(canvas, 0, 1))
        # Slight afterimage makes movement easier to perceive.
        self.trail_surface.fill((0, 0, 0, 36))
        self.trail_surface.blit(surface, (0, 0))
        screen.fill((0, 0, 0))
        screen.blit(self.trail_surface, (0, 0))

        if self.show_fps:
            status = (
                f"FPS: {fps:5.1f} | N: {ps.count} | Gravity: {'ON' if ps.gravity_enabled else 'OFF'} | "
                f"Force: {ps.mouse_strength:.0f} | Mode: {'VEL' if self.mode else 'POINT'}"
            )
            if paused:
                status += " | PAUSED"
            text = self.font.render(status, True, (200, 255, 200))
            screen.blit(text, (8, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FluidVibe MVP - GPU particle fluid playground")
    p.add_argument("--particles", type=int, default=200_000, help="Target particle count (default: 200000)")
    p.add_argument("--width", type=int, default=1280, help="Window width")
    p.add_argument("--height", type=int, default=720, help="Window height")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("FluidVibe MVP (GPU)")
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    count = ParticleSystem.recommend_count(args.particles)
    ps = ParticleSystem(count, args.width, args.height)
    renderer = Renderer(args.width, args.height, show_fps=True)

    running = True
    paused = False
    mouse_mode = 0

    while running:
        dt = min(clock.tick(120) / 1000.0, 1 / 30)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                ps.resize(e.w, e.h)
                renderer.resize(e.w, e.h)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    ps.reset()
                elif e.key == pygame.K_g:
                    ps.gravity_enabled = not ps.gravity_enabled
                elif e.key == pygame.K_f:
                    renderer.show_fps = not renderer.show_fps
                elif e.key == pygame.K_c:
                    renderer.mode = 1 - renderer.mode
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    ps.reset(min(ps.count + 20_000, 500_000))
                elif e.key == pygame.K_MINUS:
                    ps.reset(max(ps.count - 20_000, 20_000))
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    mouse_mode = -1
                elif e.button == 3:
                    mouse_mode = 1
                elif e.button == 2:
                    ps.clear_around(pygame.mouse.get_pos(), radius=50.0)
                elif e.button == 4:
                    ps.mouse_strength *= 1.15
                elif e.button == 5:
                    ps.mouse_strength /= 1.15
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button in (1, 3):
                    mouse_mode = 0

        ps.step(dt, pygame.mouse.get_pos(), mouse_mode, paused)
        renderer.draw(screen, ps, clock.get_fps(), paused)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
