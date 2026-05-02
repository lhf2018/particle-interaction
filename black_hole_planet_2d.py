"""
2D 黑洞吞噬行星模拟（独立脚本，matplotlib 滑块调参，无 Web 前端）。
粒子步进优先在 GPU（CuPy/CUDA）上用 float32 批量积分；不可用时回退 NumPy。
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import cupy as cp

    _ = cp.array([1.0], dtype=cp.float32)
    CUDA_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False


def get_xp() -> type[np] | type:
    return cp if CUDA_AVAILABLE else np


COL_BG = "#0a0e1a"
COL_BH_CORE = "#050508"
COL_ACCRETION = "#ff6b00"
COL_ACCRETION_INNER = "#ffcc00"
COL_PLANET = "#4ecdc4"
COL_PLANET_CORE = "#2dd4bf"
COL_DISK = "#ff8c42"
COL_TEXT = "#e8eef7"
COL_HORIZON = "#9d4edd"

# 滑块标题（短）+ 同行完整说明
SLIDER_ROWS: tuple[tuple[str, str], ...] = (
    ("G", "万有引力强度系数。越大黑洞与行星吸引越强，轨道更易弯曲、吞噬更快。"),
    ("黑洞质量", "越大引力越强；吸积盘与吞噬圈的视觉半径随之变大。"),
    ("行星自引力", "粒子间弱吸引，越大越不易被潮汐扯散（存活多时计算量≈n²，GPU 更合适）。"),
    ("行星半径", "初始圆盘半径；越大越「胖」，越易被潮汐撕裂。"),
    ("轨道半径", "行星中心到黑洞的初始距离；越小越易被捕获。"),
    ("切向速度", "初始切向速度；过小会直接下落，适中可先环绕再被吸入。"),
    ("时间步长", "每子步积分步长；越大演化越快，过大易数值不稳定。"),
    ("粒子数", "圆盘采样点数（滑块可调上限）；越多越实心。自引力开启时计算量≈n²，请优先用 GPU；绘图会自动抽样。"),
    ("视界比例", "吞噬半径 = 系数×√黑洞质量；越大越易被判定吞噬。"),
)


def dtype_f(xp):
    """当前后端下的 float32 标量 / dtype。"""
    return xp.float32


def soft_accel(xp, dx, dy, mass: float, eps: float):
    dt = dtype_f(xp)
    eps = dt(eps)
    r2 = dx * dx + dy * dy + eps * eps
    inv_r3 = r2 ** (-1.5)
    g = dt(mass) * inv_r3
    return -g * dx, -g * dy


def as_float(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


class Simulation:
    def __init__(self) -> None:
        self.xp = get_xp()
        self.G = 1200.0
        self.dt = 0.012
        self.soft = 8.0
        self.n_particles = 5000 if CUDA_AVAILABLE else 1200
        self.bh_mass = 180.0
        self.planet_mass = 1.0
        self.planet_radius = 42.0
        self.orbit_radius = 220.0
        self.v_tangential = 95.0
        self.event_horizon_scale = 0.35
        self.bh_x = 0.0
        self.bh_y = 0.0
        # 每帧物理子步；自引力仅在偶数子步计算以削峰 O(n²)
        self.substeps_per_frame = 3 if CUDA_AVAILABLE else 2
        self._n_total = 0

        self._seed = 42
        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.alive = None

    def reset(self) -> None:
        xp = self.xp
        dt = dtype_f(xp)
        n = int(self.n_particles)
        r0 = float(self.planet_radius)
        orbit_r = float(self.orbit_radius)
        cx = self.bh_x + orbit_r
        cy = self.bh_y

        rng = np.random.default_rng(self._seed)
        t = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        rr = (r0 * np.sqrt(rng.uniform(0, 1, n))).astype(np.float32)
        px = (cx + rr * np.cos(t)).astype(np.float32)
        py = (cy + rr * np.sin(t)).astype(np.float32)
        rhat_x = (cx - self.bh_x) / orbit_r
        rhat_y = (cy - self.bh_y) / orbit_r
        tx, ty = -rhat_y, rhat_x
        v0 = np.float32(self.v_tangential)
        vx = np.full(n, tx * v0, dtype=np.float32)
        vy = np.full(n, ty * v0, dtype=np.float32)
        alive = np.ones(n, dtype=np.bool_)

        if CUDA_AVAILABLE and xp is not np:
            self.px = cp.asarray(px, dtype=cp.float32)
            self.py = cp.asarray(py, dtype=cp.float32)
            self.vx = cp.asarray(vx, dtype=cp.float32)
            self.vy = cp.asarray(vy, dtype=cp.float32)
            self.alive = cp.asarray(alive)
        else:
            self.px = px
            self.py = py
            self.vx = vx
            self.vy = vy
            self.alive = alive.astype(bool)
        self._n_total = n

    def step(self, apply_self_gravity: bool = True) -> None:
        if self.px is None:
            return
        xp = self.xp
        dt = dtype_f(xp)
        m_bh = float(self.bh_mass)
        m_pl = float(self.planet_mass)
        eps = float(self.soft)
        step_dt = float(self.dt)
        G = float(self.G)

        dx = self.px - dt(self.bh_x)
        dy = self.py - dt(self.bh_y)
        ax_bh, ay_bh = soft_accel(xp, dx, dy, G * m_bh, eps)
        ax = ax_bh.copy()
        ay = ay_bh.copy()

        if apply_self_gravity and m_pl > 1e-6:
            idx = xp.flatnonzero(self.alive)
            if idx.size > 0:
                px_a = self.px[idx]
                py_a = self.py[idx]
                ddx = px_a[:, None] - px_a[None, :]
                ddy = py_a[:, None] - py_a[None, :]
                xp.fill_diagonal(ddx, dt(0))
                xp.fill_diagonal(ddy, dt(0))
                r2 = ddx * ddx + ddy * ddy + dt(eps) ** 2
                inv_r3 = r2 ** (-1.5)
                k = dt(G * m_pl * 0.0008)
                ax_a = k * xp.sum(ddx * inv_r3, axis=1)
                ay_a = k * xp.sum(ddy * inv_r3, axis=1)
                ax[idx] = ax[idx] + ax_a
                ay[idx] = ay[idx] + ay_a

        dtm = dt(step_dt)
        self.vx = self.vx + ax * dtm
        self.vy = self.vy + ay * dtm
        self.px = self.px + self.vx * dtm
        self.py = self.py + self.vy * dtm

        r = xp.sqrt((self.px - dt(self.bh_x)) ** 2 + (self.py - dt(self.bh_y)) ** 2)
        r_s = dt(float(self.event_horizon_scale) * (max(m_bh, 1e-6) ** 0.5))
        swallowed = self.alive & (r < r_s)
        self.alive = self.alive & (~swallowed)

    def eaten_count(self) -> int:
        if self.alive is None:
            return 0
        xp = self.xp
        return int(as_float(xp.sum(~self.alive)))

    def alive_count(self) -> int:
        if self.alive is None:
            return 0
        xp = self.xp
        return int(as_float(xp.sum(self.alive)))


def alive_xy_stats(sim: Simulation, max_render: int = 2200) -> tuple[np.ndarray, int]:
    """存活粒子坐标 + 存活数（单次 GPU 归约）；绘图抽样减轻 matplotlib 压力。"""
    empty = np.empty((0, 2), dtype=np.float32)
    if sim.px is None or sim.alive is None:
        return empty, 0
    xp = sim.xp
    if CUDA_AVAILABLE and xp is not np:
        m = sim.alive
        n_alive = int(cp.sum(m))
        if n_alive == 0:
            return empty, 0
        px_a = sim.px[m]
        py_a = sim.py[m]
        if n_alive > max_render:
            stride = int(np.ceil(n_alive / max_render))
            px_a = px_a[::stride]
            py_a = py_a[::stride]
        xy = cp.asnumpy(cp.column_stack((px_a, py_a))).astype(np.float32, copy=False)
        return xy, n_alive
    m = np.asarray(sim.alive)
    n_alive = int(np.sum(m))
    if n_alive == 0:
        return empty, 0
    px_a = np.asarray(sim.px[m], dtype=np.float32)
    py_a = np.asarray(sim.py[m], dtype=np.float32)
    if n_alive > max_render:
        stride = int(np.ceil(n_alive / max_render))
        px_a = px_a[::stride]
        py_a = py_a[::stride]
    return np.column_stack((px_a, py_a)), n_alive


def main() -> None:
    plt.rcParams.update(
        {
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "agg.path.chunksize": 25000,
        }
    )

    sim = Simulation()
    sim.reset()
    xp = sim.xp
    backend_name = "CUDA float32 · %d 子步/帧" % sim.substeps_per_frame
    if not (CUDA_AVAILABLE and xp is not np):
        backend_name = "CPU NumPy · %d 子步/帧" % sim.substeps_per_frame

    fig = plt.figure(figsize=(11.5, 10.2), facecolor=COL_BG)

    # 双列排布滑块，避免 9 行过高导致底部「粒子数」等被窗口裁掉
    panel_top = 0.382
    row_h = 0.036
    plot_bottom = panel_top + 0.018
    plot_h = 1.0 - plot_bottom - 0.048

    ax = fig.add_axes((0.06, plot_bottom, 0.88, plot_h), facecolor=COL_BG)
    ax.set_aspect("equal")
    lim = 320
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")

    (horizon_line,) = ax.plot([], [], color=COL_HORIZON, lw=2.0, zorder=5)
    accretion = plt.Circle((0, 0), 1, fc="none", ec=COL_ACCRETION, lw=3.0, zorder=4)
    accretion_inner = plt.Circle((0, 0), 1, fc="none", ec=COL_ACCRETION_INNER, lw=1.5, zorder=4)
    bh_disk = plt.Circle((0, 0), 1, fc=COL_DISK, ec=COL_ACCRETION, lw=2.0, zorder=3)
    bh_core = plt.Circle((0, 0), 1, fc=COL_BH_CORE, ec="#1a1a24", lw=1.5, zorder=6)
    for c in (accretion, accretion_inner, bh_disk, bh_core):
        ax.add_patch(c)

    off0, na0 = alive_xy_stats(sim)
    # 像素点 Line2D 比 scatter(PathCollection) 刷新轻得多
    (particles_ln,) = ax.plot(
        off0[:, 0] if len(off0) else [],
        off0[:, 1] if len(off0) else [],
        ls="none",
        marker=",",
        color=COL_PLANET,
        alpha=0.92,
        zorder=7,
    )

    title = ax.set_title("", color=COL_TEXT, fontsize=11, fontfamily="sans-serif")
    ax.text(
        0.02,
        0.98,
        "存活 / 已吞噬",
        transform=ax.transAxes,
        color=COL_TEXT,
        fontsize=10,
        va="top",
    )
    stats = ax.text(
        0.02,
        0.91,
        "",
        transform=ax.transAxes,
        color=COL_ACCRETION_INNER,
        fontsize=10,
        va="top",
        family="monospace",
    )
    ax.text(
        0.98,
        0.98,
        backend_name,
        transform=ax.transAxes,
        color="#64748b",
        fontsize=8.5,
        va="top",
        ha="right",
    )

    # GPU：自引力时显存随 n² 增长，上限可按显卡酌情再改大
    n_part_max = 18_000 if CUDA_AVAILABLE else 4500
    val_specs = (
        (200, 4000, float(sim.G), 50),
        (20, 500, float(sim.bh_mass), 5),
        (0.0, 5.0, float(sim.planet_mass), 0.1),
        (15, 90, float(sim.planet_radius), 1),
        (120, 300, float(sim.orbit_radius), 5),
        (40, 160, float(sim.v_tangential), 1),
        (0.004, 0.028, float(sim.dt), 0.001),
        (500, n_part_max, min(float(sim.n_particles), float(n_part_max)), 100),
        (0.15, 0.6, float(sim.event_horizon_scale), 0.01),
    )
    colors = (
        COL_PLANET,
        COL_ACCRETION,
        COL_PLANET_CORE,
        COL_PLANET,
        "#94a3b8",
        "#94a3b8",
        "#64748b",
        "#64748b",
        COL_HORIZON,
    )

    specs = list(zip(SLIDER_ROWS, val_specs, colors, strict=True))

    def place_slider(spec_idx: int, row: int, x0: float, w_sl: float, x_txt: float) -> Slider:
        (short_label, desc), (vmin, vmax, v0, step), color = specs[spec_idx]
        yb = panel_top - (row + 1) * row_h
        ax_sl = fig.add_axes((x0, yb, w_sl, 0.024), facecolor="#1e293b")
        sl = Slider(ax_sl, short_label, vmin, vmax, valinit=v0, valstep=step, color=color)
        fig.text(
            x_txt,
            yb + 0.012,
            desc,
            ha="left",
            va="center",
            fontsize=6.85,
            color="#cbd5e1",
            wrap=False,
        )
        return sl

    sliders = [None] * 9
    # 左列 | 右列（同一行）；粒子数在右列第 3 行，不会被挤出窗口底部）
    pairs = ((0, 5), (1, 6), (2, 7), (3, 8))
    for row, (i_left, i_right) in enumerate(pairs):
        sliders[i_left] = place_slider(i_left, row, 0.045, 0.165, 0.215)
        sliders[i_right] = place_slider(i_right, row, 0.485, 0.165, 0.655)
    sliders[4] = place_slider(4, 4, 0.045, 0.165, 0.215)

    sl_g, sl_bh, sl_plm, sl_pr, sl_or, sl_vt, sl_dt, sl_np, sl_rs = sliders

    need_reset = {"flag": False}

    def on_change(_: float | None = None) -> None:
        sim.G = float(sl_g.val)
        sim.bh_mass = float(sl_bh.val)
        sim.planet_mass = float(sl_plm.val)
        sim.planet_radius = float(sl_pr.val)
        sim.orbit_radius = float(sl_or.val)
        sim.v_tangential = float(sl_vt.val)
        sim.dt = float(sl_dt.val)
        sim.n_particles = int(sl_np.val)
        sim.event_horizon_scale = float(sl_rs.val)
        need_reset["flag"] = True

    for s in sliders:
        s.on_changed(on_change)

    def update_bh_artist() -> None:
        m = float(sim.bh_mass)
        r_h = float(sim.event_horizon_scale) * (max(m, 1e-6) ** 0.5)
        r_disk = r_h * 2.15
        r_acc = r_h * 2.85
        r_acc_in = r_h * 2.4
        bh_core.set_center((sim.bh_x, sim.bh_y))
        bh_core.set_radius(r_h * 0.92)
        bh_disk.set_center((sim.bh_x, sim.bh_y))
        bh_disk.set_radius(r_disk)
        accretion.set_center((sim.bh_x, sim.bh_y))
        accretion.set_radius(r_acc)
        accretion_inner.set_center((sim.bh_x, sim.bh_y))
        accretion_inner.set_radius(r_acc_in)
        theta = np.linspace(0, 2 * np.pi, 120)
        horizon_line.set_data(sim.bh_x + r_h * np.cos(theta), sim.bh_y + r_h * np.sin(theta))

    def refresh_particles() -> int:
        off, n_alive = alive_xy_stats(sim)
        if len(off):
            particles_ln.set_data(off[:, 0], off[:, 1])
        else:
            particles_ln.set_data([], [])
        return n_alive

    def do_reset() -> None:
        sim.reset()
        na = refresh_particles()
        update_bh_artist()
        stats.set_text(f"{na:5d}  /  {sim._n_total - na:5d}")

    update_bh_artist()

    _title_static = "2D 黑洞吞噬行星 · 调滑块后自动重置 · 双列参数（旁为说明）"

    def frame(_: int) -> None:
        if need_reset["flag"]:
            need_reset["flag"] = False
            do_reset()
            return
        for si in range(sim.substeps_per_frame):
            use_sg = (sim.planet_mass > 1e-6) and (si % 2 == 0)
            sim.step(apply_self_gravity=use_sg)
        na = refresh_particles()
        # 黑洞几何仅在 reset / 改滑块时变化，每帧不重绘以省 CPU
        stats.set_text(f"{na:5d}  /  {sim._n_total - na:5d}")

    title.set_text(_title_static)

    # interval=0 尽快调度下一帧（实际帧率受物理与绘图限制）
    _ani = FuncAnimation(fig, frame, interval=0, blit=False, cache_frame_data=False)

    stats.set_text(f"{na0:5d}  /  {sim._n_total - na0:5d}")
    plt.show()


if __name__ == "__main__":
    main()
