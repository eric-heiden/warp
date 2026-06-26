# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Program Smoothing
#
# Demonstrates Gaussian program smoothing for branchy Warp objectives. The
# example compares crisp Warp AD, finite differences, and score-function
# smoothing on a triangle intersection objective and on 2D billiards-style
# collision objectives where the optimized parameter is the initial ball
# velocity.
###########################################################################

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import warp as wp
import warp.optim

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


TRI_A = (-0.65, -0.42)
TRI_B = (0.65, -0.35)
TRI_C = (-0.05, 0.65)
TRI_A_WP = wp.constant(wp.vec2(TRI_A[0], TRI_A[1]))
TRI_B_WP = wp.constant(wp.vec2(TRI_B[0], TRI_B[1]))
TRI_C_WP = wp.constant(wp.vec2(TRI_C[0], TRI_C[1]))


@dataclass(frozen=True)
class CollisionScenario:
    name: str
    start: tuple[float, float]
    target: tuple[float, float]
    initial_velocity: tuple[float, float]
    circles: tuple[tuple[float, float, float], ...]
    steps: int
    dt: float
    damping: float
    restitution: float


SCENARIOS = (
    CollisionScenario(
        name="pinball_bank",
        start=(-0.86, -0.36),
        target=(0.78, 0.30),
        initial_velocity=(1.65, 0.12),
        circles=((-0.25, -0.08, 0.16), (0.08, 0.22, 0.17), (0.38, -0.16, 0.15)),
        steps=96,
        dt=0.018,
        damping=0.996,
        restitution=0.92,
    ),
    CollisionScenario(
        name="crowded_table",
        start=(-0.82, 0.30),
        target=(0.82, -0.30),
        initial_velocity=(1.35, -0.05),
        circles=((-0.38, 0.02, 0.13), (-0.08, -0.22, 0.16), (0.20, 0.12, 0.14), (0.48, -0.04, 0.12)),
        steps=108,
        dt=0.017,
        damping=0.995,
        restitution=0.90,
    ),
)


@wp.func
def triangle_edge_cross(a: wp.vec2, b: wp.vec2, p: wp.vec2):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@wp.func
def triangle_inside(p: wp.vec2):
    a = TRI_A_WP
    b = TRI_B_WP
    c = TRI_C_WP
    c0 = triangle_edge_cross(a, b, p)
    c1 = triangle_edge_cross(b, c, p)
    c2 = triangle_edge_cross(c, a, p)
    return c0 >= 0.0 and c1 >= 0.0 and c2 >= 0.0


@wp.func
def segment_distance_sq(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    ab = b - a
    denom = wp.dot(ab, ab)
    t = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
    q = a + t * ab
    d = p - q
    return wp.dot(d, d)


@wp.kernel
def triangle_intersection_loss_kernel(point: wp.array(dtype=float), loss: wp.array(dtype=float)):
    p = wp.vec2(point[0], point[1])
    if triangle_inside(p):
        loss[0] = 1.0
    else:
        loss[0] = 0.0


@wp.kernel
def triangle_distance_loss_kernel(point: wp.array(dtype=float), loss: wp.array(dtype=float)):
    p = wp.vec2(point[0], point[1])
    if triangle_inside(p):
        loss[0] = 0.0
    else:
        a = TRI_A_WP
        b = TRI_B_WP
        c = TRI_C_WP
        d0 = segment_distance_sq(p, a, b)
        d1 = segment_distance_sq(p, b, c)
        d2 = segment_distance_sq(p, c, a)
        loss[0] = wp.min(d0, wp.min(d1, d2))


@wp.kernel
def collision_loss_kernel(
    velocity: wp.array(dtype=float),
    circles: wp.array(dtype=float),
    circle_count: int,
    loss: wp.array(dtype=float),
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    steps: int,
    dt: float,
    damping: float,
    restitution: float,
):
    p = wp.vec2(start_x, start_y)
    v = wp.vec2(velocity[0], velocity[1])
    min_x = -0.96
    max_x = 0.96
    min_y = -0.54
    max_y = 0.54

    for _step in range(steps):
        p = p + v * dt

        if p[0] < min_x:
            p = wp.vec2(min_x, p[1])
            v = wp.vec2(-restitution * v[0], v[1])
        elif p[0] > max_x:
            p = wp.vec2(max_x, p[1])
            v = wp.vec2(-restitution * v[0], v[1])

        if p[1] < min_y:
            p = wp.vec2(p[0], min_y)
            v = wp.vec2(v[0], -restitution * v[1])
        elif p[1] > max_y:
            p = wp.vec2(p[0], max_y)
            v = wp.vec2(v[0], -restitution * v[1])

        for circle_index in range(circle_count):
            base = circle_index * 3
            center = wp.vec2(circles[base], circles[base + 1])
            radius = circles[base + 2]
            delta = p - center
            dist = wp.length(delta)
            if dist < radius:
                n = delta / wp.max(dist, 1.0e-6)
                p = center + n * radius
                vn = wp.dot(v, n)
                if vn < 0.0:
                    v = v - (1.0 + restitution) * vn * n

        v = v * damping

    target = wp.vec2(target_x, target_y)
    d = p - target
    speed = wp.vec2(velocity[0], velocity[1])
    loss[0] = wp.dot(d, d) + 0.0025 * wp.dot(speed, speed)


def make_triangle_loss(device: str, mode: str):
    kernel = triangle_intersection_loss_kernel if mode == "intersection" else triangle_distance_loss_kernel

    def loss_fn(params):
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        wp.launch(kernel, dim=1, inputs=[params[0], loss], device=device)
        return loss

    return loss_fn


def make_collision_loss(device: str, scenario: CollisionScenario):
    circle_data = np.array(scenario.circles, dtype=np.float32).reshape(-1)
    circles = wp.array(circle_data, dtype=float, device=device)

    def loss_fn(params):
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        wp.launch(
            collision_loss_kernel,
            dim=1,
            inputs=[
                params[0],
                circles,
                len(scenario.circles),
                loss,
                scenario.start[0],
                scenario.start[1],
                scenario.target[0],
                scenario.target[1],
                scenario.steps,
                scenario.dt,
                scenario.damping,
                scenario.restitution,
            ],
            device=device,
        )
        return loss

    return loss_fn


def evaluate_loss(loss_fn, x: np.ndarray, device: str) -> float:
    param = wp.array(x.astype(np.float32), dtype=float, requires_grad=True, device=device)
    return float(loss_fn([param]).numpy()[0])


def estimate_gradient(
    method: str,
    loss_fn,
    x: np.ndarray,
    device: str,
    samples: int,
    seed: int,
    smooth_sigma: float = 0.07,
):
    param = wp.array(x.astype(np.float32), dtype=float, requires_grad=True, device=device)
    if method == "autodiff":
        return warp.optim.smoothing.estimate_pathwise(loss_fn, [param], sigma=0.0, samples=1, seed=seed)
    if method == "finite_difference":
        return warp.optim.smoothing.estimate_finite_difference(
            loss_fn,
            [param],
            sigma=0.0,
            epsilon=2.0e-3,
            samples=1,
            seed=seed,
        )
    if method == "smooth_score":
        return warp.optim.smoothing.estimate_score_function(
            loss_fn,
            [param],
            sigma=smooth_sigma,
            samples=samples,
            seed=seed,
            antithetic=True,
        )
    if method == "smooth_fd":
        return warp.optim.smoothing.estimate_finite_difference(
            loss_fn,
            [param],
            sigma=smooth_sigma,
            epsilon=2.0e-2,
            samples=max(8, samples // 2),
            seed=seed,
            antithetic=True,
        )
    raise ValueError(f"Unknown method: {method}")


def gradient_norm(gradient: np.ndarray) -> float:
    return float(np.linalg.norm(gradient.reshape(-1)))


def clip_gradient(gradient: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        return gradient * (max_norm / norm)
    return gradient


def adam_step(x: np.ndarray, grad: np.ndarray, state: dict[str, np.ndarray | int], lr: float) -> np.ndarray:
    beta1 = 0.9
    beta2 = 0.99
    eps = 1.0e-8
    state["t"] = int(state["t"]) + 1
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * grad * grad
    m_hat = state["m"] / (1.0 - beta1 ** int(state["t"]))
    v_hat = state["v"] / (1.0 - beta2 ** int(state["t"]))
    return x - lr * m_hat / (np.sqrt(v_hat) + eps)


def optimize_velocity(loss_fn, scenario: CollisionScenario, device: str, samples: int, train_iters: int):
    methods = {
        "autodiff": {"lr": 0.045},
        "finite_difference": {"lr": 0.035},
        "smooth_score": {"lr": 0.070},
        "smooth_fd": {"lr": 0.070},
    }
    histories: dict[str, list[dict[str, float | list[float]]]] = {}
    final_velocities: dict[str, list[float]] = {}

    for method, config in methods.items():
        x = np.array(scenario.initial_velocity, dtype=np.float64)
        state = {"t": 0, "m": np.zeros_like(x), "v": np.zeros_like(x)}
        history = []
        for iteration in range(train_iters):
            estimate = estimate_gradient(
                method,
                loss_fn,
                x,
                device,
                samples=samples,
                seed=1000 + iteration,
                smooth_sigma=0.02,
            )
            gradient = estimate.gradients[0].numpy().astype(np.float64)
            gradient = clip_gradient(gradient, 8.0)
            crisp_loss = evaluate_loss(loss_fn, x, device)
            history.append(
                {
                    "iteration": iteration,
                    "crisp_loss": crisp_loss,
                    "estimate_value": float(estimate.value),
                    "grad_norm": gradient_norm(gradient),
                    "elapsed_ms": float(estimate.elapsed_time * 1000.0),
                }
            )
            x = adam_step(x, gradient, state, float(config["lr"]))
            x = np.clip(x, -2.6, 2.6)

        final_velocities[method] = [float(x[0]), float(x[1])]
        histories[method] = history

    return histories, final_velocities


def simulate_trajectory_np(velocity: np.ndarray, scenario: CollisionScenario) -> np.ndarray:
    p = np.array(scenario.start, dtype=np.float64)
    v = velocity.astype(np.float64).copy()
    positions = [p.copy()]
    min_x, max_x = -0.96, 0.96
    min_y, max_y = -0.54, 0.54

    for _ in range(scenario.steps):
        p = p + v * scenario.dt
        if p[0] < min_x:
            p[0] = min_x
            v[0] = -scenario.restitution * v[0]
        elif p[0] > max_x:
            p[0] = max_x
            v[0] = -scenario.restitution * v[0]
        if p[1] < min_y:
            p[1] = min_y
            v[1] = -scenario.restitution * v[1]
        elif p[1] > max_y:
            p[1] = max_y
            v[1] = -scenario.restitution * v[1]

        for cx, cy, radius in scenario.circles:
            center = np.array([cx, cy], dtype=np.float64)
            delta = p - center
            dist = np.linalg.norm(delta)
            if dist < radius:
                n = delta / max(dist, 1.0e-6)
                p = center + n * radius
                vn = float(np.dot(v, n))
                if vn < 0.0:
                    v = v - (1.0 + scenario.restitution) * vn * n

        v = v * scenario.damping
        positions.append(p.copy())

    return np.array(positions)


def triangle_numpy_value(point: np.ndarray, mode: str) -> float:
    a = np.array(TRI_A)
    b = np.array(TRI_B)
    c = np.array(TRI_C)

    def cross(u, v, p):
        return (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])

    inside = cross(a, b, point) >= 0.0 and cross(b, c, point) >= 0.0 and cross(c, a, point) >= 0.0
    if mode == "intersection":
        return 1.0 if inside else 0.0
    if inside:
        return 0.0

    def seg_dist_sq(u, v):
        ab = v - u
        t = np.clip(np.dot(point - u, ab) / np.dot(ab, ab), 0.0, 1.0)
        q = u + t * ab
        d = point - q
        return float(np.dot(d, d))

    return min(seg_dist_sq(a, b), seg_dist_sq(b, c), seg_dist_sq(c, a))


def compute_triangle_field(device: str, samples: int, grid_size: int):
    loss_fn = make_triangle_loss(device, "intersection")
    xs = np.linspace(-0.9, 0.9, grid_size)
    ys = np.linspace(-0.7, 0.8, grid_size)
    records = []

    for y in ys:
        for x in xs:
            point = np.array([x, y], dtype=np.float64)
            row: dict[str, Any] = {
                "x": float(x),
                "y": float(y),
                "value": triangle_numpy_value(point, "intersection"),
            }
            for method in ("autodiff", "smooth_score", "smooth_fd"):
                estimate = estimate_gradient(method, loss_fn, point, device, samples=samples, seed=17)
                grad = estimate.gradients[0].numpy().astype(float)
                row[f"{method}_gx"] = float(grad[0])
                row[f"{method}_gy"] = float(grad[1])
                row[f"{method}_value"] = float(estimate.value)
            records.append(row)

    return records


def compute_triangle_probe(device: str, samples: int):
    points = {
        "inside_near_edge": np.array([0.0, -0.36], dtype=np.float64),
        "outside_near_edge": np.array([0.0, -0.41], dtype=np.float64),
        "vertex_neighborhood": np.array([-0.08, 0.58], dtype=np.float64),
    }
    output: dict[str, Any] = {}
    for mode in ("intersection", "distance"):
        loss_fn = make_triangle_loss(device, mode)
        output[mode] = {}
        for name, point in points.items():
            output[mode][name] = {}
            for method in ("autodiff", "finite_difference", "smooth_score", "smooth_fd"):
                estimate = estimate_gradient(method, loss_fn, point, device, samples=samples, seed=29)
                grad = estimate.gradients[0].numpy().astype(float)
                output[mode][name][method] = {
                    "value": float(estimate.value),
                    "gradient": [float(grad[0]), float(grad[1])],
                    "grad_norm": gradient_norm(grad),
                    "elapsed_ms": float(estimate.elapsed_time * 1000.0),
                }
    return output


def estimator_variance_sweep(device: str, samples_values: list[int]):
    triangle_loss = make_triangle_loss(device, "intersection")
    triangle_point = np.array([0.0, -0.37], dtype=np.float64)
    collision_loss = make_collision_loss(device, SCENARIOS[0])
    collision_velocity = np.array(SCENARIOS[0].initial_velocity, dtype=np.float64)
    rows = []

    for samples in samples_values:
        for problem, loss_fn, x in (
            ("triangle_intersection", triangle_loss, triangle_point),
            ("pinball_bank", collision_loss, collision_velocity),
        ):
            smooth_sigma = 0.02 if problem == "pinball_bank" else 0.07
            estimate = estimate_gradient(
                "smooth_score",
                loss_fn,
                x,
                device,
                samples=samples,
                seed=123,
                smooth_sigma=smooth_sigma,
            )
            variance = estimate.gradient_variance[0].numpy().astype(float)
            rows.append(
                {
                    "problem": problem,
                    "samples": samples,
                    "estimator_variance_norm": gradient_norm(variance) / float(samples),
                    "gradient_norm": gradient_norm(estimate.gradients[0].numpy().astype(float)),
                    "elapsed_ms": float(estimate.elapsed_time * 1000.0),
                }
            )
    return rows


def benchmark_methods(device: str, samples: int, repeats: int):
    rows = []
    process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    problems = [
        ("triangle_intersection", make_triangle_loss(device, "intersection"), np.array([0.0, -0.37], dtype=np.float64))
    ]
    problems.extend(
        (
            scenario.name,
            make_collision_loss(device, scenario),
            np.array(scenario.initial_velocity, dtype=np.float64),
        )
        for scenario in SCENARIOS
    )

    for problem, loss_fn, x in problems:
        for method in ("autodiff", "finite_difference", "smooth_score", "smooth_fd"):
            times = []
            peaks = []
            rss_deltas = []
            grad_norms = []
            values = []
            for repeat in range(repeats):
                rss_before = process.memory_info().rss if process is not None else 0
                tracemalloc.start()
                start = time.perf_counter()
                smooth_sigma = 0.02 if problem != "triangle_intersection" else 0.07
                estimate = estimate_gradient(
                    method,
                    loss_fn,
                    x,
                    device,
                    samples=samples,
                    seed=700 + repeat,
                    smooth_sigma=smooth_sigma,
                )
                elapsed = time.perf_counter() - start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rss_after = process.memory_info().rss if process is not None else 0
                times.append(elapsed * 1000.0)
                peaks.append(float(peak))
                rss_deltas.append(float(max(0, rss_after - rss_before)))
                grad_norms.append(gradient_norm(estimate.gradients[0].numpy().astype(float)))
                values.append(float(estimate.value))

            rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "samples": samples if method in ("smooth_score", "smooth_fd") else 1,
                    "mean_ms": float(np.mean(times)),
                    "std_ms": float(np.std(times)),
                    "python_peak_kib": float(np.mean(peaks) / 1024.0),
                    "rss_delta_kib": float(np.mean(rss_deltas) / 1024.0),
                    "value": float(np.mean(values)),
                    "grad_norm": float(np.mean(grad_norms)),
                }
            )
    return rows


def make_plots(output_dir: Path, results: dict[str, Any]):
    if not MATPLOTLIB_AVAILABLE:
        return []

    written = []
    colors = {
        "autodiff": "#dc2626",
        "finite_difference": "#7c3aed",
        "smooth_score": "#0f766e",
        "smooth_fd": "#2563eb",
    }

    field = results["triangle_field"]
    xs = sorted({row["x"] for row in field})
    ys = sorted({row["y"] for row in field})
    xx, yy = np.meshgrid(xs, ys)
    value = np.array([row["value"] for row in field]).reshape(len(ys), len(xs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, method, title in zip(
        axes,
        ("autodiff", "smooth_score", "smooth_fd"),
        ("Crisp AD", "Score smoothing", "Smoothed finite difference"),
        strict=True,
    ):
        ax.contourf(xx, yy, value, levels=[-0.1, 0.5, 1.1], colors=["#f8fafc", "#d1fae5"], alpha=0.9)
        gx = np.array([row[f"{method}_gx"] for row in field]).reshape(len(ys), len(xs))
        gy = np.array([row[f"{method}_gy"] for row in field]).reshape(len(ys), len(xs))
        ax.quiver(xx, yy, gx, gy, color=colors.get(method, "#111827"), angles="xy", scale_units="xy", scale=24)
        tri = np.array([TRI_A, TRI_B, TRI_C, TRI_A])
        ax.plot(tri[:, 0], tri[:, 1], color="#111827", linewidth=1.5)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.7, 0.8)
    path = output_dir / "triangle_gradients.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        for method, history in results["collision_optimization"][scenario.name]["histories"].items():
            ax.plot(
                [r["iteration"] for r in history],
                [r["crisp_loss"] for r in history],
                label=method,
                color=colors[method],
            )
        ax.set_title(scenario.name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("crisp loss")
        ax.set_yscale("log")
        ax.grid(True, color="#e5e7eb")
    axes[0].legend(loc="upper right", fontsize=8)
    path = output_dir / "collision_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        ax.add_patch(plt.Rectangle((-0.96, -0.54), 1.92, 1.08, fill=False, edgecolor="#111827", linewidth=1.5))
        for cx, cy, radius in scenario.circles:
            ax.add_patch(plt.Circle((cx, cy), radius, color="#dbeafe", ec="#2563eb", alpha=0.85))
        ax.scatter([scenario.start[0]], [scenario.start[1]], marker="o", color="#111827", label="start")
        ax.scatter([scenario.target[0]], [scenario.target[1]], marker="*", s=150, color="#f59e0b", label="target")
        for method, velocity in results["collision_optimization"][scenario.name]["final_velocities"].items():
            trajectory = simulate_trajectory_np(np.array(velocity, dtype=np.float64), scenario)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[method], linewidth=1.8, label=method)
        ax.set_title(scenario.name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.04, 1.04)
        ax.set_ylim(-0.62, 0.62)
        ax.grid(True, color="#e5e7eb")
    axes[0].legend(loc="lower left", fontsize=8)
    path = output_dir / "collision_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    for problem in sorted({row["problem"] for row in results["variance_sweep"]}):
        rows = [row for row in results["variance_sweep"] if row["problem"] == problem]
        ax.plot(
            [row["samples"] for row in rows],
            [row["estimator_variance_norm"] for row in rows],
            marker="o",
            label=problem,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("score samples")
    ax.set_ylabel("estimated gradient variance norm / samples")
    ax.grid(True, color="#e5e7eb", which="both")
    ax.legend()
    path = output_dir / "gradient_variance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bench = results["benchmarks"]
    labels = [f"{row['problem']}\n{row['method']}" for row in bench]
    ax.bar(np.arange(len(bench)), [row["mean_ms"] for row in bench], color="#0f766e")
    ax.set_yscale("log")
    ax.set_ylabel("mean wall time (ms)")
    ax.set_xticks(np.arange(len(bench)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", color="#e5e7eb", which="both")
    path = output_dir / "benchmark_speed.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    return written


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    wp.set_device(args.device)
    device = str(wp.get_device(args.device))

    if args.quick:
        triangle_grid = 7
        train_iters = min(args.train_iters, 12)
        samples = min(args.samples, 32)
        benchmark_repeats = 1
        variance_samples = [8, 16, 32]
    else:
        triangle_grid = args.triangle_grid
        train_iters = args.train_iters
        samples = args.samples
        benchmark_repeats = args.benchmark_repeats
        variance_samples = [16, 32, 64, 128, 256]

    results: dict[str, Any] = {
        "metadata": {
            "device": device,
            "warp_version": wp.__version__,
            "cuda_enabled": any(wp.get_device(warp_device).is_cuda for warp_device in wp.get_devices()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "samples": samples,
            "train_iters": train_iters,
            "triangle_grid": triangle_grid,
            "note": "CUDA benchmarks were not run if this Warp build reports cuda_enabled=false.",
        },
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }

    print("Computing triangle probe...")
    results["triangle_probe"] = compute_triangle_probe(device, samples=samples)

    print("Computing triangle field...")
    results["triangle_field"] = compute_triangle_field(device, samples=samples, grid_size=triangle_grid)

    print("Optimizing collision examples...")
    collision_optimization = {}
    for scenario in SCENARIOS:
        loss_fn = make_collision_loss(device, scenario)
        histories, final_velocities = optimize_velocity(
            loss_fn,
            scenario,
            device,
            samples=samples,
            train_iters=train_iters,
        )
        collision_optimization[scenario.name] = {
            "histories": histories,
            "final_velocities": final_velocities,
            "final_losses": {
                method: evaluate_loss(loss_fn, np.array(velocity, dtype=np.float64), device)
                for method, velocity in final_velocities.items()
            },
        }
    results["collision_optimization"] = collision_optimization

    print("Computing gradient variance sweep...")
    results["variance_sweep"] = estimator_variance_sweep(device, variance_samples)

    print("Benchmarking estimators...")
    results["benchmarks"] = benchmark_methods(device, samples=samples, repeats=benchmark_repeats)

    plots = make_plots(output_dir, results)
    results["plots"] = plots

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {results_path}")
    for plot in plots:
        print(f"Wrote {output_dir / plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cpu", help="Override the default Warp device.")
    parser.add_argument(
        "--output-dir", type=str, default="program_smoothing_outputs", help="Directory for JSON and plots."
    )
    parser.add_argument("--samples", type=int, default=128, help="Smoothing samples for score-function estimates.")
    parser.add_argument("--train-iters", type=int, default=50, help="Velocity optimization iterations per method.")
    parser.add_argument("--triangle-grid", type=int, default=11, help="Triangle vector-field grid resolution.")
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Number of benchmark repeats per method.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced workload for smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Accepted for consistency with other examples.")
    run(parser.parse_args())
