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
def triangle_intersection_loss_batched_kernel(points: wp.array2d(dtype=float), losses: wp.array(dtype=float)):
    sample = wp.tid()
    p = wp.vec2(points[sample, 0], points[sample, 1])
    if triangle_inside(p):
        losses[sample] = 1.0
    else:
        losses[sample] = 0.0


@wp.kernel
def triangle_distance_loss_batched_kernel(points: wp.array2d(dtype=float), losses: wp.array(dtype=float)):
    sample = wp.tid()
    p = wp.vec2(points[sample, 0], points[sample, 1])
    if triangle_inside(p):
        losses[sample] = 0.0
    else:
        a = TRI_A_WP
        b = TRI_B_WP
        c = TRI_C_WP
        d0 = segment_distance_sq(p, a, b)
        d1 = segment_distance_sq(p, b, c)
        d2 = segment_distance_sq(p, c, a)
        losses[sample] = wp.min(d0, wp.min(d1, d2))


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


@wp.kernel
def collision_loss_batched_kernel(
    velocities: wp.array2d(dtype=float),
    circles: wp.array(dtype=float),
    circle_count: int,
    losses: wp.array(dtype=float),
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    steps: int,
    dt: float,
    damping: float,
    restitution: float,
):
    sample = wp.tid()
    p = wp.vec2(start_x, start_y)
    v = wp.vec2(velocities[sample, 0], velocities[sample, 1])
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
    speed = wp.vec2(velocities[sample, 0], velocities[sample, 1])
    losses[sample] = wp.dot(d, d) + 0.0025 * wp.dot(speed, speed)


def make_triangle_loss(device: str, mode: str):
    kernel = triangle_intersection_loss_kernel if mode == "intersection" else triangle_distance_loss_kernel

    def loss_fn(params):
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        wp.launch(kernel, dim=1, inputs=[params[0], loss], device=device)
        return loss

    return loss_fn


def make_triangle_loss_batched(device: str, mode: str):
    kernel = (
        triangle_intersection_loss_batched_kernel if mode == "intersection" else triangle_distance_loss_batched_kernel
    )

    def loss_fn(params):
        samples = params[0].shape[0]
        loss = wp.zeros(samples, dtype=float, device=device)
        wp.launch(kernel, dim=samples, inputs=[params[0], loss], device=device)
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


def make_collision_loss_batched(device: str, scenario: CollisionScenario):
    circle_data = np.array(scenario.circles, dtype=np.float32).reshape(-1)
    circles = wp.array(circle_data, dtype=float, device=device)

    def loss_fn(params):
        samples = params[0].shape[0]
        loss = wp.zeros(samples, dtype=float, device=device)
        wp.launch(
            collision_loss_batched_kernel,
            dim=samples,
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


def evaluate_losses_batched(batched_loss_fn, xs: np.ndarray, device: str) -> np.ndarray:
    param = wp.array(xs.astype(np.float32), dtype=float, device=device)
    return batched_loss_fn([param]).numpy().astype(np.float64)


def estimate_gradient(
    method: str,
    loss_fn,
    x: np.ndarray,
    device: str,
    samples: int,
    seed: int,
    smooth_sigma: float = 0.07,
    batched_loss_fn=None,
    score_backend: str = "batched",
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
        if score_backend == "batched" and batched_loss_fn is not None:
            return warp.optim.smoothing.estimate_score_function_batched(
                batched_loss_fn,
                [param],
                sigma=smooth_sigma,
                samples=samples,
                seed=seed,
                antithetic=True,
            )
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


def optimize_velocity(
    loss_fn,
    batched_loss_fn,
    scenario: CollisionScenario,
    device: str,
    samples: int,
    train_iters: int,
    score_backend: str,
):
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
                batched_loss_fn=batched_loss_fn,
                score_backend=score_backend,
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
                    "velocity": [float(x[0]), float(x[1])],
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


def compute_triangle_field(device: str, samples: int, grid_size: int, score_backend: str):
    loss_fn = make_triangle_loss(device, "intersection")
    batched_loss_fn = make_triangle_loss_batched(device, "intersection")
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
                estimate = estimate_gradient(
                    method,
                    loss_fn,
                    point,
                    device,
                    samples=samples,
                    seed=17,
                    batched_loss_fn=batched_loss_fn,
                    score_backend=score_backend,
                )
                grad = estimate.gradients[0].numpy().astype(float)
                row[f"{method}_gx"] = float(grad[0])
                row[f"{method}_gy"] = float(grad[1])
                row[f"{method}_value"] = float(estimate.value)
            records.append(row)

    return records


def compute_triangle_probe(device: str, samples: int, score_backend: str):
    points = {
        "inside_near_edge": np.array([0.0, -0.36], dtype=np.float64),
        "outside_near_edge": np.array([0.0, -0.41], dtype=np.float64),
        "vertex_neighborhood": np.array([-0.08, 0.58], dtype=np.float64),
    }
    output: dict[str, Any] = {}
    for mode in ("intersection", "distance"):
        loss_fn = make_triangle_loss(device, mode)
        batched_loss_fn = make_triangle_loss_batched(device, mode)
        output[mode] = {}
        for name, point in points.items():
            output[mode][name] = {}
            for method in ("autodiff", "finite_difference", "smooth_score", "smooth_fd"):
                estimate = estimate_gradient(
                    method,
                    loss_fn,
                    point,
                    device,
                    samples=samples,
                    seed=29,
                    batched_loss_fn=batched_loss_fn,
                    score_backend=score_backend,
                )
                grad = estimate.gradients[0].numpy().astype(float)
                output[mode][name][method] = {
                    "value": float(estimate.value),
                    "gradient": [float(grad[0]), float(grad[1])],
                    "grad_norm": gradient_norm(grad),
                    "elapsed_ms": float(estimate.elapsed_time * 1000.0),
                }
    return output


def estimator_variance_sweep(device: str, samples_values: list[int], score_backend: str):
    triangle_loss = make_triangle_loss(device, "intersection")
    triangle_batched_loss = make_triangle_loss_batched(device, "intersection")
    triangle_point = np.array([0.0, -0.37], dtype=np.float64)
    collision_loss = make_collision_loss(device, SCENARIOS[0])
    collision_batched_loss = make_collision_loss_batched(device, SCENARIOS[0])
    collision_velocity = np.array(SCENARIOS[0].initial_velocity, dtype=np.float64)
    rows = []

    for samples in samples_values:
        for problem, loss_fn, batched_loss_fn, x in (
            ("triangle_intersection", triangle_loss, triangle_batched_loss, triangle_point),
            ("pinball_bank", collision_loss, collision_batched_loss, collision_velocity),
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
                batched_loss_fn=batched_loss_fn,
                score_backend=score_backend,
            )
            variance = estimate.gradient_variance[0].numpy().astype(float)
            variance_of_mean = variance / float(samples)
            rows.append(
                {
                    "problem": problem,
                    "samples": samples,
                    "estimator_variance_norm": gradient_norm(variance_of_mean),
                    "standard_error_norm": float(np.sqrt(max(0.0, np.sum(variance_of_mean)))),
                    "gradient_norm": gradient_norm(estimate.gradients[0].numpy().astype(float)),
                    "elapsed_ms": float(estimate.elapsed_time * 1000.0),
                }
            )
    return rows


def score_sample_gradient_sweep(
    device: str,
    samples_values: list[int],
    seeds: int,
    reference_samples: int,
    score_backend: str,
):
    problems = [
        (
            "triangle_intersection",
            make_triangle_loss(device, "intersection"),
            make_triangle_loss_batched(device, "intersection"),
            np.array([0.0, -0.37], dtype=np.float64),
            0.07,
        )
    ]
    problems.extend(
        (
            scenario.name,
            make_collision_loss(device, scenario),
            make_collision_loss_batched(device, scenario),
            np.array(scenario.initial_velocity, dtype=np.float64),
            0.02,
        )
        for scenario in SCENARIOS
    )

    rows = []
    references = {}
    for problem, loss_fn, batched_loss_fn, x, smooth_sigma in problems:
        reference = estimate_gradient(
            "smooth_score",
            loss_fn,
            x,
            device,
            samples=reference_samples,
            seed=9101,
            smooth_sigma=smooth_sigma,
            batched_loss_fn=batched_loss_fn,
            score_backend=score_backend,
        )
        reference_gradient = reference.gradients[0].numpy().astype(np.float64)
        reference_norm = gradient_norm(reference_gradient)
        references[problem] = {
            "samples": reference_samples,
            "gradient": [float(v) for v in reference_gradient.reshape(-1)],
            "grad_norm": reference_norm,
            "value": float(reference.value),
        }

        for samples in samples_values:
            errors = []
            cosines = []
            grad_norms = []
            standard_errors = []
            elapsed = []
            for seed_index in range(seeds):
                estimate = estimate_gradient(
                    "smooth_score",
                    loss_fn,
                    x,
                    device,
                    samples=samples,
                    seed=9200 + 101 * seed_index + samples,
                    smooth_sigma=smooth_sigma,
                    batched_loss_fn=batched_loss_fn,
                    score_backend=score_backend,
                )
                gradient = estimate.gradients[0].numpy().astype(np.float64)
                variance = estimate.gradient_variance[0].numpy().astype(np.float64)
                variance_of_mean = variance / float(samples)
                gradient_flat = gradient.reshape(-1)
                reference_flat = reference_gradient.reshape(-1)
                denom = float(np.linalg.norm(gradient_flat) * np.linalg.norm(reference_flat))
                cosine = float(np.dot(gradient_flat, reference_flat) / denom) if denom > 0.0 else 0.0
                errors.append(gradient_norm(gradient - reference_gradient))
                cosines.append(cosine)
                grad_norms.append(gradient_norm(gradient))
                standard_errors.append(float(np.sqrt(max(0.0, np.sum(variance_of_mean)))))
                elapsed.append(float(estimate.elapsed_time * 1000.0))

            rows.append(
                {
                    "problem": problem,
                    "samples": samples,
                    "seeds": seeds,
                    "reference_samples": reference_samples,
                    "mean_error_norm": float(np.mean(errors)),
                    "std_error_norm": float(np.std(errors)),
                    "mean_cosine_to_reference": float(np.mean(cosines)),
                    "min_cosine_to_reference": float(np.min(cosines)),
                    "mean_gradient_norm": float(np.mean(grad_norms)),
                    "std_gradient_norm": float(np.std(grad_norms)),
                    "mean_standard_error_norm": float(np.mean(standard_errors)),
                    "mean_elapsed_ms": float(np.mean(elapsed)),
                    "std_elapsed_ms": float(np.std(elapsed)),
                }
            )

    return {"references": references, "rows": rows}


def optimize_velocity_score_samples(
    loss_fn,
    batched_loss_fn,
    scenario: CollisionScenario,
    device: str,
    samples: int,
    train_iters: int,
    score_backend: str,
    seed_offset: int,
):
    x = np.array(scenario.initial_velocity, dtype=np.float64)
    state = {"t": 0, "m": np.zeros_like(x), "v": np.zeros_like(x)}
    best_loss = float("inf")
    elapsed = []

    for iteration in range(train_iters):
        estimate = estimate_gradient(
            "smooth_score",
            loss_fn,
            x,
            device,
            samples=samples,
            seed=12000 + seed_offset * 1000 + iteration,
            smooth_sigma=0.02,
            batched_loss_fn=batched_loss_fn,
            score_backend=score_backend,
        )
        gradient = estimate.gradients[0].numpy().astype(np.float64)
        gradient = clip_gradient(gradient, 8.0)
        best_loss = min(best_loss, evaluate_loss(loss_fn, x, device))
        elapsed.append(float(estimate.elapsed_time * 1000.0))
        x = adam_step(x, gradient, state, 0.070)
        x = np.clip(x, -2.6, 2.6)

    final_loss = evaluate_loss(loss_fn, x, device)
    best_loss = min(best_loss, final_loss)
    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "final_velocity": [float(x[0]), float(x[1])],
        "mean_estimator_ms": float(np.mean(elapsed)),
    }


def score_sample_optimization_sweep(
    device: str,
    samples_values: list[int],
    train_iters: int,
    repeats: int,
    score_backend: str,
):
    rows = []
    trials = []
    for scenario in SCENARIOS:
        loss_fn = make_collision_loss(device, scenario)
        batched_loss_fn = make_collision_loss_batched(device, scenario)
        for samples in samples_values:
            final_losses = []
            best_losses = []
            estimator_times = []
            for repeat in range(repeats):
                result = optimize_velocity_score_samples(
                    loss_fn,
                    batched_loss_fn,
                    scenario,
                    device,
                    samples=samples,
                    train_iters=train_iters,
                    score_backend=score_backend,
                    seed_offset=repeat + 17 * samples,
                )
                final_losses.append(float(result["final_loss"]))
                best_losses.append(float(result["best_loss"]))
                estimator_times.append(float(result["mean_estimator_ms"]))
                trials.append(
                    {
                        "scenario": scenario.name,
                        "samples": samples,
                        "repeat": repeat,
                        **result,
                    }
                )

            rows.append(
                {
                    "scenario": scenario.name,
                    "samples": samples,
                    "train_iters": train_iters,
                    "repeats": repeats,
                    "mean_final_loss": float(np.mean(final_losses)),
                    "std_final_loss": float(np.std(final_losses)),
                    "mean_best_loss": float(np.mean(best_losses)),
                    "std_best_loss": float(np.std(best_losses)),
                    "mean_estimator_ms": float(np.mean(estimator_times)),
                }
            )

    return {"rows": rows, "trials": trials}


def compute_collision_landscapes(device: str, grid_size: int):
    landscapes = {}
    vx = np.linspace(-0.35, 2.6, grid_size)
    vy = np.linspace(-1.45, 1.45, grid_size)
    xx, yy = np.meshgrid(vx, vy)
    velocities = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    for scenario in SCENARIOS:
        batched_loss_fn = make_collision_loss_batched(device, scenario)
        losses = evaluate_losses_batched(batched_loss_fn, velocities, device).reshape(grid_size, grid_size)
        landscapes[scenario.name] = {
            "vx": [float(v) for v in vx],
            "vy": [float(v) for v in vy],
            "losses": losses.astype(float).tolist(),
        }

    return {"grid_size": grid_size, "scenarios": landscapes}


def benchmark_methods(device: str, samples: int, repeats: int, score_backend: str):
    rows = []
    process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    problems = [
        (
            "triangle_intersection",
            make_triangle_loss(device, "intersection"),
            make_triangle_loss_batched(device, "intersection"),
            np.array([0.0, -0.37], dtype=np.float64),
        )
    ]
    problems.extend(
        (
            scenario.name,
            make_collision_loss(device, scenario),
            make_collision_loss_batched(device, scenario),
            np.array(scenario.initial_velocity, dtype=np.float64),
        )
        for scenario in SCENARIOS
    )

    method_specs = (
        ("autodiff", "autodiff", None),
        ("finite_difference", "finite_difference", None),
        ("smooth_score_batched", "smooth_score", "batched"),
        ("smooth_score_scalar", "smooth_score", "scalar"),
        ("smooth_fd", "smooth_fd", None),
    )

    for problem, loss_fn, batched_loss_fn, x in problems:
        for method, estimator_method, backend_override in method_specs:
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
                    estimator_method,
                    loss_fn,
                    x,
                    device,
                    samples=samples,
                    seed=700 + repeat,
                    smooth_sigma=smooth_sigma,
                    batched_loss_fn=batched_loss_fn,
                    score_backend=backend_override or score_backend,
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
                    "samples": samples if method in ("smooth_score_batched", "smooth_score_scalar", "smooth_fd") else 1,
                    "score_backend": backend_override if method.startswith("smooth_score") else None,
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
            [row["standard_error_norm"] for row in rows],
            marker="o",
            label=problem,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("score samples")
    ax.set_ylabel("estimated gradient standard-error norm")
    ax.grid(True, color="#e5e7eb", which="both")
    ax.legend()
    path = output_dir / "gradient_variance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    landscapes = results["collision_landscapes"]["scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        landscape = landscapes[scenario.name]
        vx = np.array(landscape["vx"])
        vy = np.array(landscape["vy"])
        losses = np.array(landscape["losses"])
        clipped = np.log10(np.clip(losses, 1.0e-4, np.percentile(losses, 95)))
        contour = ax.contourf(vx, vy, clipped, levels=28, cmap="viridis")
        for method, history in results["collision_optimization"][scenario.name]["histories"].items():
            path_xy = np.array([row["velocity"] for row in history], dtype=np.float64)
            ax.plot(path_xy[:, 0], path_xy[:, 1], color=colors[method], linewidth=1.8, label=method)
            ax.scatter(path_xy[-1, 0], path_xy[-1, 1], color=colors[method], s=28)
        ax.scatter(
            [scenario.initial_velocity[0]],
            [scenario.initial_velocity[1]],
            marker="x",
            color="white",
            s=70,
            linewidth=2,
            label="initial",
        )
        ax.set_title(scenario.name)
        ax.set_xlabel("initial vx")
        ax.set_ylabel("initial vy")
        ax.grid(True, color="white", alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.colorbar(contour, ax=axes, shrink=0.86, label="log10 crisp loss")
    path = output_dir / "collision_landscape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    sweep_rows = results["score_sample_gradient_sweep"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for problem in sorted({row["problem"] for row in sweep_rows}):
        rows = [row for row in sweep_rows if row["problem"] == problem]
        samples_x = [row["samples"] for row in rows]
        axes[0].errorbar(
            samples_x,
            [row["mean_error_norm"] for row in rows],
            yerr=[row["std_error_norm"] for row in rows],
            marker="o",
            capsize=3,
            label=problem,
        )
        axes[1].plot(
            samples_x,
            [row["mean_cosine_to_reference"] for row in rows],
            marker="o",
            label=problem,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("score samples")
    axes[0].set_ylabel("gradient error norm vs high-sample reference")
    axes[0].grid(True, color="#e5e7eb", which="both")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("score samples")
    axes[1].set_ylabel("cosine to high-sample reference")
    axes[1].grid(True, color="#e5e7eb", which="both")
    axes[1].legend(fontsize=8)
    path = output_dir / "score_sample_gradient_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    opt_rows = results["score_sample_optimization_sweep"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for scenario in sorted({row["scenario"] for row in opt_rows}):
        rows = [row for row in opt_rows if row["scenario"] == scenario]
        axes[0].errorbar(
            [row["samples"] for row in rows],
            [row["mean_final_loss"] for row in rows],
            yerr=[row["std_final_loss"] for row in rows],
            marker="o",
            capsize=3,
            label=scenario,
        )
        axes[1].errorbar(
            [row["samples"] for row in rows],
            [row["mean_best_loss"] for row in rows],
            yerr=[row["std_best_loss"] for row in rows],
            marker="o",
            capsize=3,
            label=scenario,
        )
    for ax, ylabel in zip(axes, ("final crisp loss", "best crisp loss seen"), strict=True):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("score samples per optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e5e7eb", which="both")
    axes[1].legend()
    path = output_dir / "score_sample_optimization.png"
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
        sample_sweep_values = [8, 16, 32]
        sample_sweep_seeds = 3
        sample_reference_samples = max(128, samples * 4)
        sample_optimization_values = [8, 16, 32]
        sample_optimization_repeats = 1
        sample_optimization_iters = min(train_iters, 12)
        landscape_grid = min(args.landscape_grid, 21)
    else:
        triangle_grid = args.triangle_grid
        train_iters = args.train_iters
        samples = args.samples
        benchmark_repeats = args.benchmark_repeats
        variance_samples = args.variance_samples
        sample_sweep_values = args.sample_sweep_values
        sample_sweep_seeds = args.sample_sweep_seeds
        sample_reference_samples = args.sample_reference_samples
        sample_optimization_values = args.sample_optimization_values
        sample_optimization_repeats = args.sample_optimization_repeats
        sample_optimization_iters = args.sample_optimization_iters
        landscape_grid = args.landscape_grid

    results: dict[str, Any] = {
        "metadata": {
            "device": device,
            "warp_version": wp.__version__,
            "cuda_enabled": any(wp.get_device(warp_device).is_cuda for warp_device in wp.get_devices()),
            "cuda_available": wp.is_cuda_available(),
            "cuda_toolkit": wp.get_cuda_toolkit_version(),
            "cuda_driver": wp.get_cuda_driver_version(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "samples": samples,
            "train_iters": train_iters,
            "triangle_grid": triangle_grid,
            "score_backend": args.score_backend,
            "variance_samples": variance_samples,
            "sample_sweep_values": sample_sweep_values,
            "sample_sweep_seeds": sample_sweep_seeds,
            "sample_reference_samples": sample_reference_samples,
            "sample_optimization_values": sample_optimization_values,
            "sample_optimization_iters": sample_optimization_iters,
            "sample_optimization_repeats": sample_optimization_repeats,
            "landscape_grid": landscape_grid,
            "note": "Score smoothing uses the batched estimator when score_backend='batched'.",
        },
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }

    print("Computing triangle probe...")
    results["triangle_probe"] = compute_triangle_probe(device, samples=samples, score_backend=args.score_backend)

    print("Computing triangle field...")
    results["triangle_field"] = compute_triangle_field(
        device,
        samples=samples,
        grid_size=triangle_grid,
        score_backend=args.score_backend,
    )

    print("Optimizing collision examples...")
    collision_optimization = {}
    for scenario in SCENARIOS:
        loss_fn = make_collision_loss(device, scenario)
        batched_loss_fn = make_collision_loss_batched(device, scenario)
        histories, final_velocities = optimize_velocity(
            loss_fn,
            batched_loss_fn,
            scenario,
            device,
            samples=samples,
            train_iters=train_iters,
            score_backend=args.score_backend,
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

    print("Computing collision landscapes...")
    results["collision_landscapes"] = compute_collision_landscapes(device, grid_size=landscape_grid)

    print("Computing gradient variance sweep...")
    results["variance_sweep"] = estimator_variance_sweep(
        device,
        variance_samples,
        score_backend=args.score_backend,
    )

    print("Computing score sample gradient sweep...")
    results["score_sample_gradient_sweep"] = score_sample_gradient_sweep(
        device,
        sample_sweep_values,
        seeds=sample_sweep_seeds,
        reference_samples=sample_reference_samples,
        score_backend=args.score_backend,
    )

    print("Computing score sample optimization sweep...")
    results["score_sample_optimization_sweep"] = score_sample_optimization_sweep(
        device,
        sample_optimization_values,
        train_iters=sample_optimization_iters,
        repeats=sample_optimization_repeats,
        score_backend=args.score_backend,
    )

    print("Benchmarking estimators...")
    results["benchmarks"] = benchmark_methods(
        device,
        samples=samples,
        repeats=benchmark_repeats,
        score_backend=args.score_backend,
    )

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
    parser.add_argument("--train-iters", type=int, default=150, help="Velocity optimization iterations per method.")
    parser.add_argument("--triangle-grid", type=int, default=11, help="Triangle vector-field grid resolution.")
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Number of benchmark repeats per method.")
    parser.add_argument(
        "--score-backend",
        choices=("batched", "scalar"),
        default="batched",
        help="Score-function sampling backend. Batched launches one loss kernel over all samples.",
    )
    parser.add_argument(
        "--variance-samples",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="Sample counts for the score estimator standard-error plot.",
    )
    parser.add_argument(
        "--sample-sweep-values",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
        help="Sample counts for gradient accuracy against a high-sample reference.",
    )
    parser.add_argument(
        "--sample-sweep-seeds",
        type=int,
        default=8,
        help="Seeds per sample count in the score gradient sample sweep.",
    )
    parser.add_argument(
        "--sample-reference-samples",
        type=int,
        default=8192,
        help="High-sample score estimate used as the reference in the gradient sample sweep.",
    )
    parser.add_argument(
        "--sample-optimization-values",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Sample counts for smooth-score collision optimization sweeps.",
    )
    parser.add_argument(
        "--sample-optimization-iters",
        type=int,
        default=100,
        help="Iterations per score-only optimization sample-count sweep trial.",
    )
    parser.add_argument(
        "--sample-optimization-repeats",
        type=int,
        default=3,
        help="Repeats per sample count in score-only optimization sweeps.",
    )
    parser.add_argument(
        "--landscape-grid", type=int, default=71, help="Velocity-grid resolution for collision landscapes."
    )
    parser.add_argument("--quick", action="store_true", help="Run a reduced workload for smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Accepted for consistency with other examples.")
    run(parser.parse_args())
