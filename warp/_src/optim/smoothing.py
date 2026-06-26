# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp

_wp_module_name_ = "warp.optim.smoothing"


@dataclass(frozen=True)
class GradientEstimate:
    """Result returned by a smoothed gradient estimator.

    Args:
        method: Name of the estimator used to compute this result.
        value: Estimated scalar objective value.
        gradients: Gradient arrays matching the optimized parameters.
        gradient_variance: Per-parameter sample variance of the gradient estimator.
        samples: Number of perturbed program executions used by the estimate.
        sigma: Per-parameter Gaussian smoothing standard deviations.
        elapsed_time: Wall-clock time spent in the estimator, in seconds.
        values: Per-sample scalar objective values used by the estimate.
    """

    method: str
    value: float
    gradients: list[wp.array]
    gradient_variance: list[wp.array]
    samples: int
    sigma: tuple[float, ...]
    elapsed_time: float
    values: np.ndarray


def estimate_score_function(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int = 64,
    seed: int | None = None,
    antithetic: bool = True,
) -> GradientEstimate:
    """Estimate the gradient of a Gaussian-smoothed program with a score-function estimator.

    This estimator uses only scalar objective values from perturbed program executions,
    so it can carry gradient signal across data-dependent branches. For a program
    ``f`` and Gaussian perturbation ``eps ~ N(0, sigma^2 I)``, it estimates
    ``grad_x E[f(x + eps)]`` with ``E[(f(x + eps) - baseline) eps / sigma^2]``.

    Args:
        loss_fn: Callable that receives sampled parameter arrays and returns a scalar
            Warp array or Python number.
        params: Warp arrays to perturb and differentiate with respect to.
        sigma: Gaussian smoothing standard deviation. A scalar is broadcast to all
            parameter arrays; a sequence must match ``params``.
        samples: Number of perturbed executions.
        seed: Seed for NumPy's random number generator.
        antithetic: Whether to use antithetic normal samples.

    Returns:
        A :class:`GradientEstimate` containing Warp gradient arrays and variance estimates.
    """

    return _estimate_score_function(loss_fn, params, sigma=sigma, samples=samples, seed=seed, antithetic=antithetic)


def estimate_score_function_batched(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int = 64,
    seed: int | None = None,
    antithetic: bool = True,
) -> GradientEstimate:
    """Estimate a score-function gradient from one batched loss evaluation.

    The callable receives parameter arrays with a leading sample dimension and
    must return one scalar loss per sample. For example, if a parameter has
    shape ``(d,)``, the batched callable receives an array of shape
    ``(samples, d)`` and returns a Warp array of shape ``(samples,)``.

    This has the same estimator as :func:`estimate_score_function`, but it lets
    Warp kernels parallelize the perturbed executions across CPU threads or CUDA
    threads instead of looping over samples in Python.
    """

    return _estimate_score_function_batched(
        loss_fn,
        params,
        sigma=sigma,
        samples=samples,
        seed=seed,
        antithetic=antithetic,
    )


def estimate_pathwise(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int = 64,
    seed: int | None = None,
    antithetic: bool = True,
) -> GradientEstimate:
    """Estimate a Gaussian-smoothed gradient by averaging Warp Tape gradients.

    This is the standard pathwise estimator: it differentiates each perturbed
    execution with Warp's AD and averages the resulting gradients. It is efficient
    for smooth programs but may miss jump terms introduced by input-dependent
    branches.

    Args:
        loss_fn: Callable that receives sampled parameter arrays and returns a scalar
            Warp array or Python number.
        params: Warp arrays to perturb and differentiate with respect to.
        sigma: Gaussian smoothing standard deviation. A scalar is broadcast to all
            parameter arrays; a sequence must match ``params``.
        samples: Number of perturbed executions.
        seed: Seed for NumPy's random number generator.
        antithetic: Whether to use antithetic normal samples.

    Returns:
        A :class:`GradientEstimate` containing Warp gradient arrays and variance estimates.
    """

    return _estimate_pathwise(loss_fn, params, sigma=sigma, samples=samples, seed=seed, antithetic=antithetic)


def estimate_finite_difference(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    epsilon: float = 1.0e-3,
    samples: int = 64,
    seed: int | None = None,
    antithetic: bool = True,
) -> GradientEstimate:
    """Estimate a smoothed gradient with central finite differences.

    The same Gaussian perturbations are reused for the positive and negative
    finite-difference evaluations of each parameter component, which reduces
    Monte Carlo noise when differentiating the smoothed objective.

    Args:
        loss_fn: Callable that receives sampled parameter arrays and returns a scalar
            Warp array or Python number.
        params: Warp arrays to perturb and differentiate with respect to.
        sigma: Gaussian smoothing standard deviation. A scalar is broadcast to all
            parameter arrays; a sequence must match ``params``.
        epsilon: Central finite-difference step.
        samples: Number of smoothing samples per objective evaluation.
        seed: Seed for NumPy's random number generator.
        antithetic: Whether to use antithetic normal samples.

    Returns:
        A :class:`GradientEstimate` containing Warp gradient arrays. The variance
        fields are zero because central finite differences produce one aggregate
        gradient estimate per parameter component.
    """

    return _estimate_finite_difference(
        loss_fn,
        params,
        sigma=sigma,
        epsilon=epsilon,
        samples=samples,
        seed=seed,
        antithetic=antithetic,
    )


def _estimate_score_function(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int,
    seed: int | None,
    antithetic: bool,
) -> GradientEstimate:
    params = _validate_params(params)
    sigmas = _normalize_sigma(sigma, params)
    base_values = _params_to_numpy(params)
    noises = _make_noises(samples, base_values, seed=seed, antithetic=antithetic)

    start = time.perf_counter()
    values = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_params = _make_sample_params(params, base_values, sigmas, noises, sample_index, requires_grad=False)
        values[sample_index] = _loss_to_float(loss_fn(sampled_params))

    baseline = float(np.mean(values))
    gradients = []
    variances = []
    for param_index, param in enumerate(params):
        sigma_i = sigmas[param_index]
        if sigma_i == 0.0:
            grad_samples = np.zeros((samples, *base_values[param_index].shape), dtype=np.float64)
        else:
            scale = ((values - baseline) / sigma_i).reshape((samples,) + (1,) * base_values[param_index].ndim)
            grad_samples = scale * noises[param_index]

        gradients.append(_array_like(param, np.mean(grad_samples, axis=0)))
        variances.append(_array_like(param, _sample_variance(grad_samples)))

    elapsed_time = time.perf_counter() - start
    return GradientEstimate(
        method="score_function",
        value=baseline,
        gradients=gradients,
        gradient_variance=variances,
        samples=samples,
        sigma=sigmas,
        elapsed_time=elapsed_time,
        values=values,
    )


def _estimate_score_function_batched(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int,
    seed: int | None,
    antithetic: bool,
) -> GradientEstimate:
    params = _validate_params(params)
    sigmas = _normalize_sigma(sigma, params)
    base_values = _params_to_numpy(params)
    noises = _make_noises(samples, base_values, seed=seed, antithetic=antithetic)

    start = time.perf_counter()
    sampled_params = _make_batched_sample_params(params, base_values, sigmas, noises)
    values = _losses_to_numpy(loss_fn(sampled_params), samples)

    baseline = float(np.mean(values))
    gradients = []
    variances = []
    for param_index, param in enumerate(params):
        sigma_i = sigmas[param_index]
        if sigma_i == 0.0:
            grad_samples = np.zeros((samples, *base_values[param_index].shape), dtype=np.float64)
        else:
            scale = ((values - baseline) / sigma_i).reshape((samples,) + (1,) * base_values[param_index].ndim)
            grad_samples = scale * noises[param_index]

        gradients.append(_array_like(param, np.mean(grad_samples, axis=0)))
        variances.append(_array_like(param, _sample_variance(grad_samples)))

    elapsed_time = time.perf_counter() - start
    return GradientEstimate(
        method="score_function_batched",
        value=baseline,
        gradients=gradients,
        gradient_variance=variances,
        samples=samples,
        sigma=sigmas,
        elapsed_time=elapsed_time,
        values=values,
    )


def _estimate_pathwise(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    samples: int,
    seed: int | None,
    antithetic: bool,
) -> GradientEstimate:
    params = _validate_params(params)
    sigmas = _normalize_sigma(sigma, params)
    base_values = _params_to_numpy(params)
    noises = _make_noises(samples, base_values, seed=seed, antithetic=antithetic)

    start = time.perf_counter()
    values = np.empty(samples, dtype=np.float64)
    gradient_samples = [np.zeros((samples, *base_value.shape), dtype=np.float64) for base_value in base_values]

    for sample_index in range(samples):
        sampled_params = _make_sample_params(params, base_values, sigmas, noises, sample_index, requires_grad=True)
        with wp.Tape() as tape:
            loss = loss_fn(sampled_params)

        values[sample_index] = _loss_to_float(loss)
        tape.backward(loss)

        for param_index, sampled_param in enumerate(sampled_params):
            grad = tape.gradients.get(sampled_param)
            if grad is not None:
                gradient_samples[param_index][sample_index] = grad.numpy()

    gradients = []
    variances = []
    for param_index, param in enumerate(params):
        gradients.append(_array_like(param, np.mean(gradient_samples[param_index], axis=0)))
        variances.append(_array_like(param, _sample_variance(gradient_samples[param_index])))

    elapsed_time = time.perf_counter() - start
    return GradientEstimate(
        method="pathwise",
        value=float(np.mean(values)),
        gradients=gradients,
        gradient_variance=variances,
        samples=samples,
        sigma=sigmas,
        elapsed_time=elapsed_time,
        values=values,
    )


def _estimate_finite_difference(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    params: Sequence[wp.array],
    *,
    sigma: float | Sequence[float],
    epsilon: float,
    samples: int,
    seed: int | None,
    antithetic: bool,
) -> GradientEstimate:
    params = _validate_params(params)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    sigmas = _normalize_sigma(sigma, params)
    base_values = _params_to_numpy(params)
    noises = _make_noises(samples, base_values, seed=seed, antithetic=antithetic)

    start = time.perf_counter()
    values = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_params = _make_sample_params(params, base_values, sigmas, noises, sample_index, requires_grad=False)
        values[sample_index] = _loss_to_float(loss_fn(sampled_params))

    gradients_np: list[np.ndarray] = []
    for param_index, base_value in enumerate(base_values):
        grad = np.zeros_like(base_value, dtype=np.float64)
        for flat_index in range(base_value.size):
            index = np.unravel_index(flat_index, base_value.shape)
            plus_values = [value.copy() for value in base_values]
            minus_values = [value.copy() for value in base_values]
            plus_values[param_index][index] += epsilon
            minus_values[param_index][index] -= epsilon

            plus = _evaluate_smoothed_value(loss_fn, params, plus_values, sigmas, noises, samples)
            minus = _evaluate_smoothed_value(loss_fn, params, minus_values, sigmas, noises, samples)
            grad[index] = (plus - minus) / (2.0 * epsilon)

        gradients_np.append(grad)

    elapsed_time = time.perf_counter() - start
    gradients = [_array_like(param, grad) for param, grad in zip(params, gradients_np, strict=True)]
    variances = [_array_like(param, np.zeros_like(grad)) for param, grad in zip(params, gradients_np, strict=True)]

    return GradientEstimate(
        method="finite_difference",
        value=float(np.mean(values)),
        gradients=gradients,
        gradient_variance=variances,
        samples=samples,
        sigma=sigmas,
        elapsed_time=elapsed_time,
        values=values,
    )


def _evaluate_smoothed_value(
    loss_fn: Callable[[Sequence[wp.array]], Any],
    template_params: Sequence[wp.array],
    base_values: Sequence[np.ndarray],
    sigmas: Sequence[float],
    noises: Sequence[np.ndarray],
    samples: int,
) -> float:
    values = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_params = _make_sample_params(
            template_params,
            base_values,
            sigmas,
            noises,
            sample_index,
            requires_grad=False,
        )
        values[sample_index] = _loss_to_float(loss_fn(sampled_params))
    return float(np.mean(values))


def _validate_params(params: Sequence[wp.array]) -> tuple[wp.array, ...]:
    if len(params) == 0:
        raise ValueError("params must contain at least one Warp array")

    result = tuple(params)
    for param in result:
        if not isinstance(param, wp.array):
            raise TypeError("params must contain Warp arrays")
        if param.dtype not in (wp.float16, wp.float32, wp.float64):
            raise TypeError("program smoothing estimators currently support scalar floating-point arrays")

    return result


def _normalize_sigma(sigma: float | Sequence[float], params: Sequence[wp.array]) -> tuple[float, ...]:
    if isinstance(sigma, int | float):
        sigmas = (float(sigma),) * len(params)
    else:
        sigmas = tuple(float(value) for value in sigma)
        if len(sigmas) != len(params):
            raise ValueError("sigma must be a scalar or a sequence matching params")

    if any(value < 0.0 for value in sigmas):
        raise ValueError("sigma values must be non-negative")
    return sigmas


def _params_to_numpy(params: Sequence[wp.array]) -> list[np.ndarray]:
    return [param.numpy().astype(np.float64, copy=True) for param in params]


def _make_noises(
    samples: int,
    base_values: Sequence[np.ndarray],
    *,
    seed: int | None,
    antithetic: bool,
) -> list[np.ndarray]:
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    if antithetic and samples > 1:
        half = samples // 2
        noises = [rng.standard_normal((half, *value.shape)) for value in base_values]
        noises = [np.concatenate([noise, -noise], axis=0) for noise in noises]
        if samples % 2 == 1:
            extra = [rng.standard_normal((1, *value.shape)) for value in base_values]
            noises = [np.concatenate([noise, extra_i], axis=0) for noise, extra_i in zip(noises, extra, strict=True)]
        return noises

    return [rng.standard_normal((samples, *value.shape)) for value in base_values]


def _make_sample_params(
    template_params: Sequence[wp.array],
    base_values: Sequence[np.ndarray],
    sigmas: Sequence[float],
    noises: Sequence[np.ndarray],
    sample_index: int,
    *,
    requires_grad: bool,
) -> list[wp.array]:
    sampled_params = []
    for param, base_value, sigma, noise in zip(template_params, base_values, sigmas, noises, strict=True):
        sample = base_value + sigma * noise[sample_index]
        sampled_params.append(wp.array(sample, dtype=param.dtype, device=param.device, requires_grad=requires_grad))
    return sampled_params


def _make_batched_sample_params(
    template_params: Sequence[wp.array],
    base_values: Sequence[np.ndarray],
    sigmas: Sequence[float],
    noises: Sequence[np.ndarray],
) -> list[wp.array]:
    sampled_params = []
    for param, base_value, sigma, noise in zip(template_params, base_values, sigmas, noises, strict=True):
        samples = base_value.reshape((1, *base_value.shape)) + sigma * noise
        sampled_params.append(wp.array(samples, dtype=param.dtype, device=param.device, requires_grad=False))
    return sampled_params


def _loss_to_float(loss: Any) -> float:
    if isinstance(loss, wp.array):
        values = loss.numpy().reshape(-1)
        if values.size != 1:
            raise ValueError("loss_fn must return a scalar Warp array or Python number")
        return float(values[0])

    return float(loss)


def _losses_to_numpy(losses: Any, samples: int) -> np.ndarray:
    if isinstance(losses, wp.array):
        values = losses.numpy().reshape(-1).astype(np.float64, copy=False)
    else:
        values = np.asarray(losses, dtype=np.float64).reshape(-1)

    if values.size != samples:
        raise ValueError("batched loss_fn must return one scalar loss per sample")

    return values


def _array_like(param: wp.array, values: np.ndarray) -> wp.array:
    return wp.array(values, dtype=param.dtype, device=param.device)


def _sample_variance(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1:], dtype=np.float64)
    return np.var(values, axis=0, ddof=1)
