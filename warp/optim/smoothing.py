# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program smoothing estimators for branchy Warp objectives.

This module provides Gaussian-smoothed gradient estimators that can be used
with ordinary Warp programs. Import it explicitly through :mod:`warp.optim`::

    import warp.optim

    result = warp.optim.smoothing.estimate_score_function(loss_fn, params, sigma=0.1)
"""

# isort: skip_file

from warp._src.optim.smoothing import GradientEstimate as GradientEstimate
from warp._src.optim.smoothing import estimate_finite_difference as estimate_finite_difference
from warp._src.optim.smoothing import estimate_pathwise as estimate_pathwise
from warp._src.optim.smoothing import estimate_score_function as estimate_score_function
