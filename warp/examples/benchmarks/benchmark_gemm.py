import time
from typing import Any

import torch as tc
import numpy as np
import warp as wp

tc.backends.cuda.matmul.allow_tf32 = False      # Disable TF32 for matrix multiplications
tc.backends.cudnn.allow_tf32 = False            # Disable TF32 for cuDNN operations

wp.clear_kernel_cache()
wp.set_module_options({"fast_math": True,
                       "enable_backward": False})

def create_mlp_kernel(m, n, k):

    TILE_M = m
    TILE_N = n
    TILE_K = k
       
    @wp.kernel
    def mlp(x: wp.array2d(dtype=float),
            weights_wp: wp.array2d(dtype=float),
            n_k: int,
            output: wp.array2d(dtype=float)):

        i_m, i_n = wp.tid()
        sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)
        for count in range(n_k):
            feat = wp.tile_load(x, i_m, count, TILE_M, TILE_K)
            weight = wp.tile_load(weights_wp, count, i_n, TILE_K, TILE_N)
            wp.tile_matmul(feat, weight, sum)

        wp.tile_store(output, i_m, i_n, sum)

    return mlp


def benchmark_torch(A, B, iterations):
    linear = tc.nn.Linear(K, N, bias=False).cuda()
    
    # warm-up
    for i in range(10):
        tc.matmul(A, B)

    timers = {}
    with wp.ScopedTimer("torch", print=False, dict=timers, synchronize=True):
        for i in range(iterations):
            tc.matmul(A, B)

    return timers["torch"][0]

def benchmark_warp(A, B, config, iterations):
    
    TILE_M = config[0]
    TILE_N = config[1]
    TILE_K = config[2]
    BLOCK_DIM = config[3]

    mlp = create_mlp_kernel(TILE_M, TILE_N, TILE_K)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    output = wp.zeros((M, N), dtype=float)

    # warm-up
    for i in range(10):
        wp.launch_tiled(kernel=mlp, dim=[M//TILE_M, N//TILE_N], inputs=[A, B, K//TILE_K, output], block_dim=BLOCK_DIM)

    # check output
    assert (np.allclose(output.numpy(), A.numpy()@B.numpy(), atol=1e-3, rtol=1e-3))

    # benchmark
    timers = {}
    with wp.ScopedTimer("warp", print=False, dict=timers, synchronize=True):
        for i in range(iterations):
            wp.launch_tiled(kernel=mlp, dim=[M//TILE_M, N//TILE_N], inputs=[A, B, K//TILE_K, output], block_dim=BLOCK_DIM)

    return timers["warp"][0]



tile_m = [16, 32, 64, 128]
tile_n = [16, 32, 64, 64]
tile_k = [8, 16, 64]
block = [32, 64, 128]

M = 1024
N = 1024
K = 1024

A = tc.randn(M, K).cuda()
B = tc.randn(K, N).cuda()

iterations = 1000

time_torch = benchmark_torch(A, B, 1000)
print(f"Torch: {time_torch}")

from itertools import product
configs = list(product(tile_m, tile_n, tile_k, block))

wp.config.quiet = True

# header
print("{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format("TILE_M", 12,
                                    "TILE_N", 12,
                                    "TILE_K", 12,
                                    "BLOCK", 12,
                                    "Time", 12,
                                    "Relative", 12))
for c in configs:           
    time_warp = benchmark_warp(wp.from_torch(A), wp.from_torch(B), c, iterations)
    print("{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format(c[0], 12,
                                        c[1], 12,
                                        c[2], 12,
                                        c[3], 12,
                                        time_warp, 12,
                                        time_warp/time_torch, 12))
    
