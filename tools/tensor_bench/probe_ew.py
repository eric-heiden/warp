import math, numpy as np, warp as wp
from common import time_warp_graph
wp.config.quiet = True
n = 64 * 1024 * 1024
x, w, b, r = (wp.array(np.random.randn(n).astype(np.float32), device="cuda:0") for _ in range(4))
out = wp.empty_like(x)

@wp.func
def chain(x: float, w: float, b: float, r: float):
    z = x * w + b
    t = wp.tanh(0.7978845608 * (z + 0.044715 * z * z * z))
    return 0.5 * z * (1.0 + t) * 0.5 + r

@wp.kernel
def k_scalar(x: wp.array[float], w: wp.array[float], b: wp.array[float], r: wp.array[float], o: wp.array[float]):
    i = wp.tid()
    o[i] = chain(x[i], w[i], b[i], r[i])

@wp.kernel
def k_gridstride(x: wp.array[float], w: wp.array[float], b: wp.array[float], r: wp.array[float], o: wp.array[float]):
    i = wp.tid()
    s = wp.vec4(0.0)
    # vec4 via float4 view is not available here; plain loop of 4 per thread
    base = i * 4
    for j in range(4):
        o[base + j] = chain(x[base + j], w[base + j], b[base + j], r[base + j])

print("map      ", time_warp_graph(lambda: wp.map(chain, x, w, b, r, out=out)))
print("scalar   ", time_warp_graph(lambda: wp.launch(k_scalar, n, [x, w, b, r, out])))
print("x4/thread", time_warp_graph(lambda: wp.launch(k_gridstride, n // 4, [x, w, b, r, out])))
