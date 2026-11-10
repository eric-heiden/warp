import time, numpy as np, warp as wp, torch
wp.config.quiet = True
n = 1024
x, w, b, r = (wp.array(np.random.randn(n).astype(np.float32), device="cuda:0") for _ in range(4))
out = wp.empty_like(x)
@wp.func
def chain(x: float, w: float, b: float, r: float):
    return x * w + b + r
@wp.kernel
def k(x: wp.array[float], w: wp.array[float], b: wp.array[float], r: wp.array[float], o: wp.array[float]):
    i = wp.tid(); o[i] = chain(x[i], w[i], b[i], r[i])
def t(f, N=2000):
    f(); wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): f()
    wp.synchronize(); return (time.perf_counter() - t0) / N * 1e6
print("wp.map        us/call", t(lambda: wp.map(chain, x, w, b, r, out=out)))
print("wp.launch     us/call", t(lambda: wp.launch(k, n, [x, w, b, r, out])))
cmd = wp.launch(k, n, [x, w, b, r, out], record_cmd=True)
print("launch cmd    us/call", t(lambda: cmd.launch()))
tx, tw, tb, tr = (torch.randn(n, device="cuda") for _ in range(4))
print("torch x*w+b+r us/call", t(lambda: tx * tw + tb + tr))
