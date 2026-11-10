import numpy as np, torch, warp as wp
from common import time_warp_graph
wp.config.quiet = True
dev = "cuda:0"
ROWS, COLS = 16384, 1024

def make(BD):
    V = COLS // BD  # elements per thread
    @wp.kernel(module="unique")
    def k(x: wp.array2d[float], g: wp.array[float], b: wp.array[float], y: wp.array2d[float]):
        row, t = wp.tid()
        v = wp.vector(length=V, dtype=float)
        s = float(0.0)
        for j in range(V):
            e = x[row, t + j * BD]
            v[j] = e
            s += e
        mean = wp.tile_sum(wp.tile(s))[0] * (1.0 / float(COLS))
        q = float(0.0)
        for j in range(V):
            d = v[j] - mean
            q += d * d
        var = wp.tile_sum(wp.tile(q))[0] * (1.0 / float(COLS))
        inv = 1.0 / wp.sqrt(var + 1.0e-5)
        m = float(-1.0e30)
        for j in range(V):
            n = (v[j] - mean) * inv * g[t + j * BD] + b[t + j * BD]
            v[j] = n
            m = wp.max(m, n)
        mx = wp.tile_max(wp.tile(m))[0]
        z = float(0.0)
        for j in range(V):
            e = wp.exp(v[j] - mx)
            v[j] = e
            z += e
        tot = wp.tile_sum(wp.tile(z))[0]
        r = 1.0 / tot
        for j in range(V):
            y[row, t + j * BD] = v[j] * r
    return k

x = np.random.randn(ROWS, COLS).astype(np.float32)
wx = wp.array(x, device=dev); wg = wp.ones(COLS, dtype=float, device=dev); wb = wp.zeros(COLS, dtype=float, device=dev)
wy = wp.zeros_like(wx)
ref = torch.softmax(torch.nn.functional.layer_norm(torch.from_numpy(x).cuda(), (COLS,)), -1).cpu().numpy()
for BD in [32, 64, 128, 256]:
    k = make(BD)
    L = lambda: wp.launch_tiled(k, dim=[ROWS], inputs=[wx, wg, wb, wy], block_dim=BD, device=dev)
    L()
    print("BD", BD, "err", np.abs(wy.numpy() - ref).max(), "us", round(time_warp_graph(L), 1))
