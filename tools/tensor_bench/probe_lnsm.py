import numpy as np, torch, warp as wp
from common import time_warp_graph
wp.config.quiet = True
dev = "cuda:0"
ROWS, COLS = 16384, 1024

@wp.func
def rs(v: float):
    return 1.0 / wp.sqrt(v + 1.0e-5)

def make(R, BD):
    @wp.kernel(module="unique")
    def k(x: wp.array2d[float], g: wp.array2d[float], b: wp.array2d[float], y: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(x, shape=(R, COLS), offset=(i * R, 0))
        mean = wp.tile_sum(t, axis=1) * (1.0 / float(COLS))
        d = t - wp.tile_broadcast(wp.tile_reshape(mean, shape=(R, 1)), shape=(R, COLS))
        var = wp.tile_sum(wp.tile_map(wp.mul, d, d), axis=1) * (1.0 / float(COLS))
        inv = wp.tile_map(rs, var)
        n = d * wp.tile_broadcast(wp.tile_reshape(inv, shape=(R, 1)), shape=(R, COLS))
        n = n * wp.tile_broadcast(wp.tile_load(g, shape=(1, COLS)), shape=(R, COLS)) + wp.tile_broadcast(wp.tile_load(b, shape=(1, COLS)), shape=(R, COLS))
        m = wp.tile_reduce(wp.max, n, axis=1)
        e = wp.tile_map(wp.exp, n - wp.tile_broadcast(wp.tile_reshape(m, shape=(R, 1)), shape=(R, COLS)))
        s = wp.tile_sum(e, axis=1)
        wp.tile_store(y, e / wp.tile_broadcast(wp.tile_reshape(s, shape=(R, 1)), shape=(R, COLS)), offset=(i * R, 0))
    return k

x = np.random.randn(ROWS, COLS).astype(np.float32)
wx = wp.array(x, device=dev); wg = wp.ones((1, COLS), dtype=float, device=dev); wb = wp.zeros((1, COLS), dtype=float, device=dev)
wy = wp.zeros_like(wx)
ref = torch.softmax(torch.nn.functional.layer_norm(torch.from_numpy(x).cuda(), (COLS,)), -1).cpu().numpy()
for R, BD in [(1, 32), (1, 64), (2, 64), (2, 128), (1, 128)]:
    k = make(R, BD)
    L = lambda: wp.launch_tiled(k, dim=[ROWS // R], inputs=[wx, wg, wb, wy], block_dim=BD, device=dev)
    L()
    print("R", R, "BD", BD, "err", np.abs(wy.numpy() - ref).max(), "us", round(time_warp_graph(L), 1))
