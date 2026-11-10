import sys, numpy as np, torch, warp as wp
from common import time_warp_graph, time_torch_graph
wp.config.quiet = True
dev = "cuda:0"

def make_fwd(sizes, TM, act):
    L = len(sizes) - 1
    @wp.func
    def activ(x: float):
        if wp.static(act == "elu"):
            return wp.where(x > 0.0, x, wp.exp(x) - 1.0)
        elif wp.static(act == "softsign"):
            return x / (1.0 + wp.abs(x))
        else:
            return wp.max(x, 0.0)
    S0, S1, S2, S3 = sizes
    @wp.kernel(module="unique")
    def k(x: wp.array2d[float], w0: wp.array2d[float], b0: wp.array2d[float],
          w1: wp.array2d[float], b1: wp.array2d[float],
          w2: wp.array2d[float], b2: wp.array2d[float], y: wp.array2d[float]):
        i = wp.tid()
        h = wp.tile_load(x, shape=(TM, S0), offset=(i * TM, 0))
        h = wp.tile_map(activ, wp.tile_matmul(h, wp.tile_load(w0, shape=(S0, S1))) + wp.tile_broadcast(wp.tile_load(b0, shape=(1, S1)), shape=(TM, S1)))
        h = wp.tile_map(activ, wp.tile_matmul(h, wp.tile_load(w1, shape=(S1, S2))) + wp.tile_broadcast(wp.tile_load(b1, shape=(1, S2)), shape=(TM, S2)))
        o = wp.tile_matmul(h, wp.tile_load(w2, shape=(S2, S3))) + wp.tile_broadcast(wp.tile_load(b2, shape=(1, S3)), shape=(TM, S3))
        wp.tile_store(y, o, offset=(i * TM, 0))
    return k

if __name__ == "__main__":
    sizes = [6, 32, 32, 1]; B = 49152; act = "softsign"
    ps = []
    g = np.random.default_rng(0)
    x = g.standard_normal((B, sizes[0])).astype(np.float32)
    W = [(g.standard_normal((i, o)) / np.sqrt(i)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
    Bv = [np.zeros((1, o), np.float32) for o in sizes[1:]]
    wx = wp.array(x, device=dev); wW = [wp.array(w, device=dev) for w in W]; wB = [wp.array(b, device=dev) for b in Bv]
    wy = wp.zeros((B, sizes[-1]), dtype=float, device=dev)
    for TM in (32, 64, 128):
        k = make_fwd(sizes, TM, act)
        args = [wx, wW[0], wB[0], wW[1], wB[1], wW[2], wB[2], wy]
        wp.launch_tiled(k, dim=[B // TM], inputs=args, block_dim=128, device=dev)
        # ref
        h = x
        for l in range(3):
            h = h @ W[l] + Bv[l]
            if l < 2: h = h / (1 + np.abs(h))
        print("TM", TM, "err", np.abs(wy.numpy() - h).max(),
              "us", round(time_warp_graph(lambda: wp.launch_tiled(k, dim=[B // TM], inputs=args, block_dim=128, device=dev)), 1))
