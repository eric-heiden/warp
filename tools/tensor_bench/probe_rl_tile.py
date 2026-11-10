import sys, numpy as np, warp as wp
from common import time_warp_graph
wp.config.quiet = True
d = "cuda:0"; B = 4096
S0, S1, S2, S3, S4 = 64, 512, 256, 128, 24

@wp.func
def elu(x: float):
    return wp.where(x > 0.0, x, wp.exp(x) - 1.0)

def make(TM, NC, KC):
    @wp.kernel(module="unique")
    def kA(x: wp.array2d[float], w1: wp.array2d[float], b1: wp.array2d[float], w2: wp.array2d[float], b2: wp.array2d[float], h2o: wp.array2d[float]):
        i = wp.tid()
        xt = wp.tile_load(x, shape=(TM, S0), offset=(i * TM, 0))
        acc2 = wp.tile_load(b2, shape=(1, S2))
        acc2 = wp.tile_broadcast(acc2, shape=(TM, S2))
        acc2 = wp.tile_zeros(shape=(TM, S2), dtype=float) + acc2
        for c in range(S1 // NC):
            w1c = wp.tile_load(w1, shape=(S0, NC), offset=(0, c * NC))
            b1c = wp.tile_load(b1, shape=(1, NC), offset=(0, c * NC))
            h1c = wp.tile_broadcast(b1c, shape=(TM, NC))
            wp.tile_matmul(xt, w1c, h1c)
            h1c = wp.tile_map(elu, h1c)
            w2c = wp.tile_load(w2, shape=(NC, S2), offset=(c * NC, 0))
            wp.tile_matmul(h1c, w2c, acc2)
        wp.tile_store(h2o, wp.tile_map(elu, acc2), offset=(i * TM, 0))

    @wp.kernel(module="unique")
    def kB(h2: wp.array2d[float], w3: wp.array2d[float], b3: wp.array2d[float], w4: wp.array2d[float], b4: wp.array2d[float], y: wp.array2d[float]):
        i = wp.tid()
        acc3 = wp.tile_broadcast(wp.tile_load(b3, shape=(1, S3)), shape=(TM, S3))
        acc3 = wp.tile_zeros(shape=(TM, S3), dtype=float) + acc3
        for c in range(S2 // KC):
            h2c = wp.tile_load(h2, shape=(TM, KC), offset=(i * TM, c * KC))
            w3c = wp.tile_load(w3, shape=(KC, S3), offset=(c * KC, 0))
            wp.tile_matmul(h2c, w3c, acc3)
        h3 = wp.tile_map(elu, acc3)
        o = wp.tile_matmul(h3, wp.tile_load(w4, shape=(S3, S4))) + wp.tile_broadcast(wp.tile_load(b4, shape=(1, S4)), shape=(TM, S4))
        wp.tile_store(y, o, offset=(i * TM, 0))
    return kA, kB

if __name__ == "__main__":
    g = np.random.default_rng(0)
    sizes = [S0, S1, S2, S3, S4]
    Wn = [(g.standard_normal((i, o)) / np.sqrt(i)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
    Bn = [(0.1 * g.standard_normal((1, o))).astype(np.float32) for o in sizes[1:]]
    xn = g.standard_normal((B, S0)).astype(np.float32)
    h = xn
    for l in range(4):
        h = h @ Wn[l] + Bn[l]
        if l < 3: h = np.where(h > 0, h, np.exp(h) - 1)
    wx = wp.array(xn, device=d); ws = [wp.array(w, device=d) for w in Wn]; bs = [wp.array(b, device=d) for b in Bn]
    wy = wp.zeros((B, S4), dtype=float, device=d)
    h2 = wp.zeros((B, S2), dtype=float, device=d)
    for TM, NC, KC, BD in [(16, 16, 32, 128), (32, 16, 32, 128), (16, 32, 32, 128)]:
        try:
            kA, kB = make(TM, NC, KC)
            def L():
                wp.launch_tiled(kA, dim=[B // TM], inputs=[wx, ws[0], bs[0], ws[1], bs[1], h2], block_dim=BD, device=d)
                wp.launch_tiled(kB, dim=[B // TM], inputs=[h2, ws[2], bs[2], ws[3], bs[3], wy], block_dim=BD, device=d)
            L()
            print("TM", TM, "NC", NC, "KC", KC, "BD", BD, "err", float(np.abs(wy.numpy() - h).max()), "us", round(time_warp_graph(L), 1), flush=True)
        except Exception as e:
            print("TM", TM, NC, KC, BD, "FAILED", str(e)[:200])
