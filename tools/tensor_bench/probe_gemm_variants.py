import numpy as np, warp as wp, torch
from common import time_warp_graph, time_torch_graph
wp.config.quiet = True
B, D = 8192, 1024
d="cuda:0"
r=lambda *s: wp.array(np.random.randn(*s).astype(np.float32), device=d)
X, W, dZ = r(B,D), r(D,D), r(B,D)
O, Z, gW, bias, gb = wp.zeros((B,D),dtype=float,device=d), wp.zeros((B,D),dtype=float,device=d), wp.zeros((D,D),dtype=float,device=d), r(D), wp.zeros(D,dtype=float,device=d)
cases = {
 "fwd plain X@W":        lambda: wp.gemm(X, W, O),
 "fwd bias+gelu_aux":    lambda: wp.gemm(X, W, O, bias=bias, activation="gelu", aux=Z),
 "fwd bias":             lambda: wp.gemm(X, W, O, bias=bias),
 "dW  X^T@dZ":           lambda: wp.gemm(X, dZ, gW, trans_a=True),
 "dX  dZ@W^T":           lambda: wp.gemm(dZ, W, O, trans_b=True),
 "dX dgelu_bgrad":       lambda: wp.gemm(dZ, W, O, trans_b=True, activation="gelu", grad_aux=Z, bias_grad=gb),
}
tX,tW,tdZ = (torch.randn(*s,device="cuda") for s in [(B,D),(D,D),(B,D)])
tcases = {"fwd plain X@W": lambda: tX@tW, "dW  X^T@dZ": lambda: tX.t()@tdZ, "dX  dZ@W^T": lambda: tdZ@tW.t()}
for k,f in cases.items():
    t = time_warp_graph(f, iters=50, reps=3)
    ref = time_torch_graph(tcases[k], iters=50, reps=3) if k in tcases else float("nan")
    print(f"{k:22s} warp {t:8.1f}  torch {ref:8.1f}")
