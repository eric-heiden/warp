import numpy as np, warp as wp
from mlp_gemm import GemmMLP
from common import time_warp_graph
wp.config.quiet = True
sizes=[1024,1024,1024,1024]; B=8192
g=np.random.default_rng(1)
W=[(g.standard_normal((i,o))/np.sqrt(i)).astype(np.float32) for i,o in zip(sizes[:-1],sizes[1:])]
Bv=[np.zeros(o,np.float32) for o in sizes[1:]]
wx=wp.array(g.standard_normal((B,sizes[0])).astype(np.float32),device="cuda:0"); wy=wp.array(g.standard_normal((B,sizes[-1])).astype(np.float32),device="cuda:0")
for tf32, fused in [(False, True), (False, False), (True, True), (True, False)]:
    m=GemmMLP(sizes,B,W,Bv,tf32=tf32,fused_epilogue=fused); m.prepare()
    print("tf32",tf32,"fused",fused,"step us", round(time_warp_graph(lambda: m.step(wx,wy), iters=50, reps=3),1))
