import numpy as np, warp as wp
from common import time_warp_graph
wp.config.quiet = True
d="cuda:0"; sizes=[64,512,256,128,24]; B=4096
@wp.kernel
def k_elu(z: wp.array2d[float], bias: wp.array[float], x: wp.array2d[float]):
    i, j = wp.tid()
    v = z[i, j] + bias[j]
    x[i, j] = wp.where(v > 0.0, v, wp.exp(v) - 1.0)
g=np.random.default_rng(0)
W=[wp.array((g.standard_normal((i,o))/np.sqrt(i)).astype(np.float32),device=d) for i,o in zip(sizes[:-1],sizes[1:])]
b=[wp.zeros(o,dtype=float,device=d) for o in sizes[1:]]
x=wp.array(g.standard_normal((B,sizes[0])).astype(np.float32),device=d)
H=[x]+[wp.zeros((B,o),dtype=float,device=d) for o in sizes[1:]]
Z=[None]+[wp.zeros((B,o),dtype=float,device=d) for o in sizes[1:-1]]
def f(tf32=False):
    for l in range(4):
        if l<3:
            wp.gemm(H[l],W[l],Z[l+1],tf32=tf32); wp.launch(k_elu,H[l+1].shape,[Z[l+1],b[l],H[l+1]],device=d)
        else: wp.gemm(H[l],W[l],H[l+1],bias=b[l],tf32=tf32)
for tf32 in (False,True): print("tf32",tf32,"rl fwd gemm+pointwise us",round(time_warp_graph(lambda: f(tf32)),1))
