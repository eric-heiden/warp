import numpy as np, torch, warp as wp
from common import time_warp_graph, time_torch_graph
wp.config.quiet = True
for (M,N,K,tf32) in [(4096,4096,4096,False),(4096,4096,4096,True),(8192,1024,1024,False)]:
    a = np.random.randn(M,K).astype(np.float32); b = np.random.randn(K,N).astype(np.float32)
    wa, wb = wp.array(a, device="cuda:0"), wp.array(b, device="cuda:0"); wc = wp.empty((M,N), dtype=float, device="cuda:0")
    wp.gemm(wa, wb, wc, tf32=tf32); 
    ref = torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda() if not tf32 else None
    if ref is not None: print("maxerr", float(np.abs(wc.numpy()-ref.cpu().numpy()).max()))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    print(M,N,K,"tf32" if tf32 else "fp32","warp", round(time_warp_graph(lambda: wp.gemm(wa, wb, wc, tf32=tf32)),1), "torch", round(time_torch_graph(lambda: ta@tb),1))
