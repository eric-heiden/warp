import numpy as np, torch, warp as wp, torch.nn.functional as F
from mlp_gemm import GemmMLP
import sys; FUSED = sys.argv[1]=="1"
wp.config.quiet = True
sizes=[256,256,256,16]; B=512
g=np.random.default_rng(1)
W=[(g.standard_normal((i,o))/np.sqrt(i)).astype(np.float32) for i,o in zip(sizes[:-1],sizes[1:])]
Bv=[(0.1*g.standard_normal(o)).astype(np.float32) for o in sizes[1:]]
x=g.standard_normal((B,sizes[0])).astype(np.float32); y=g.standard_normal((B,sizes[-1])).astype(np.float32)
m=GemmMLP(sizes,B,W,Bv,fused_epilogue=FUSED); m.prepare()
wx=wp.array(x,device="cuda:0"); wy=wp.array(y,device="cuda:0")
m.forward(wx); m.backward(wy)
tp=[torch.tensor(a,device="cuda",requires_grad=True) for pair in zip(W,Bv) for a in pair]
h=torch.tensor(x,device="cuda")
for l in range(3):
    h=h@tp[2*l]+tp[2*l+1]
    if l<2: h=F.gelu(h,approximate="tanh")
loss=F.mse_loss(h,torch.tensor(y,device="cuda")); loss.backward()
print("loss",float(loss),m.loss.numpy())
for l in range(3):
    print(l,"gW err",np.abs(m.gW[l].numpy()-tp[2*l].grad.cpu().numpy()).max(),"gb err",np.abs(m.gb[l].numpy()-tp[2*l+1].grad.cpu().numpy()).max())
