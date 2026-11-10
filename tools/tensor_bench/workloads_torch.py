"""PyTorch reference implementations (eager and torch.compile) of the benchmark workloads."""
import torch, torch.nn.functional as F
dev = "cuda"

def make_mlp_params(sizes, g):
    ps = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        w = (torch.randn(i, o, device=dev, generator=g) / i**0.5).requires_grad_()
        b = torch.zeros(o, device=dev, requires_grad=True)
        ps += [w, b]
    return ps

def mlp_fwd(x, ps, act):
    n = len(ps) // 2
    for k in range(n):
        x = x @ ps[2 * k] + ps[2 * k + 1]
        if k < n - 1:
            x = act(x)
    return x

def gemm(a, b): return a @ b
def ew_chain(x, w, b, r): return F.gelu(x * w + b, approximate="tanh") * 0.5 + r
def layernorm_softmax(x, g, b):
    return torch.softmax(F.layer_norm(x, x.shape[-1:], g, b), dim=-1)
