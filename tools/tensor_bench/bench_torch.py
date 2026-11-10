import sys, json, torch, torch.nn.functional as F
from common import time_torch_graph, time_torch_eager
from suite import WORKLOADS
import workloads_torch as W

ACTS = {"gelu": lambda x: F.gelu(x, approximate="tanh"), "elu": F.elu, "softsign": F.softsign}

def build(name, spec):
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    k = spec["kind"]
    if k == "gemm":
        torch.backends.cuda.matmul.allow_tf32 = spec["tf32"]
        a = torch.randn(spec["m"], spec["k"], device="cuda", generator=g); b = torch.randn(spec["k"], spec["n"], device="cuda", generator=g)
        return (lambda: W.gemm(a, b)), W.gemm, (a, b)
    torch.backends.cuda.matmul.allow_tf32 = False
    if k == "ew":
        n = spec["n"]
        x, w, b, r = (torch.randn(n, device="cuda", generator=g) for _ in range(4))
        return (lambda: W.ew_chain(x, w, b, r)), W.ew_chain, (x, w, b, r)
    if k == "lnsm":
        x = torch.randn(spec["rows"], spec["cols"], device="cuda", generator=g)
        gm = torch.ones(spec["cols"], device="cuda"); bt = torch.zeros(spec["cols"], device="cuda")
        return (lambda: W.layernorm_softmax(x, gm, bt)), W.layernorm_softmax, (x, gm, bt)
    sizes = spec["sizes"]; act = ACTS[spec["act"]]
    ps = W.make_mlp_params(sizes, g)
    x = torch.randn(spec["batch"], sizes[0], device="cuda", generator=g)
    if k == "mlp_fwd":
        with torch.no_grad():
            f = lambda: W.mlp_fwd(x, ps, act)
        return f, (lambda x, *p: W.mlp_fwd(x, list(p), act)), (x, *ps)
    y = torch.randn(spec["batch"], sizes[-1], device="cuda", generator=g)
    return None, None, (x, y, ps, act)

def run_train(spec, compile_):
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    sizes = spec["sizes"]; act = ACTS[spec["act"]]
    ps = W.make_mlp_params(sizes, g)
    x = torch.randn(spec["batch"], sizes[0], device="cuda", generator=g)
    y = torch.randn(spec["batch"], sizes[-1], device="cuda", generator=g)
    opt = torch.optim.Adam(ps, lr=1e-3, capturable=True, fused=True)
    def loss_fn(x, y, *p):
        return F.mse_loss(W.mlp_fwd(x, list(p), act), y)
    lf = torch.compile(loss_fn, mode="max-autotune-no-cudagraphs") if compile_ else loss_fn
    def step():
        loss = lf(x, y, *ps); loss.backward(); opt.step()
    # warmup eagerly on side stream per official CUDA-graph recipe (grads must exist)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            opt.zero_grad(set_to_none=True); step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        step()
    for _ in range(10): gr.replay()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100): gr.replay()
        b.record(); torch.cuda.synchronize(); best = min(best, a.elapsed_time(b) * 10)
    return best

def main():
    names = sys.argv[1:] or list(WORKLOADS)
    out = {}
    for name in names:
        spec = WORKLOADS[name]
        r = {}
        if spec["kind"] == "mlp_train":
            r["eager_graph"] = run_train(spec, False)
            r["compile_graph"] = run_train(spec, True)
        else:
            f, fn, args = build(name, spec)
            r["eager"] = time_torch_eager(f)
            r["eager_graph"] = time_torch_graph(f)
            cf = torch.compile(fn, mode="max-autotune-no-cudagraphs")
            with torch.no_grad():
                r["compile_graph"] = time_torch_graph(lambda: cf(*args))
        out[name] = r
        print(name, {k: round(v, 1) for k, v in r.items()}, flush=True)
    json.dump(out, open("results_torch.json", "w"), indent=1)

if __name__ == "__main__":
    main()
