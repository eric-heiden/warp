"""Shared benchmark helpers: CUDA-graph timing for PyTorch and Warp workloads."""
import time
import torch

def time_torch_graph(fn, warmup=5, iters=200, reps=5):
    """Capture ``fn`` once in a torch CUDA graph, replay it ``iters`` times, return best-of-reps us/iter."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / iters)
    return best

def time_torch_eager(fn, warmup=5, iters=100, reps=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / iters)
    return best

def time_warp_graph(fn, warmup=3, iters=200, reps=5, device="cuda:0"):
    """Capture ``fn`` in a Warp CUDA graph and replay it. Returns best-of-reps us/iter."""
    import warp as wp
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as cap:
        fn()
    for _ in range(warmup):
        wp.capture_launch(cap.graph)
    wp.synchronize_device(device)
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
        best = min(best, (time.perf_counter() - t0) * 1e6 / iters)
    return best
