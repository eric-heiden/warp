"""MLP training step built on wp.gemm (cuBLASLt epilogues) + small fused kernels. Graph-capturable."""
import numpy as np
import warp as wp

MSE_BD = 256

@wp.kernel
def mse_grad(out: wp.array[float], y: wp.array[float], d: wp.array[float], loss: wp.array[float], inv_n: float, n: int):
    blk, t = wp.tid()
    i = blk * MSE_BD + t
    l = float(0.0)
    if i < n:
        e = out[i] - y[i]
        d[i] = 2.0 * e * inv_n
        l = e * e * inv_n
    s = wp.tile_sum(wp.tile(l))
    if t == 0:
        wp.atomic_add(loss, 0, s[0])

@wp.kernel
def adam_step(p: wp.array[float], g: wp.array[float], m: wp.array[float], v: wp.array[float], step: wp.array[int],
              lr: float, b1: float, b2: float, eps: float):
    i = wp.tid()
    t = float(step[0])
    gi = g[i]
    mi = b1 * m[i] + (1.0 - b1) * gi
    vi = b2 * v[i] + (1.0 - b2) * gi * gi
    m[i] = mi
    v[i] = vi
    mh = mi / (1.0 - wp.pow(b1, t))
    vh = vi / (1.0 - wp.pow(b2, t))
    p[i] = p[i] - lr * mh / (wp.sqrt(vh) + eps)

@wp.func
def gelu_tanh(z: float):
    u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    return 0.5 * z * (1.0 + wp.tanh(u))

@wp.func
def gelu_tanh_grad(z: float):
    u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    t = wp.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z)

@wp.kernel
def k_gelu(z: wp.array[float], x: wp.array[float]):
    i = wp.tid()
    x[i] = gelu_tanh(z[i])

@wp.kernel
def k_dgelu(dx: wp.array[float], z: wp.array[float], dz: wp.array[float]):
    i = wp.tid()
    dz[i] = dx[i] * gelu_tanh_grad(z[i])

@wp.kernel
def inc(step: wp.array[int]):
    step[0] = step[0] + 1

@wp.kernel
def zero1(a: wp.array[float]):
    a[0] = 0.0

class GemmMLP:
    def __init__(self, sizes, batch, W, B, device="cuda:0", tf32=False, fused_epilogue=True):
        self.fused = fused_epilogue
        self.sizes, self.batch, self.dev, self.tf32 = sizes, batch, device, tf32
        L = self.L = len(sizes) - 1
        z = lambda *s: wp.zeros(s, dtype=float, device=device)
        self.W = [wp.array(w, dtype=float, device=device) for w in W]
        self.b = [wp.array(b, dtype=float, device=device) for b in B]
        self.gW = [z(*w.shape) for w in W]
        self.gb = [z(b.shape[0]) for b in B]
        self.X = [None] + [z(batch, s) for s in sizes[1:]]
        self.Z = [z(batch, s) for s in sizes[1:-1]]
        self.dZ = [z(batch, s) for s in sizes[1:]]
        self.tmp = z(batch, max(sizes[1:-1]))
        self.ones = wp.array(np.ones((1, batch), np.float32), device=device)
        self.m = [z(*w.shape) for w in W] + [z(*b.shape) for b in B]
        self.v = [z(*w.shape) for w in W] + [z(*b.shape) for b in B]
        self.step_ctr = wp.zeros(1, dtype=int, device=device)
        self.loss = z(1)

    def forward(self, x):
        self.X[0] = x
        g = lambda *a, **k: wp.gemm(*a, tf32=self.tf32, **k)
        for l in range(self.L):
            if l < self.L - 1 and self.fused:
                g(self.X[l], self.W[l], self.X[l + 1], bias=self.b[l], activation="gelu", aux=self.Z[l])
            elif l < self.L - 1:
                g(self.X[l], self.W[l], self.Z[l], bias=self.b[l])
                wp.launch(k_gelu, self.Z[l].size, [self.Z[l].flatten(), self.X[l + 1].flatten()], device=self.dev)
            else:
                g(self.X[l], self.W[l], self.X[l + 1], bias=self.b[l])
        return self.X[-1]

    def backward(self, y):
        g = lambda *a, **k: wp.gemm(*a, tf32=self.tf32, **k)
        L = self.L
        wp.launch(zero1, 1, [self.loss], device=self.dev)
        n = self.X[L].size
        wp.launch_tiled(mse_grad, dim=[(n + MSE_BD - 1) // MSE_BD], inputs=[self.X[L].flatten(), y.flatten(), self.dZ[L - 1].flatten(), self.loss, 1.0 / n, n], block_dim=MSE_BD, device=self.dev)
        for l in range(L - 1, -1, -1):
            g(self.X[l], self.dZ[l], self.gW[l], trans_a=True)
            if l == L - 1:
                g(self.ones, self.dZ[l], self._gb2d[l])
            if l > 0 and self.fused:
                g(self.dZ[l], self.W[l], self.dZ[l - 1], trans_b=True, activation="gelu", grad_aux=self.Z[l - 1], bias_grad=self.gb[l - 1])
            elif l > 0:
                g(self.dZ[l], self.W[l], self.tmp, trans_b=True)
                wp.launch(k_dgelu, self.Z[l - 1].size, [self.tmp.flatten(), self.Z[l - 1].flatten(), self.dZ[l - 1].flatten()], device=self.dev)
                g(self.ones, self.dZ[l - 1], self._gb2d[l - 1])

    def optimizer(self, lr=1e-3):
        wp.launch(inc, 1, [self.step_ctr], device=self.dev)
        params = self.W + self.b; grads = self.gW + self.gb
        for p, gr, m, v in zip(params, grads, self.m, self.v):
            wp.launch(adam_step, p.size, [p.flatten(), gr.flatten(), m.flatten(), v.flatten(), self.step_ctr, lr, 0.9, 0.999, 1e-8], device=self.dev)

    def prepare(self):
        self._gb2d = {l: wp.array(ptr=self.gb[l].ptr, dtype=float, shape=(1, self.sizes[l + 1]), device=self.dev, copy=False) for l in range(self.L)}

    def step(self, x, y):
        self.forward(x); self.backward(y); self.optimizer()
