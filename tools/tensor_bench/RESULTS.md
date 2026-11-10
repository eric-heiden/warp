# Tensor benchmark status (MIG 1g.24gb RTX PRO 6000 Blackwell slice, CUDA graphs, fp32 strict)

| workload | torch eager+graph | torch.compile+graph | Warp | status |
|---|---|---|---|---|
| gemm 4096^3 fp32 | 7276 us | 7222 | 7268 (`wp.gemm`) | tie (same cuBLASLt kernels) |
| gemm 4096^3 tf32 | 2850 | 2852 | 2861 | tie |
| elementwise chain 64M | 9423 | 3572 | 3575 (`wp.map`) | tie, bandwidth bound |
| layernorm+softmax 16k x 1k | 733 | 366 | 366 (register row kernel, `probe_lnsm2.py`) | tie, bandwidth bound |
| big MLP train step (8192 x 1024^3, fp32) | 10468 | 9667 | 9520 (`mlp_gemm.py`, unfused epilogue) | ~1.5% faster |
| big MLP train step, tf32 | n/a (not yet measured for torch) | n/a | 4798 | torch tf32 baseline missing |
| ActuatorNet fwd (49152 x 6-32-32-1) | 125 | 55.6 | 36.9 (fused tile kernel, `probe_tile_mlp.py`) | faster |
| RL policy fwd (4096 x 64-512-256-128-24) | 176 | 149.5 | 172 (gemm + pointwise) | NOT met |
| RL policy train, rl_policy_small, ActuatorNet train | 590 / 24.6 / 746 | 527 / 20.5 / 329 | not implemented | open |

Findings: wp.map is already at torch.compile speed under graph capture (only eager call overhead is higher, ~16 us vs ~11 us).
Fused tile kernels win for tiny nets, but cuBLASDx JIT is slow (~40 s per large tile shape), tile temporaries
are never freed (shared-memory limit ~100 KB on sm_120) so 512-wide layers do not fit, and a split-kernel
variant (`probe_rl_tile.py`) was both numerically wrong (accumulating into a broadcast tile) and 3x slower.
Warp NN was not touched; no Warp NN code is in this branch.
