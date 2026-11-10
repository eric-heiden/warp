"""Workload definitions shared between backends (sizes only)."""
WORKLOADS = {
    # name: dict(kind, ...)
    "gemm_4096_fp32":    dict(kind="gemm", m=4096, n=4096, k=4096, tf32=False),
    "gemm_4096_tf32":    dict(kind="gemm", m=4096, n=4096, k=4096, tf32=True),
    "ew_chain_64M":      dict(kind="ew", n=64 * 1024 * 1024),
    "ln_softmax_16kx1k": dict(kind="lnsm", rows=16384, cols=1024),
    "mlp_big_train":     dict(kind="mlp_train", batch=8192, sizes=[1024, 1024, 1024, 1024], act="gelu"),
    "rl_policy_fwd":     dict(kind="mlp_fwd", batch=4096, sizes=[64, 512, 256, 128, 24], act="elu"),
    "rl_policy_train":   dict(kind="mlp_train", batch=4096, sizes=[64, 512, 256, 128, 24], act="elu"),
    "rl_policy_small":   dict(kind="mlp_fwd", batch=1024, sizes=[48, 128, 128, 12], act="elu"),
    "actuator_net_fwd":  dict(kind="mlp_fwd", batch=49152, sizes=[6, 32, 32, 1], act="softsign"),
    "actuator_net_train":dict(kind="mlp_train", batch=49152, sizes=[6, 32, 32, 1], act="softsign"),
}
