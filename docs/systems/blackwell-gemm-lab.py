"""Source-pinned SM100 BF16 GEMM lab; requires a reviewed CUTLASS checkout."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time


CUTLASS_COMMIT = "2c85f497547eb441f0fe80bfb0fb2e6f9add1719"
DSL_VERSION = "4.8.1"
REFERENCE_PATH = Path(
    "examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
)
REFERENCE_SHA256 = "f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-root", required=True, type=Path)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--two-cta", action="store_true")
    parser.add_argument("--direct-store", action="store_true")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--pattern", choices=("random", "zeros", "identity"), default="random")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Isolate one validated kernel launch")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.iterations, args.trials) <= 0:
        parser.error("Dimensions, iterations, and trials must be positive")
    if args.n % 8 or args.k % 8:
        parser.error("This BF16 layout requires N and K to be multiples of 8 (16-byte strides)")
    if args.direct_store and (args.m % 128 or args.n % args.tile_n):
        parser.error("Direct stores require complete 128-by-tile_n CTA output tiles")
    return args


def reference_source(root):
    source = root.resolve() / REFERENCE_PATH
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(
            f"Reference source differs from CUTLASS {CUTLASS_COMMIT}: {source}. "
            "Review changes and establish a new pin; do not bypass this check."
        )
    return source


def load_reference(source):
    spec = importlib.util.spec_from_file_location("blackwell_dense_reference", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reviewed reference: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DenseGemmKernel


def time_batch(torch, operation, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    args = parse_args()
    source = reference_source(args.cutlass_root)
    installed_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if installed_version != DSL_VERSION:
        raise RuntimeError(f"Expected CuTe DSL {DSL_VERSION}, found {installed_version}")

    import torch
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    if not torch.cuda.is_available():
        raise RuntimeError("This lab requires a CUDA-enabled environment and an SM100 GPU")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("This bounded lab targets SM100 (B200/GB200), not SM103 or SM120")
    torch_device = torch.device("cuda", device)

    DenseGemmKernel = load_reference(source)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    a_storage = torch.randn((1, args.m, args.k), device=torch_device, dtype=torch.bfloat16)
    b_storage = torch.randn((1, args.k, args.n), device=torch_device, dtype=torch.bfloat16)
    if args.pattern == "random":
        a_storage.div_(math.sqrt(args.k))
    elif args.pattern == "zeros":
        a_storage.zero_()
    else:
        a_storage.zero_()
        a_storage[0].diagonal().fill_(1)
    c_storage = torch.full(
        (1, args.m, args.n), float("nan"), device=torch_device, dtype=torch.bfloat16
    )
    baseline_output = torch.empty_like(c_storage[0])

    # CuTe GEMM uses B[N,K,L], not the conventional PyTorch B[K,N].
    a = a_storage.permute(1, 2, 0)
    b = b_storage.permute(2, 1, 0)
    c = c_storage.permute(1, 2, 0)
    views = tuple(from_dlpack(tensor, assumed_align=16) for tensor in (a, b, c))
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    gemm = DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=args.two_cta,
        mma_tiler_mn=(256 if args.two_cta else 128, args.tile_n),
        cluster_shape_mn=(2, 1) if args.two_cta else (1, 1),
        use_tma_store=not args.direct_store,
    )
    if not gemm.can_implement(*views):
        raise ValueError("The pinned kernel rejects this shape, layout, or configuration")

    def identity(x):
        return x

    def relu(x):
        return cute.where(x > 0, x, cute.full_like(x, 0))

    compile_start = time.perf_counter()
    compiled = cute.compile(gemm, *views, stream, relu if args.relu else identity)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    def launch():
        compiled(*views, stream)

    def baseline():
        torch.mm(a_storage[0], b_storage[0], out=baseline_output)
        if args.relu:
            baseline_output.relu_()

    launch()
    baseline()
    torch.cuda.synchronize()
    reference = (a_storage[0].float() @ b_storage[0].float()).to(torch.bfloat16)
    if args.relu:
        reference.relu_()
    atol, rtol = (0.01, 0.02) if args.pattern == "random" else (0.0, 0.0)
    for name, result in (("CuTe", c_storage[0]), ("PyTorch", baseline_output)):
        max_abs = (result.float() - reference.float()).abs().max().item()
        print(f"{name}: max_abs_error={max_abs:.6g}, atol={atol}, rtol={rtol}")
        torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)

    print(json.dumps({
        "status": "correctness_passed",
        "cutlass_commit": CUTLASS_COMMIT,
        "reference_sha256": REFERENCE_SHA256,
        "dsl": installed_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "pytorch_blas_preference": str(torch.backends.cuda.preferred_blas_library()),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "shape_mnk": [args.m, args.n, args.k],
        "strides_a_b_c": [list(tensor.stride()) for tensor in (a, b, c)],
        "pattern": args.pattern,
        "seed": args.seed,
        "mma_tiler_mnk": list(gemm.mma_tiler),
        "cta_tile_mnk": list(gemm.cta_tile_shape_mnk),
        "cluster_mn": list(gemm.cluster_shape_mn),
        "ab_stages": gemm.num_ab_stage,
        "acc_stages": gemm.num_acc_stage,
        "tmem_columns": gemm.num_tmem_alloc_cols,
        "tma_store": not args.direct_store,
        "relu": args.relu,
        "compile_ms_not_kernel_time": compile_ms,
    }, sort_keys=True))

    for _ in range(10):
        launch()
        baseline()
    torch.cuda.synchronize()
    if args.profile:
        torch.cuda.profiler.start()
        try:
            launch()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        return

    samples = {"cute": [], "pytorch": []}
    operations = (("cute", launch), ("pytorch", baseline))
    for trial in range(args.trials):
        order = operations if trial % 2 == 0 else tuple(reversed(operations))
        for name, operation in order:
            samples[name].append(time_batch(torch, operation, args.iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    flops = 2 * args.m * args.n * args.k
    print(json.dumps({
        "timing": "warm_same_buffers_cuda_events_no_graph",
        "iterations_per_trial": args.iterations,
        "samples_ms": samples,
        "median_ms": medians,
        "cute_tflops": flops / (medians["cute"] * 1e9),
        "pytorch_tflops": flops / (medians["pytorch"] * 1e9),
        "pytorch_over_cute_time_ratio": medians["pytorch"] / medians["cute"],
        "baseline": "torch.mm + separate in-place ReLU" if args.relu else "torch.mm",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
