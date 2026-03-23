#=

This entire file was written by iterating with Claude.

=#

"""
    test_naff_cuda.jl  ─  CUDA test suite for NAFF

Tests that naff works correctly and performantly when `data` is a CuArray.
NAFF accepts any AbstractMatrix, so passing a CuArray dispatches FFT and linear
algebra operations to the GPU automatically via CUDA.jl / CUDA.CUFFT.

Run with:

    julia --project=. test/test_naff_cuda.jl

Requirements: a CUDA-capable GPU with CUDA.jl installed.

GPU vs CPU floating-point note
───────────────────────────────
  GPU arithmetic uses fused multiply-add (FMA) and a different floating-point
  reduction order compared to CPU.  Results therefore differ from CPU at the
  level of a few ULPs even for the same algorithm and inputs.  Tests comparing
  GPU output to CPU output use a tolerance of 1e-10 (not bit-identity) for
  this reason.  Tests comparing GPU output to the mathematical ground truth
  use the same tolerances as the CPU suite.

Test groups
───────────
  A. Device dispatch    – CuArray input returns CuArray output; stays on GPU
  B. Correctness        – GPU results match CPU results to 1e-10
  C. Multi-term         – 2/3-term decomposition on GPU matches CPU
  D. Window order       – window_order=1…4 work on GPU
  E. Scale              – 1 / 100 / 1k / 10k / 100k rows; shapes correct
  F. Accuracy at scale  – 1000-row GPU batch vs ground truth
  G. Performance        – 100k × 1024: GPU faster than CPU; throughput printed
  H. Data types         – ComplexF32 and ComplexF64 both work on GPU
  I. Determinism        – repeated GPU calls are bit-identical to each other;
                          GPU vs CPU agrees to 1e-10
"""

using Test
using CUDA
using QuasiperiodicFrequencies
using Printf, Random, Statistics, LinearAlgebra

# Skip the whole file gracefully if no GPU is available
if !CUDA.functional()
    @warn "No functional CUDA device found — skipping CUDA test suite."
    exit(0)
end

println("\n  CUDA device: $(CUDA.name(CUDA.device()))")
println("  Julia:       $(VERSION)")
println("  CUDA.jl:     $(pkgversion(CUDA))\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

cpu_tone(f, A, N) = A .* exp.(im .* 2π .* f .* (0:N-1))

function cpu_batch(f_truths, N; A=1.0+0im)
    vcat([reshape(cpu_tone(f, A, N), 1, N) for f in f_truths]...)
end

rand_freqs(n, rng; lo=0.05, hi=0.45) = lo .+ (hi - lo) .* rand(rng, n)

function warmup_gpu(N=256, nrows=4)
    data_gpu = CuArray(cpu_batch(rand_freqs(nrows, MersenneTwister(0)), N))
    naff(data_gpu, 1)
    CUDA.synchronize()
end

# ─────────────────────────────────────────────────────────────────────────────

@testset "NAFF CUDA" begin

warmup_gpu()

# ═════════════════════════════════════════════════════════════════════════════
@testset "A. Device dispatch" begin

    N = 512; f_true = 0.2731
    sig_cpu = cpu_tone(f_true, 1.0+0im, N)
    mat_gpu = CuArray(reshape(sig_cpu, 1, N))

    freqs_gpu, amps_gpu = naff(mat_gpu, 1)
    CUDA.synchronize()

    @testset "frequencies returned as CuArray" begin
        @test freqs_gpu isa CuArray
    end

    @testset "amplitudes returned as CuArray" begin
        @test amps_gpu isa CuArray
    end

    @testset "output shape is (1,1)" begin
        @test size(freqs_gpu) == (1, 1)
        @test size(amps_gpu)  == (1, 1)
    end

    @testset "result is on the same device as input" begin
        @test CUDA.device(freqs_gpu) == CUDA.device(mat_gpu)
        @test CUDA.device(amps_gpu)  == CUDA.device(mat_gpu)
    end

    @testset "GPU frequency is in (0, 0.5)" begin
        f = Array(freqs_gpu)[1,1]
        @test 0.0 < f < 0.5
    end

end # A

# ═════════════════════════════════════════════════════════════════════════════
@testset "B. Correctness — GPU vs CPU" begin

    rng = MersenneTwister(42)

    @testset "single signal: |Δf| < 1e-8 vs ground truth" begin
        N = 1000; f_true = sqrt(2)/10
        mat = reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)
        freqs_gpu, _ = naff(CuArray(mat), 1)
        CUDA.synchronize()
        @test abs(Array(freqs_gpu)[1,1] - f_true) < 1e-8
    end

    @testset "single signal: GPU agrees with CPU to 1e-10" begin
        N = 1000; f_true = sqrt(2)/10
        mat = reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)
        freqs_cpu, _ = naff(mat, 1)
        freqs_gpu, _ = naff(CuArray(mat), 1)
        CUDA.synchronize()
        @test abs(Array(freqs_gpu)[1,1] - freqs_cpu[1,1]) < 1e-10
    end

    @testset "50-row batch: all |Δf| < 1e-8 vs ground truth" begin
        N = 1000; nrows = 50
        f_truths  = rand_freqs(nrows, rng)
        batch_cpu = cpu_batch(f_truths, N)
        freqs_gpu, _ = naff(CuArray(batch_cpu), 1)
        CUDA.synchronize()
        @test maximum(abs.(Array(freqs_gpu)[:,1] .- f_truths)) < 1e-8
    end

    @testset "50-row batch: GPU agrees with CPU to 1e-10" begin
        N = 1000; nrows = 50
        f_truths  = rand_freqs(nrows, rng)
        batch_cpu = cpu_batch(f_truths, N)
        freqs_cpu, _ = naff(batch_cpu, 1)
        freqs_gpu, _ = naff(CuArray(batch_cpu), 1)
        CUDA.synchronize()
        @test maximum(abs.(Array(freqs_gpu)[:,1] .- freqs_cpu[:,1])) < 1e-10
    end

    @testset "amplitudes: rel err < 1e-4 GPU vs CPU" begin
        N = 1000; nrows = 20
        f_truths = rand_freqs(nrows, rng)
        A_truths = exp.(im .* 2π .* rand(rng, nrows))
        batch_cpu = vcat([reshape(cpu_tone(f, A, N), 1, N)
                          for (f, A) in zip(f_truths, A_truths)]...)
        _, amps_cpu = naff(batch_cpu, 1)
        _, amps_gpu = naff(CuArray(batch_cpu), 1)
        CUDA.synchronize()
        rel_errs = abs.(Array(amps_gpu)[:,1] .- amps_cpu[:,1]) ./ abs.(amps_cpu[:,1])
        @test maximum(rel_errs) < 1e-4
    end

    @testset "irrational frequency ($label): GPU |Δf| < 1e-10" for (label, f_true) in [
            ("√2/10", sqrt(2)/10), ("π/10", π/10), ("e/10", exp(1)/10)]
        N = 1000
        mat = reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)
        freqs_gpu, _ = naff(CuArray(mat), 1)
        CUDA.synchronize()
        @test abs(Array(freqs_gpu)[1,1] - f_true) < 1e-10
    end

end # B

# ═════════════════════════════════════════════════════════════════════════════
@testset "C. Multi-term on GPU" begin

    @testset "3 harmonics: all |Δf| < 1e-8 on GPU" begin
        N     = 2000
        comps = [(0.1234567, 2.0+1.0im), (0.2718281, 1.0-0.5im), (0.3141592, 0.5+0.5im)]
        sig   = sum(cpu_tone(f, A, N) for (f, A) in comps)
        freqs_gpu, _ = naff(CuArray(reshape(sig, 1, N)), 3)
        CUDA.synchronize()
        fs = sort(vec(Array(freqs_gpu[1,:])))
        @test maximum(abs.(fs .- sort(first.(comps)))) < 1e-8
    end

    @testset "5×2 batch on GPU: all |Δf| < 1e-7" begin
        N   = 2000
        f1s = [0.12, 0.18, 0.25, 0.31, 0.38]
        f2s = [0.30, 0.37, 0.43, 0.17, 0.22]
        batch_cpu = vcat([reshape(cpu_tone(fa, 1.0+0im, N) .+ cpu_tone(fb, 0.8+0im, N),
                                  1, N) for (fa, fb) in zip(f1s, f2s)]...)
        freqs_gpu, _ = naff(CuArray(batch_cpu), 2)
        CUDA.synchronize()
        freqs_h = Array(freqs_gpu)
        @test all(1:5) do i
            maximum(abs.(sort(vec(freqs_h[i,:])) .- sort([f1s[i], f2s[i]]))) < 1e-7
        end
    end

    @testset "nterms=1 and nterms=2 first column agree on GPU" begin
        N   = 1500
        sig = cpu_tone(0.20, 2.0+0im, N) .+ cpu_tone(0.35, 1.0+0im, N)
        mat = CuArray(reshape(sig, 1, N))
        f1b, _ = naff(mat, 1)
        f2b, _ = naff(mat, 2)
        CUDA.synchronize()
        @test Array(f1b)[1,1] == Array(f2b)[1,1]
    end

    @testset "output shape ($nrows × $nterms)" for (nrows, nterms) in [
            (1,1), (1,3), (10,2), (50,3)]
        N     = 1000
        comps = [(0.05 + k*0.08, 1.0+0im) for k in 0:4]
        sig   = sum(cpu_tone(f, A, N) for (f, A) in comps)
        batch = CuArray(vcat([reshape(sig, 1, N) for _ in 1:nrows]...))
        freqs_gpu, amps_gpu = naff(batch, nterms)
        CUDA.synchronize()
        @test size(freqs_gpu) == (nrows, nterms)
        @test size(amps_gpu)  == (nrows, nterms)
    end

    @testset "GPU 3-term matches CPU 3-term to 1e-10" begin
        N     = 2000
        comps = [(0.1234567, 2.0+1.0im), (0.2718281, 1.0-0.5im), (0.3141592, 0.5+0.5im)]
        sig   = sum(cpu_tone(f, A, N) for (f, A) in comps)
        mat   = reshape(sig, 1, N)
        freqs_cpu, _ = naff(mat, 3)
        freqs_gpu, _ = naff(CuArray(mat), 3)
        CUDA.synchronize()
        @test maximum(abs.(sort(vec(Array(freqs_gpu[1,:]))) .- sort(vec(freqs_cpu[1,:])))) < 1e-10
    end

end # C

# ═════════════════════════════════════════════════════════════════════════════
@testset "D. Window order on GPU" begin

    N = 1000; f_true = 0.3141592653
    mat_cpu = reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)
    mat_gpu = CuArray(mat_cpu)

    @testset "window_order=$p on GPU: |Δf| < 1e-7" for p in 1:4
        freqs_gpu, _ = naff(mat_gpu, 1; window_order=p)
        CUDA.synchronize()
        @test abs(Array(freqs_gpu)[1,1] - f_true) < 1e-7
    end

    @testset "GPU window_order=$p agrees with CPU to 1e-10" for p in 1:4
        freqs_cpu, _ = naff(mat_cpu, 1; window_order=p)
        freqs_gpu, _ = naff(mat_gpu, 1; window_order=p)
        CUDA.synchronize()
        @test abs(Array(freqs_gpu)[1,1] - freqs_cpu[1,1]) < 1e-10
    end

end # D

# ═════════════════════════════════════════════════════════════════════════════
@testset "E. Scale — shape and range" begin

    N = 512

    @testset "$nrows rows: output shape ($nrows, 1)" for nrows in [1, 100, 1_000, 10_000, 100_000]
        batch = CuArray(cpu_batch(fill(0.25, nrows), N))
        freqs_gpu, amps_gpu = naff(batch, 1)
        CUDA.synchronize()
        @test size(freqs_gpu) == (nrows, 1)
        @test size(amps_gpu)  == (nrows, 1)
    end

    @testset "100k rows: all freqs in (0, 0.5)" begin
        nrows = 100_000
        batch = CuArray(cpu_batch(fill(0.25, nrows), N))
        freqs_gpu, _ = naff(batch, 1)
        CUDA.synchronize()
        @test all(0.0 .< Array(freqs_gpu)[:,1] .< 0.5)
    end

    @testset "100k rows: all freqs |Δf| < 1e-5" begin
        nrows  = 100_000; f_true = 0.2731
        batch  = CuArray(cpu_batch(fill(f_true, nrows), N))
        freqs_gpu, _ = naff(batch, 1)
        CUDA.synchronize()
        @test maximum(abs.(Array(freqs_gpu)[:,1] .- f_true)) < 1e-5
    end

end # E

# ═════════════════════════════════════════════════════════════════════════════
@testset "F. Accuracy at scale — 1000-row GPU batch vs ground truth" begin

    rng      = MersenneTwister(7777)
    N        = 1000; nrows = 1000
    f_truths = rand_freqs(nrows, rng)
    A_truths = exp.(im .* 2π .* rand(rng, nrows))

    batch_cpu = vcat([reshape(cpu_tone(f, A, N), 1, N)
                      for (f, A) in zip(f_truths, A_truths)]...)
    batch_gpu = CuArray(batch_cpu)

    freqs_gpu, _ = naff(batch_gpu, 1)
    CUDA.synchronize()
    freqs_h = Array(freqs_gpu)[:,1]

    @testset "all 1000 freqs |Δf| < 1e-8 vs ground truth" begin
        @test maximum(abs.(freqs_h .- f_truths)) < 1e-8
    end

    @testset "median |Δf| < 1e-10" begin
        @test median(abs.(freqs_h .- f_truths)) < 1e-10
    end

    @testset "99th percentile |Δf| < 1e-9" begin
        errs = sort(abs.(freqs_h .- f_truths))
        p99  = errs[ceil(Int, 0.99 * length(errs))]
        @test p99 < 1e-9
    end

    @testset "GPU 1000-row agrees with CPU 1000-row to 1e-10" begin
        # GPU uses FMA and a different FP reduction order — bit-identity with
        # CPU is not guaranteed; 1e-10 captures any real algorithmic divergence.
        freqs_cpu, _ = naff(batch_cpu, 1)
        @test maximum(abs.(freqs_h .- freqs_cpu[:,1])) < 1e-10
    end

end # F

# ═════════════════════════════════════════════════════════════════════════════
@testset "G. Performance — 100k × 1024 signals" begin

    N          = 1024
    nrows_perf = 100_000
    rng_perf   = MersenneTwister(999)
    f_truths   = rand_freqs(nrows_perf, rng_perf)

    batch_cpu = cpu_batch(f_truths, N)
    batch_gpu = CuArray(batch_cpu)

    # Warm up JIT + cuFFT plan caching
    naff(batch_gpu[1:4, :], 1)
    CUDA.synchronize()

    # GPU: minimum over 3 trials
    gpu_times = map(1:3) do _
        CUDA.synchronize()
        t = time()
        naff(batch_gpu, 1)
        CUDA.synchronize()
        time() - t
    end
    t_gpu = minimum(gpu_times)

    # CPU: time a 1k-row run and extrapolate
    cpu_times = map(1:2) do _
        t = time()
        naff(batch_cpu[1:1_000, :], 1)
        time() - t
    end
    t_cpu_1k           = minimum(cpu_times)
    t_cpu_extrapolated = t_cpu_1k * (nrows_perf / 1_000)

    throughput_gpu = nrows_perf / t_gpu

    println("\n  ── Performance: 100k × 1024 signals ──────────────────────")
    @printf("  GPU time (best of 3):    %.3f s\n",   t_gpu)
    @printf("  CPU 1k-row (best of 2):  %.3f s  →  extrapolated 100k: %.1f s\n",
            t_cpu_1k, t_cpu_extrapolated)
    @printf("  GPU throughput:          %.0f signals/s  (%.1f k signals/s)\n",
            throughput_gpu, throughput_gpu/1000)
    @printf("  Estimated GPU speedup:   %.1f×\n", t_cpu_extrapolated / t_gpu)
    println()

    @testset "GPU completes 100k × 1024 in under 30 s" begin
        @test t_gpu < 30.0
    end

    @testset "GPU throughput > 5k signals/s" begin
        @test throughput_gpu > 5_000
    end

    @testset "GPU faster than extrapolated single-threaded CPU" begin
        @test t_gpu < t_cpu_extrapolated
    end

end # G

# ═════════════════════════════════════════════════════════════════════════════
@testset "H. Data types" begin

    N = 512; f_true = 0.25

    @testset "ComplexF64: eltype of outputs" begin
        mat = CuArray(ComplexF64.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        freqs, amps = naff(mat, 1)
        CUDA.synchronize()
        @test eltype(freqs) == Float64
        @test eltype(amps)  == ComplexF64
    end

    @testset "ComplexF64: |Δf| < 1e-8" begin
        mat = CuArray(ComplexF64.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        freqs, _ = naff(mat, 1)
        CUDA.synchronize()
        @test abs(Array(freqs)[1,1] - f_true) < 1e-8
    end

    @testset "ComplexF32: eltype of outputs" begin
        mat = CuArray(ComplexF32.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        freqs, amps = naff(mat, 1)
        CUDA.synchronize()
        @test eltype(freqs) == Float32
        @test eltype(amps)  == ComplexF32
    end

    @testset "ComplexF32: |Δf| < 1e-5" begin
        # F32 has ~7 significant digits — 1e-5 is realistic for N=512
        mat = CuArray(ComplexF32.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        freqs, _ = naff(mat, 1)
        CUDA.synchronize()
        @test abs(Array(freqs)[1,1] - Float32(f_true)) < 1e-5
    end

    @testset "ComplexF32 result close to ComplexF64 result |Δf| < 1e-5" begin
        mat64 = CuArray(ComplexF64.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        mat32 = CuArray(ComplexF32.(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N)))
        f64, _ = naff(mat64, 1)
        f32, _ = naff(mat32, 1)
        CUDA.synchronize()
        @test abs(Float64(Array(f32)[1,1]) - Array(f64)[1,1]) < 1e-5
    end

    @testset "ComplexF32 10k-row batch: shape and range" begin
        nrows = 10_000
        batch = CuArray(ComplexF32.(cpu_batch(fill(0.3f0, nrows), N)))
        freqs, _ = naff(batch, 1)
        CUDA.synchronize()
        @test size(freqs) == (nrows, 1)
        @test all(0.0f0 .< Array(freqs)[:,1] .< 0.5f0)
    end

end # H

# ═════════════════════════════════════════════════════════════════════════════
@testset "I. Determinism on GPU" begin

    N = 1000; f_true = sqrt(2)/10
    mat_gpu = CuArray(reshape(cpu_tone(f_true, 1.0+0im, N), 1, N))

    @testset "repeated calls give bit-identical results" begin
        f1b, a1 = naff(mat_gpu, 2; window_order=2)
        f2b, a2 = naff(mat_gpu, 2; window_order=2)
        CUDA.synchronize()
        @test Array(f1b) == Array(f2b)
        @test Array(a1)  == Array(a2)
    end

    @testset "batch result matches per-row result for row $i" for i in 1:10
        nrows    = 10; N2 = 800
        rng      = MersenneTwister(321)
        f_truths = rand_freqs(nrows, rng)
        sigs_cpu = [reshape(cpu_tone(f, 1.0+0im, N2), 1, N2) for f in f_truths]
        batch_gpu = CuArray(vcat(sigs_cpu...))

        freqs_batch, _ = naff(batch_gpu, 1)
        CUDA.synchronize()

        freqs_row, _ = naff(CuArray(sigs_cpu[i]), 1)
        CUDA.synchronize()
        @test Array(freqs_batch)[i,1] == Array(freqs_row)[1,1]
    end

    @testset "GPU and CPU agree to 1e-10 (FMA/reduction-order tolerance)" begin
        # GPU FMA and FP reduction order differ from CPU → not bit-identical,
        # but must agree to well within 1e-10 for double precision.
        nrows    = 20; N3 = 1000
        rng      = MersenneTwister(456)
        f_truths = rand_freqs(nrows, rng)
        batch_cpu = cpu_batch(f_truths, N3)
        batch_gpu = CuArray(batch_cpu)

        freqs_cpu, _ = naff(batch_cpu, 1)
        freqs_gpu, _ = naff(batch_gpu, 1)
        CUDA.synchronize()

        @test maximum(abs.(Array(freqs_gpu)[:,1] .- freqs_cpu[:,1])) < 1e-10
    end

end # I

end # NAFF CUDA

# ─────────────────────────────────────────────────────────────────────────────
# Memory summary
# ─────────────────────────────────────────────────────────────────────────────

println()
free, total = CUDA.Mem.info()
used = total - free
@printf("  GPU memory after tests:  %.1f MB used / %.1f MB total\n",
        used / 1e6, total / 1e6)
println()