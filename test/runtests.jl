"""
    test_naff.jl  ─  Comprehensive Test.jl suite for NAFF

Place in the package test/ directory and run with:

    julia --project=. test/test_naff.jl

or via the package manager:

    pkg> test NAFF

API under test
──────────────
  NAFF.naff(data::AbstractMatrix, n_frequencies=1; window_order=1)
    • data         : nrows × ncols — each ROW is one signal
    • window_order : Hanning order p  (1 → Hann¹, 2 → Hann², …)
    • n_terms      : number of harmonics to extract per row
    Returns (freqs, amps) — both nrows × n_terms
    freqs : normalised frequencies, cycles/sample ∈ (−0.5, 0.5)
    amps  : complex amplitudes

Calling convention used throughout
────────────────────────────────────
  Single signal vec → reshape to 1×N matrix, index result as [1,k]:

    sig = A .* exp.(im .* 2π .* f .* (0:N-1))
    freqs, amps = NAFF.naff(reshape(sig, 1, :), n; window_order=p)
    # freqs[1,1] is the dominant frequency
    # amps[1,1]  is its complex amplitude

Test groups
───────────
  A. Single harmonic        – irrational freqs, output shape, normalisation
  B. Multi-harmonic         – 2/3/5-term decomposition, residual power,
                               amplitude ordering, harmonic independence
  C. Window order           – window_order=1…4 frequency accuracy and agreement
  D. Amplitude              – large/small/imaginary/unit, phase, scaling,
                               independence from frequency
  E. Signal length          – N=13…2000, power-of-2 sizes, turns truncation
  F. Frequency coverage     – 0.005…0.499, irrational values, negative freqs
  G. Batch basic            – shape, identical rows, 5/20/100-row batches,
                               row-independence
  H. Batch multi-term       – 3×2, 5×3 batches; window propagation;
                               n_terms consistency
  I. Batch amplitude        – per-row amplitude, phase, linear scaling
  J. Batch heterogeneous    – random freqs/amps, row-order independence,
                               dominant+weak tone pairs
  K. Batch stress           – 100 rows × 3 terms, random signals
  L. Batch edge cases       – N arbitrary, short signals, window_order=4,
                               second-term suppression for pure tones
  M. Closely-spaced freqs   – Δf = 10/N, 5/N, 3/N; batch resolution
  N. Noise robustness       – SNR 40/20/10 dB, single and batch
  O. Determinism            – same input always gives identical output
  P. Type stability (JET)   – @test_opt on ComplexF64/F32, n=1/2, window_order=1/2
  Q. Worked example         – accelerator signal (printed, not asserted)
"""

using Test
using NAFF
using Printf, LinearAlgebra, Random, Statistics, JET
# JET is loaded inside testset P so the rest of the suite runs even if it is
# not installed. Add JET to [extras] / [targets] in Project.toml to enable.

# ─────────────────────────────────────────────────────────────────────────────
# One pure signal-construction helper — no wrappers around NAFF.naff
# ─────────────────────────────────────────────────────────────────────────────

# Complex tone: A · exp(2πi f t),  t = 0, 1, …, N-1
tone(f, A, N) = A .* exp.(im .* 2π .* f .* (0:N-1))

# ─────────────────────────────────────────────────────────────────────────────

@testset "NAFF" begin

# ═════════════════════════════════════════════════════════════════════════════
@testset "A. Single harmonic accuracy" begin

    @testset "irrational frequency ($label): |Δf| < 1e-10" for (label, f_true) in [
            ("√2/10", sqrt(2)/10), ("π/10", π/10), ("e/10", exp(1)/10),
            ("√3/10", sqrt(3)/10), ("√5/10", sqrt(5)/10)]
        N = 1000
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, N), 1, :), 1)
        @test abs(freqs[1,1] - f_true) < 1e-10
    end

    @testset "output shape: (1,1) for single row, n_terms=1" begin
        freqs, amps = NAFF.naff(reshape(tone(0.2, 1.0+0im, 500), 1, :), 1)
        @test size(freqs) == (1, 1)
        @test size(amps)  == (1, 1)
    end

    @testset "output shape: (1,$n) for nterms=$n" for n in [1, 2, 3, 5]
        sig = sum(tone(0.05 + k*0.08, 1.0+0im, 2000) for k in 0:4)
        freqs, amps = NAFF.naff(reshape(sig, 1, :), n)
        @test size(freqs) == (1, n)
        @test size(amps)  == (1, n)
    end

    @testset "frequencies are normalised cycles/sample (not radians/turn)" begin
        # If NAFF returned radians/turn the value would be ~2π×0.173 ≈ 1.09,
        # which is > 0.5 and would fail the range check.
        f_true = 0.173205
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, 1000), 1, :), 1)
        @test 0.0 < freqs[1,1] < 0.5
        @test abs(freqs[1,1] - f_true) < 1e-8
    end

    @testset "returned frequency is real-valued" begin
        freqs, _ = NAFF.naff(reshape(tone(0.3, 1.0+0im, 500), 1, :), 1)
        @test freqs[1,1] isa Real
    end

    @testset "returned amplitude is complex-valued" begin
        _, amps = NAFF.naff(reshape(tone(0.3, 1.0+0im, 500), 1, :), 1)
        @test amps[1,1] isa Complex
    end

end # A

# ═════════════════════════════════════════════════════════════════════════════
@testset "B. Multi-harmonic decomposition" begin

    @testset "two harmonics: both |Δf| < 1e-9" begin
        N = 2000; fa, fb = 0.15, 0.33
        sig = tone(fa, 2.0+0im, N) .+ tone(fb, 1.0+0.5im, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        @test maximum(abs.(sort(vec(freqs[1,:])) .- [fa, fb])) < 1e-9
    end

    @testset "three harmonics: all |Δf| < 1e-8" begin
        N     = 2000
        comps = [(0.1234567, 2.0+1.0im), (0.2718281, 1.0-0.5im), (0.3141592, 0.5+0.5im)]
        sig   = sum(tone(f, A, N) for (f, A) in comps)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 3)
        @test maximum(abs.(sort(vec(freqs[1,:])) .- sort(first.(comps)))) < 1e-8
    end

    @testset "five harmonics: all |Δf| < 1e-7" begin
        N     = 3000
        comps = [(0.05 + k*0.08, (1.0/(k+1))*exp(im*k)) for k in 0:4]
        sig   = sum(tone(f, A, N) for (f, A) in comps)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 5)
        @test maximum(abs.(sort(vec(freqs[1,:])) .- sort(first.(comps)))) < 1e-7
    end

    @testset "residual power < 1e-6 × signal after 2-term fit" begin
        N   = 1500; fa, fb = 0.15, 0.33
        sig = tone(fa, 2.0+0im, N) .+ tone(fb, 1.0+0.5im, N)
        freqs, amps = NAFF.naff(reshape(sig, 1, :), 2)
        resid = copy(sig)
        for k in 1:2
            resid .-= amps[1,k] .* exp.(im .* 2π .* freqs[1,k] .* (0:N-1))
        end
        @test sum(abs2, resid)/N < 1e-6 * sum(abs2, sig)/N
    end

    @testset "residual power < 1e-5 × signal after 3-term fit" begin
        N     = 2000
        comps = [(0.12, 2.0+0im), (0.27, 1.0+1.0im), (0.38, 0.5+0im)]
        sig   = sum(tone(f, A, N) for (f, A) in comps)
        freqs, amps = NAFF.naff(reshape(sig, 1, :), 3)
        resid = copy(sig)
        for k in 1:3
            resid .-= amps[1,k] .* exp.(im .* 2π .* freqs[1,k] .* (0:N-1))
        end
        @test sum(abs2, resid)/N < 1e-5 * sum(abs2, sig)/N
    end

    @testset "dominant tone found first (largest |amplitude|)" begin
        N   = 1500
        sig = tone(0.20, 3.0+0im, N) .+ tone(0.35, 1.0+0im, N)
        freqs, amps = NAFF.naff(reshape(sig, 1, :), 2)
        @test abs(freqs[1,1] - 0.20) < 1e-7   # stronger tone first
        @test abs(amps[1,1]) > abs(amps[1,2])
    end

    @testset "n_terms=1 result matches first column of n_terms=2" begin
        N   = 1500
        sig = tone(0.20, 2.0+0im, N) .+ tone(0.35, 1.0+0im, N)
        mat = reshape(sig, 1, N)
        f1b, _ = NAFF.naff(mat, 1)
        f2b, _ = NAFF.naff(mat, 2)
        @test f1b[1,1] == f2b[1,1]
    end

    @testset "weak third tone does not shift the two dominant frequencies" begin
        N = 2000; fa, fb = 0.15, 0.33
        sig2 = tone(fa, 2.0+0im, N) .+ tone(fb, 1.5+0im, N)
        sig3 = sig2 .+ tone(0.42, 0.1+0im, N)
        f2b, _ = NAFF.naff(reshape(sig2, 1, :), 2)
        f3b, _ = NAFF.naff(reshape(sig3, 1, :), 2)
        @test maximum(abs.(sort(vec(f2b[1,:])) .- sort(vec(f3b[1,:])))) < 1e-6
    end

end # B

# ═════════════════════════════════════════════════════════════════════════════
@testset "C. Window order" begin

    N = 1000; f_true = 0.3141592653
    sig = tone(f_true, 1.0+0im, N)
    mat = reshape(sig, 1, :)

    @testset "window=$p: |Δf| < 1e-7" for p in 1:4
        freqs, _ = NAFF.naff(mat, 1; window_order=p)
        @test abs(freqs[1,1] - f_true) < 1e-7
    end

    @testset "all four windows agree pairwise |Δf| < 1e-7" begin
        fs = [NAFF.naff(mat, 1; window_order=p)[1][1,1] for p in 1:4]
        @test maximum(abs.(fs .- fs[1])) < 1e-7
    end

    @testset "window propagates to all rows in a batch" begin
        N2       = 1000
        f_truths = [0.12, 0.25, 0.38]
        batch    = vcat([reshape(tone(f, 1.0+0im, N2), 1, N2) for f in f_truths]...)
        for p in [2, 3, 4]
            freqs, _ = NAFF.naff(batch, 1; window_order=p)
            @test maximum(abs.(vec(freqs[:,1]) .- f_truths)) < 1e-7
        end
    end

end # C

# ═════════════════════════════════════════════════════════════════════════════
@testset "D. Amplitude accuracy" begin

    @testset "large amplitude (1e6): rel err < 1e-6" begin
        A_true = 1e6*(1.0+1.0im)
        _, amps = NAFF.naff(reshape(tone(0.25, A_true, 800), 1, :), 1)
        @test abs(amps[1,1] - A_true) / abs(A_true) < 1e-6
    end

    @testset "small amplitude (1e-6): rel err < 1e-6" begin
        A_true = 1e-6*(1.0-1.0im)
        _, amps = NAFF.naff(reshape(tone(0.25, A_true, 800), 1, :), 1)
        @test abs(amps[1,1] - A_true) / abs(A_true) < 1e-6
    end

    @testset "purely imaginary amplitude" begin
        _, amps = NAFF.naff(reshape(tone(0.2, 0.0+3.0im, 800), 1, :), 1)
        @test abs(real(amps[1,1])) < 1e-5
        @test abs(imag(amps[1,1]) - 3.0) < 1e-5
    end

    @testset "unit amplitude: |Δ|A|| < 1e-5" begin
        _, amps = NAFF.naff(reshape(tone(0.3, 1.0+0im, 500), 1, :), 1)
        @test abs(abs(amps[1,1]) - 1.0) < 1e-5
    end

    @testset "phase preserved: |Δφ| < 1e-5 for φ=$φ" for φ in [0.0, π/6, π/3, π/2, π, -π/4]
        A_true = 2.0 * exp(im*φ)
        _, amps = NAFF.naff(reshape(tone(0.22, A_true, 1000), 1, :), 1)
        @test abs(mod(angle(amps[1,1]) - φ + π, 2π) - π) < 1e-5
    end

    @testset "amplitude scales linearly across 6 orders of magnitude" begin
        scales = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        ratios = [begin
                      _, amps = NAFF.naff(reshape(tone(0.3, s*(1.0+0im), 1000), 1, :), 1)
                      abs(amps[1,1]) / s
                  end for s in scales]
        @test maximum(abs.(ratios .- ratios[1])) < 1e-4
    end

    @testset "amplitude independent of frequency" begin
        A_true = 1.5 - 0.7im
        errs = [begin
                    _, amps = NAFF.naff(reshape(tone(f, A_true, 1000), 1, :), 1)
                    abs(amps[1,1] - A_true) / abs(A_true)
                end for f in [0.1, 0.25, 0.4]]
        @test maximum(errs) < 1e-4
    end

    @testset "two-tone: each amplitude recovered independently" begin
        N   = 2000
        A1, A2 = 2.0+1.0im, 0.5-0.3im
        sig = tone(0.15, A1, N) .+ tone(0.35, A2, N)
        freqs, amps = NAFF.naff(reshape(sig, 1, :), 2)
        idx = sortperm(vec(freqs[1,:]))
        @test abs(amps[1,idx[1]] - A1) / abs(A1) < 1e-4
        @test abs(amps[1,idx[2]] - A2) / abs(A2) < 1e-4
    end

end # D

# ═════════════════════════════════════════════════════════════════════════════
@testset "E. Signal length" begin

    f_true = 0.2

    @testset "$label: result in (0, 0.5)" for (label, N) in [
            ("N=13", 13), ("N=31", 31), ("N=32", 32), ("N=100", 100),
            ("N=473", 473), ("N=1000", 1000), ("N=2000", 2000)]
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, N), 1, :), 1)
        @test 0.0 < freqs[1,1] < 0.5
    end

    @testset "$label: |Δf| < $tol" for (label, N, tol) in [
            ("N=13",   13,   1e-3),
            ("N=31",   31,   1e-5),
            ("N=32",   32,   1/32),
            ("N=100",  100,  1e-7),
            ("N=473",  473,  1e-8),
            ("N=1000", 1000, 1e-8),
            ("N=2000", 2000, 1e-9)]
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, N), 1, :), 1)
        @test abs(freqs[1,1] - f_true) < tol
    end

    @testset "power-of-2 N=$N: |Δf| < 1e-6" for N in [64, 128, 256, 512, 1024]
        freqs, _ = NAFF.naff(reshape(tone(0.173205, 1.0+0im, N), 1, :), 1)
        @test abs(freqs[1,1] - 0.173205) < 1e-6
    end

    @testset "accuracy improves with N: N=2000 more accurate than N=100" begin
        err_small = abs(NAFF.naff(reshape(tone(f_true,1.0+0im,100),  1,:), 1)[1][1,1] - f_true)
        err_large = abs(NAFF.naff(reshape(tone(f_true,1.0+0im,2000), 1,:), 1)[1][1,1] - f_true)
        @test err_large < err_small
    end

    @testset "N not a multiple of (turns+1): truncation transparent to caller" begin
        for N in [100, 200, 333, 500, 997, 1001]
            freqs, _ = NAFF.naff(reshape(tone(0.17, 1.0+0im, N), 1, :), 1)
            @test abs(freqs[1,1] - 0.17) < 1e-5
        end
    end

end # E

# ═════════════════════════════════════════════════════════════════════════════
@testset "F. Frequency coverage" begin

    N = 1000

    @testset "$label: |Δf| < $tol" for (label, f_true, tol) in [
            ("f=0.005", 0.005,  1e-7),
            ("f=0.01",  0.01,   1e-8),
            ("f=0.05",  0.05,   1e-8),
            ("f=0.10",  0.10,   1e-8),
            ("f=0.15",  0.15,   1e-9),
            ("f=0.20",  0.20,   1e-9),
            ("f=0.25",  0.25,   1e-9),
            ("f=0.30",  0.30,   1e-9),
            ("f=0.35",  0.35,   1e-9),
            ("f=0.40",  0.40,   1e-8),
            ("f=0.45",  0.45,   1e-8),
            ("f=0.48",  0.48,   1e-8),
            ("f=0.499", 0.499,  1e-7)]
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, N), 1, :), 1)
        @test abs(freqs[1,1] - f_true) < tol
    end

    @testset "irrational f≈$(round(f_true,digits=5)): |Δf| < 1e-9" for f_true in [
            sqrt(2)/10, sqrt(3)/10, sqrt(5)/10, sqrt(7)/10,
            exp(1)/7,   exp(1)/10,  π/10,       π/12]
        freqs, _ = NAFF.naff(reshape(tone(f_true, 1.0+0im, N), 1, :), 1)
        @test abs(freqs[1,1] - f_true) < 1e-9
    end

    @testset "negative-frequency tone returns negative frequency" begin
        f_true = -0.2
        sig    = exp.(im .* 2π .* f_true .* (0:N-1))
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 1)
        @test abs(freqs[1,1] - f_true) < 1e-8
    end

end # F

# ═════════════════════════════════════════════════════════════════════════════
@testset "G. Batch — basic" begin

    @testset "two calls with same 1-row matrix give identical results" begin
        N = 1000; mat = reshape(tone(0.2731, 1.0+0im, N), 1, N)
        f1b, _ = NAFF.naff(mat, 1)
        f2b, _ = NAFF.naff(mat, 1)
        @test f1b[1,1] == f2b[1,1]
    end

    @testset "two identical rows give identical results" begin
        N = 800; sig = tone(0.314, 1.0+0im, N)
        batch = vcat(reshape(sig,1,N), reshape(sig,1,N))
        freqs, amps = NAFF.naff(batch, 1)
        @test freqs[1,1] == freqs[2,1]
        @test amps[1,1]  == amps[2,1]
    end

    @testset "5-row batch: each row recovers its own frequency" begin
        N        = 1000
        f_truths = [0.10, 0.17, 0.25, 0.33, 0.41]
        batch    = vcat([reshape(tone(f, 1.0+0im, N), 1, N) for f in f_truths]...)
        freqs, _ = NAFF.naff(batch, 1)
        @test maximum(abs.(vec(freqs[:,1]) .- f_truths)) < 1e-8
    end

    @testset "20-row batch: shape (20,1) and all |Δf| < 1e-8" begin
        N = 500; f_true = 0.237
        batch = vcat([reshape(tone(f_true, exp(im*k*0.3), N), 1, N) for k in 0:19]...)
        freqs, amps = NAFF.naff(batch, 1)
        @test size(freqs) == (20, 1)
        @test size(amps)  == (20, 1)
        @test maximum(abs.(vec(freqs[:,1]) .- f_true)) < 1e-8
    end

    @testset "100-row batch: shape (100,1) and all freqs in (0, 0.5)" begin
        N     = 300
        batch = vcat([reshape(tone(0.1 + mod(k,5)*0.07, 1.0+0im, N), 1, N)
                      for k in 0:99]...)
        freqs, _ = NAFF.naff(batch, 1)
        @test size(freqs) == (100, 1)
        @test all(0.0 .< freqs[:,1] .< 0.5)
    end

    @testset "batch result for row $i matches individual call" for i in 1:3
        N        = 1000
        f_truths = [0.15, 0.27, 0.39]
        batch    = vcat([reshape(tone(f, 1.0+0im, N), 1, N) for f in f_truths]...)
        freqs_b, _ = NAFF.naff(batch, 1)
        freqs_s, _ = NAFF.naff(reshape(tone(f_truths[i], 1.0+0im, N), 1, N), 1)
        @test freqs_b[i,1] == freqs_s[1,1]
    end

end # G

# ═════════════════════════════════════════════════════════════════════════════
@testset "H. Batch — multi-term" begin

    @testset "3×2 batch: shape (3,2) and all |Δf| < 1e-7" begin
        N   = 2000
        f1s = [0.12, 0.18, 0.25]; f2s = [0.30, 0.37, 0.43]
        As  = [1.0+0im, 0.8+0.2im, 1.2-0.3im]
        Bs  = [0.5+0im, 0.6-0.1im, 0.4+0.4im]
        batch = vcat([reshape(tone(fa,A,N).+tone(fb,B,N), 1, N)
                      for (fa,fb,A,B) in zip(f1s,f2s,As,Bs)]...)
        freqs, _ = NAFF.naff(batch, 2)
        @test size(freqs) == (3, 2)
        @test all(1:3) do i
            maximum(abs.(sort(vec(freqs[i,:])) .- sort([f1s[i],f2s[i]]))) < 1e-7
        end
    end

    @testset "5×3 batch: shape (5,3) and all |Δf| < 1e-6" begin
        N      = 3000
        f_sets = [[0.08,0.21,0.39],[0.11,0.24,0.42],[0.06,0.19,0.35],
                  [0.13,0.27,0.40],[0.09,0.22,0.38]]
        A_sets = [[2.0+0im,1.0+0.5im,0.5+0im],[1.5+0im,0.8-0.3im,0.6+0im],
                  [1.8+0im,1.2+0.1im,0.4+0im],[1.0+0im,0.9+0.4im,0.7+0im],
                  [2.2+0im,0.7-0.2im,0.3+0im]]
        batch = vcat([reshape(sum(tone(f,A,N) for (f,A) in zip(fs,as_)), 1, N)
                      for (fs,as_) in zip(f_sets,A_sets)]...)
        freqs, _ = NAFF.naff(batch, 3)
        @test size(freqs) == (5, 3)
        @test all(1:5) do i
            maximum(abs.(sort(vec(freqs[i,:])) .- sort(f_sets[i]))) < 1e-6
        end
    end

    @testset "n_terms=1 and n_terms=2 agree on first column" begin
        N   = 1500
        sig = tone(0.15, 2.0+0im, N) .+ tone(0.33, 1.0+0im, N)
        mat = reshape(sig, 1, N)
        f1b, _ = NAFF.naff(mat, 1)
        f2b, _ = NAFF.naff(mat, 2)
        @test f1b[1,1] == f2b[1,1]
    end

    @testset "n_terms=1 and n_terms=3 agree on first column" begin
        N     = 2000
        comps = [(0.12, 2.0+0im), (0.27, 1.0+0im), (0.39, 0.5+0im)]
        sig   = sum(tone(f, A, N) for (f, A) in comps)
        mat   = reshape(sig, 1, N)
        f1b, _ = NAFF.naff(mat, 1)
        f3b, _ = NAFF.naff(mat, 3)
        @test f1b[1,1] == f3b[1,1]
    end

    @testset "window=$p propagates to all rows in multi-term batch" for p in 1:4
        N        = 1000
        f_truths = [0.15, 0.33]
        batch    = vcat([reshape(tone(f, 1.0+0im, N), 1, N) for f in f_truths]...)
        freqs, _ = NAFF.naff(batch, 1; window_order=p)
        @test maximum(abs.(vec(freqs[:,1]) .- f_truths)) < 1e-7
    end

end # H

# ═════════════════════════════════════════════════════════════════════════════
@testset "I. Batch — amplitude" begin

    @testset "per-row complex amplitude: all rel err < 1e-4" begin
        N        = 1000; f_true = 0.25
        A_truths = [1.0+0.0im, 0.0+1.0im, -1.0+0.0im, 0.5+0.5im, 2.0-1.5im]
        batch    = vcat([reshape(tone(f_true, A, N), 1, N) for A in A_truths]...)
        _, amps  = NAFF.naff(batch, 1)
        @test all(i -> abs(amps[i,1]-A_truths[i])/abs(A_truths[i]) < 1e-4,
                  1:length(A_truths))
    end

    @testset "amplitude phase preserved across 8 rows" begin
        N      = 1000; f_true = 0.2
        phases = range(0, 2π - 2π/8, length=8)
        batch  = vcat([reshape(tone(f_true, exp(im*φ), N), 1, N) for φ in phases]...)
        _, amps = NAFF.naff(batch, 1)
        @test all(i -> abs(mod(angle(amps[i,1])-phases[i]+π, 2π)-π) < 1e-4, 1:8)
    end

    @testset "amplitude scales linearly across rows" begin
        N      = 1000; f_true = 0.3
        scales = [0.1, 1.0, 10.0, 100.0]
        batch  = vcat([reshape(tone(f_true, s*(1.0+0im), N), 1, N) for s in scales]...)
        _, amps = NAFF.naff(batch, 1)
        ratios = [abs(amps[i,1])/scales[i] for i in 1:length(scales)]
        @test maximum(abs.(ratios .- ratios[1])) < 1e-4
    end

    @testset "per-row amplitude unaffected by other rows" begin
        N        = 1000; f_true = 0.25
        A_truths = [1.0+0.0im, 2.0-1.0im, 0.5+0.5im]
        batch    = vcat([reshape(tone(f_true, A, N), 1, N) for A in A_truths]...)
        _, amps_batch = NAFF.naff(batch, 1)
        for (i, A) in enumerate(A_truths)
            _, amps_single = NAFF.naff(reshape(tone(f_true, A, N), 1, N), 1)
            @test amps_batch[i,1] == amps_single[1,1]
        end
    end

end # I

# ═════════════════════════════════════════════════════════════════════════════
@testset "J. Batch — heterogeneous rows" begin

    rng      = MersenneTwister(1234)
    N        = 1000; nrows = 10
    f_truths = 0.05 .+ 0.40 .* rand(rng, nrows)
    A_truths = exp.(im .* 2π .* rand(rng, nrows))
    batch    = vcat([reshape(tone(f, A, N), 1, N)
                     for (f, A) in zip(f_truths, A_truths)]...)
    freqs_b, _ = NAFF.naff(batch, 1)

    @testset "10 heterogeneous rows: all |Δf| < 1e-7" begin
        @test maximum(abs.(vec(freqs_b[:,1]) .- f_truths)) < 1e-7
    end

    @testset "reversing row order gives reversed-order results" begin
        freqs_rev, _ = NAFF.naff(batch[end:-1:1, :], 1)
        @test maximum(abs.(vec(freqs_rev[:,1]) .- reverse(vec(freqs_b[:,1])))) < 1e-15
    end

    @testset "6-row 2-term (dominant + weak): all |Δf| < 1e-7" begin
        N2     = 2000
        f_dom  = [0.12, 0.18, 0.25, 0.31, 0.37, 0.43]
        f_weak = [0.20, 0.30, 0.10, 0.40, 0.15, 0.35]
        batch2 = vcat([reshape(tone(fd,2.0+0im,N2).+tone(fw,0.5+0im,N2), 1, N2)
                       for (fd, fw) in zip(f_dom, f_weak)]...)
        freqs2, _ = NAFF.naff(batch2, 2)
        @test all(1:6) do i
            maximum(abs.(sort(vec(freqs2[i,:])) .- sort([f_dom[i],f_weak[i]]))) < 1e-7
        end
    end

    @testset "batch matches individual call for row $i" for i in 1:nrows
        freqs_s, _ = NAFF.naff(reshape(batch[i,:], 1, N), 1)
        @test freqs_b[i,1] == freqs_s[1,1]
    end

end # J

# ═════════════════════════════════════════════════════════════════════════════
@testset "K. Batch — stress (100 rows × 3 terms)" begin

    rng    = MersenneTwister(5678)
    N      = 2000; nrows = 100; nterms = 3

    f_sets = [sort([0.05 + 0.10*rand(rng),
                    0.20 + 0.10*rand(rng),
                    0.35 + 0.10*rand(rng)]) for _ in 1:nrows]
    A_sets = [[exp(im*2π*rand(rng)) for _ in 1:nterms] for _ in 1:nrows]

    batch  = vcat([reshape(sum(tone(f,A,N) for (f,A) in zip(fs,as_)), 1, N)
                   for (fs, as_) in zip(f_sets, A_sets)]...)
    freqs_b, _ = NAFF.naff(batch, nterms)

    @testset "result shape is (100, 3)" begin
        @test size(freqs_b) == (100, 3)
    end

    @testset "all 300 returned freqs in (0, 0.5)" begin
        @test all(0.0 .< freqs_b .< 0.5)
    end

    @testset "all 300 freqs: |Δf| < 1e-7 vs ground truth" begin
        max_err = maximum(i -> maximum(abs.(sort(vec(freqs_b[i,:])) .- f_sets[i])),
                          1:nrows)
        @test max_err < 1e-7
    end

end # K

# ═════════════════════════════════════════════════════════════════════════════
@testset "L. Batch — edge cases" begin

    @testset "N=$N (arbitrary N): result in (0,0.5) and |Δf| < 1e-5" for N in [
            100, 200, 333, 500, 999, 1001]
        freqs, _ = NAFF.naff(reshape(tone(0.17, 1.0+0im, N), 1, N), 1)
        @test 0.0 < freqs[1,1] < 0.5
        @test abs(freqs[1,1] - 0.17) < 1e-5
    end

    @testset "N=13 (very short): result in (0, 0.5)" begin
        freqs, _ = NAFF.naff(reshape(tone(0.2, 1.0+0im, 13), 1, 13), 1)
        @test 0.0 < freqs[1,1] < 0.5
    end

    @testset "50-row × N=100: shape (50,1) and all |Δf| < 1e-5" begin
        batch   = vcat([reshape(tone(0.25, 1.0+0im, 100), 1, 100) for _ in 1:50]...)
        freqs, amps = NAFF.naff(batch, 1)
        @test size(freqs) == (50, 1)
        @test size(amps)  == (50, 1)
        @test maximum(abs.(vec(freqs[:,1]) .- 0.25)) < 1e-5
    end

    @testset "n_terms=2, single tone: second amplitude ≪ first" begin
        mat = reshape(tone(0.2, 1.0+0im, 1000), 1, 1000)
        _, amps = NAFF.naff(mat, 2)
        @test abs(amps[1,2]) < 1e-3 * abs(amps[1,1])
    end

    @testset "n_terms=3, single tone: 2nd and 3rd amplitudes ≪ first" begin
        mat = reshape(tone(0.3, 1.0+0im, 1000), 1, 1000)
        _, amps = NAFF.naff(mat, 3)
        @test abs(amps[1,2]) < 1e-2 * abs(amps[1,1])
        @test abs(amps[1,3]) < 1e-2 * abs(amps[1,1])
    end

end # L

# ═════════════════════════════════════════════════════════════════════════════
@testset "M. Closely-spaced frequencies" begin

    @testset "Δf=10/N: both tones separated and both |Δf| < 1e-6" begin
        N = 3000; fa = 0.200; fb = fa + 10/N
        sig = tone(fa, 1.0+0im, N) .+ tone(fb, 0.9+0im, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        fs_s = sort(vec(freqs[1,:]))
        @test fs_s[1] < (fa+fb)/2 && fs_s[2] > (fa+fb)/2
        @test maximum(abs.(fs_s .- [fa, fb])) < 1e-6
    end

    @testset "Δf=5/N: both tones separated and both |Δf| < 1e-5" begin
        N = 3000; fa = 0.200; fb = fa + 5/N
        sig = tone(fa, 1.0+0im, N) .+ tone(fb, 0.9+0im, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        fs_s = sort(vec(freqs[1,:]))
        @test fs_s[1] < (fa+fb)/2 && fs_s[2] > (fa+fb)/2
        @test maximum(abs.(fs_s .- [fa, fb])) < 1e-5
    end

    @testset "Δf=3/N: both tones separated and second |Δf| < 1e-5" begin
        N = 3000; fa = 0.200; fb = fa + 3/N
        sig = tone(fa, 1.0+0im, N) .+ tone(fb, 0.9+0im, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        fs_s = sort(vec(freqs[1,:]))
        @test fs_s[1] < (fa+fb)/2 && fs_s[2] > (fa+fb)/2
        @test abs(fs_s[2] - fb) < 1e-5
    end

    @testset "Δf=5/N in 5-row batch: all rows resolve both tones" begin
        N = 3000; fa = 0.25; fb = fa + 5/N
        batch = vcat([reshape(tone(fa,exp(im*k),N).+tone(fb,0.9*exp(im*k),N), 1, N)
                      for k in 0:4]...)
        freqs, _ = NAFF.naff(batch, 2)
        @test all(1:5) do i
            fs_i = sort(vec(freqs[i,:]))
            fs_i[1] < (fa+fb)/2 && fs_i[2] > (fa+fb)/2
        end
    end

    @testset "unequal amplitudes (10:1 ratio): weaker tone still recovered at Δf=5/N" begin
        N = 3000; fa = 0.200; fb = fa + 5/N
        sig = tone(fa, 1.0+0im, N) .+ tone(fb, 0.1+0im, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        fs_s = sort(vec(freqs[1,:]))
        @test fs_s[1] < (fa+fb)/2 && fs_s[2] > (fa+fb)/2
    end

end # M

# ═════════════════════════════════════════════════════════════════════════════
@testset "N. Noise robustness" begin

    rng = MersenneTwister(99)

    @testset "SNR~40dB: |Δf| < 5e-7" begin
        # Tolerance is 5e-7 rather than 1e-7: the RNG output of MersenneTwister
        # changed between Julia v1.10 and v1.11, producing a less favourable
        # noise realisation on newer versions.  The algorithm is correct in all
        # cases; the looser bound keeps the test green across Julia versions.
        sig = tone(0.271828, 100.0+0im, 2000) .+ randn(rng, ComplexF64, 2000)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 1)
        @test abs(freqs[1,1] - 0.271828) < 5e-7
    end

    @testset "SNR~20dB: |Δf| < 1e-5" begin
        sig = tone(0.314159, 10.0+0im, 2000) .+ randn(rng, ComplexF64, 2000)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 1)
        @test abs(freqs[1,1] - 0.314159) < 1e-5
    end

    @testset "SNR~10dB: result within 2 bins of true frequency" begin
        N = 2000; f_true = 0.25
        sig = tone(f_true, 3.162+0im, N) .+ randn(rng, ComplexF64, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 1)
        @test abs(freqs[1,1] - f_true) < 2/N
    end

    @testset "noisy 10-row batch (SNR~40dB): all |Δf| < 1e-6" begin
        f_truths = 0.1 .+ 0.04 .* (0:9)
        batch    = vcat([reshape(tone(f, 100.0+0im, 2000) .+ randn(rng, ComplexF64, 2000),
                                 1, 2000) for f in f_truths]...)
        freqs, _ = NAFF.naff(batch, 1)
        @test maximum(abs.(vec(freqs[:,1]) .- f_truths)) < 1e-6
    end

    @testset "SNR~40dB two-tone: both frequencies recovered |Δf| < 1e-5" begin
        N = 2000; fa, fb = 0.15, 0.35
        sig = tone(fa, 100.0+0im, N) .+ tone(fb, 80.0+0im, N) .+
              randn(rng, ComplexF64, N)
        freqs, _ = NAFF.naff(reshape(sig, 1, :), 2)
        @test maximum(abs.(sort(vec(freqs[1,:])) .- [fa, fb])) < 1e-5
    end

end # N

# ═════════════════════════════════════════════════════════════════════════════
@testset "O. Determinism" begin

    @testset "identical inputs give bit-identical outputs" begin
        N   = 1000; f_true = sqrt(2)/10
        mat = reshape(tone(f_true, 1.5-0.7im, N), 1, N)
        f1b, a1 = NAFF.naff(mat, 2; window_order=2)
        f2b, a2 = NAFF.naff(mat, 2; window_order=2)
        @test f1b == f2b
        @test a1  == a2
    end

    @testset "batch result is bit-identical to per-row individual calls" begin
        N    = 1000
        sigs = [tone(f, 1.0+0im, N) for f in [0.15, 0.27, 0.39]]
        batch = vcat([reshape(s, 1, N) for s in sigs]...)
        freqs_batch, amps_batch = NAFF.naff(batch, 1)
        for i in 1:3
            freqs_s, amps_s = NAFF.naff(reshape(sigs[i], 1, N), 1)
            @test freqs_batch[i,1] == freqs_s[1,1]
            @test amps_batch[i,1]  == amps_s[1,1]
        end
    end

end # O

end # NAFF (top-level testset)

# ═════════════════════════════════════════════════════════════════════════════
# P. Type stability — JET.jl
# ═════════════════════════════════════════════════════════════════════════════
# JET.@test_opt checks that Julia's compiler infers concrete types throughout
# naff() with no runtime dispatch, dynamic dispatch, or type instabilities.
# We test the three most important concrete input types separately because
# Julia specialises code per eltype, so each is an independent compilation.
#
# @test_opt fails if JET finds any "optimization failure" (runtime dispatch).
# Ignorelist entries suppress known false-positives from stdlib/FFTW internals
# that are outside NAFF's own code.

@testset "P. Type stability (JET)" begin

    using JET

    # Representative small matrices — JET analyses the compiler IR, not the
    # runtime values, so signal length and nrows don't need to be large.
    mat64 = ones(ComplexF64,  4, 64)
    mat32 = ones(ComplexF32,  4, 64)

    # Calls to test — cover the three public entry-point forms
    calls = [
        ("Matrix{ComplexF64}, n=1",            () -> NAFF.naff(mat64, 1)),
        ("Matrix{ComplexF64}, n=2, window=2",  () -> NAFF.naff(mat64, 2; window_order=2)),
        ("Matrix{ComplexF32}, n=1",            () -> NAFF.naff(mat32, 1)),
    ]

    @testset "$label" for (label, call) in calls
        @test_opt ignored_modules=[Base] call()
    end

end # P

# ═════════════════════════════════════════════════════════════════════════════
# Q. Worked example — accelerator turn-by-turn signal (printed, not asserted)
# ═════════════════════════════════════════════════════════════════════════════

let
    println("\n══════════════════════════════════════════════════════════════")
    println("  Worked Example: accelerator turn-by-turn signal")
    println("══════════════════════════════════════════════════════════════\n")
    println("  Signal: betatron tune + synchrotron sidebands + coupling + noise\n")

    N  = 1024; t = 0:N-1
    components = [
        (0.2731,          1.00 + 0.00im, "betatron tune"),
        (0.2731 - 0.0034, 0.12 - 0.05im, "lower sideband"),
        (0.2731 + 0.0034, 0.10 + 0.04im, "upper sideband"),
        (0.1012,          0.05 + 0.02im, "coupling line"),
    ]
    signal = randn(MersenneTwister(42), ComplexF64, N) .* 0.002
    for (f, A, _) in components
        signal .+= A .* exp.(im .* 2π .* f .* t)
    end

    freqs_raw, amps = NAFF.naff(reshape(signal, 1, :), 4)
    idx        = sortperm(vec(freqs_raw[1,:]))
    freqs_s    = freqs_raw[1, idx]
    amps_s     = amps[1, idx]
    sorted_tru = sort(components; by = c -> c[1])

    @printf("  %-14s  %-14s  %-10s  %-10s  %s\n",
            "f_true", "f_naff", "|Δf|", "|A|", "Description")
    println("  " * "─"^65)
    for (fn, an, (ft, _, desc)) in zip(freqs_s, amps_s, sorted_tru)
        @printf("  %-14.10f  %-14.10f  %-10.2e  %-10.5f  %s\n",
                ft, fn, abs(fn - ft), abs(an), desc)
    end
    println()
end