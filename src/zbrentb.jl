#=

This batched zbrent was written entirely using Claude.

=#
"""
    zbrentb(f, a, b; tol, maxiter) -> (roots, converged)

Find roots of the vectorized function `f` for each bracket `[a[i], b[i]]`.

Convergence uses the same adaptive tolerance as Numerical Recipes' zbrent:

    tol1[i] = 2*eps(T)*|b[i]| + tol/2

which blends an absolute floor (`tol`) with a relative term that scales with
the current best estimate `b`. This naturally handles Float32 large-magnitude
roots where the absolute ULP spacing exceeds any fixed tolerance.

The loop performs one `all()` reduction per iteration to enable early exit
once every element has converged. This costs one GPU sync per iteration but
can save significant work when convergence rates vary across the batch.

# Arguments
- `f`       - vectorized callable, `f(x)` accepts and returns an array of the
              same type/backend as `a`.
- `a`, `b`  - 1-D AbstractArrays defining the left/right brackets.
              Can be `Array` (CPU) or `CuArray`/`MtlArray` (GPU).
- `tol`     - absolute component of the convergence tolerance.
- `maxiter` - maximum Brent iterations.

# Returns
- `roots`     - array of root estimates, same type/backend as `a`.
- `converged` - Bool array indicating per-element convergence.
"""
function zbrentb(
    f,
    a::AbstractArray{T,1},
    b::AbstractArray{T,1};
    tol     = T(max(1e-8, eps(T) * 100)),
    maxiter = 100,
) where {T}

    @assert length(a) == length(b)

    a  = copy(a)
    b  = copy(b)
    fa = f(a)
    fb = f(b)

    # Swap so |f(b)| <= |f(a)| everywhere
    swap = @. abs(fa) < abs(fb)
    a, b   = _swap(swap, a, b)
    fa, fb = _swap(swap, fa, fb)

    c     = copy(a)
    fc    = copy(fa)
    d     = copy(b)
    mflag     = similar(a, Bool)
    mflag    .= true
    converged = similar(a, Bool)
    converged .= false

    for _ in 1:maxiter

        # zbrent-style adaptive tolerance, recomputed each iteration from
        # the current best estimate b:  tol1 = 2*eps*|b| + tol/2
        tol1 = @. 2 * eps(T) * abs(b) + T(tol) / 2

        # Early exit once every element has converged — one scalar reduction
        # per iteration, which syncs the GPU but saves all remaining iterations
        # when the batch converges before maxiter.
        @. converged = (abs(fb) < T(tol)) | (abs(b - a) < tol1)
        all(converged) && break

        # ── Inverse quadratic interpolation ───────────────────────────────────
        s_iqi = @. (  a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb))  )

        # ── Secant step ───────────────────────────────────────────────────────
        s_sec = @. b - fb * (b - a) / (fb - fa)

        # Use IQI only when all three f-values are distinct, else secant
        use_iqi = @. (fa != fc) & (fb != fc)
        s = @. ifelse(use_iqi, s_iqi, s_sec)

        # ── Brent acceptance conditions: fall back to bisection? ──────────────
        mid = @. (a + b) / T(2)
        lo  = @. min((3a + b) / 4, b)
        hi  = @. max((3a + b) / 4, b)

        c1 = @. (s <= lo) | (s >= hi)                           # outside bracket sub-interval
        c2 = @.  mflag  & (abs(s - b) >= abs(b - c) / 2)       # step too large (bisection was last)
        c3 = @. !mflag  & (abs(s - b) >= abs(c - d) / 2)       # step too large (interp was last)
        c4 = @.  mflag  & (abs(b - c) < tol1)                  # bracket nearly degenerate
        c5 = @. !mflag  & (abs(c - d) < tol1)

        use_bisect = @. c1 | c2 | c3 | c4 | c5
        s          = @. ifelse(use_bisect, mid, s)
        mflag     .= use_bisect

        fs = f(s)

        # ── Update bracket ────────────────────────────────────────────────────
        d  .= c
        c  .= b
        fc .= fb

        # s replaces the endpoint that shares a sign with f(s)
        s_replaces_b = @. (fa * fs) < T(0)
        a  = @. ifelse(s_replaces_b, a,  s)
        fa = @. ifelse(s_replaces_b, fa, fs)
        b  = @. ifelse(s_replaces_b, s,  b)
        fb = @. ifelse(s_replaces_b, fs, fb)

        # Keep |f(b)| <= |f(a)|
        swap = @. abs(fa) < abs(fb)
        a, b   = _swap(swap, a, b)
        fa, fb = _swap(swap, fa, fb)

    end

    return b, converged
end

# ── Utilities ─────────────────────────────────────────────────────────────────

# Swap pairs element-wise where mask is true
function _swap(mask, x, y)
    x_new = @. ifelse(mask, y, x)
    y_new = @. ifelse(mask, x, y)
    return x_new, y_new
end