#=

This entire file was written by iterating with Claude.

=#

"""
    brentb(g, a, b; tol, maxiter, check_every) -> (xmax, gmax, converged)

Find the maximum of vectorized `g` over `[a[i], b[i]]` for each element i.
To minimize, pass `x -> -g(x)` and negate `gmax`.

`g` must accept a vector input and return a vector of the same type/backend
(broadcasting-compatible with the input arrays).

`check_every`: iterations between convergence checks (default 1). Higher
values reduce host-sync overhead on GPU for expensive `g`.
Set to 0 to skip mid-loop checks (one final check always runs at return).
"""
@generated function brentb(
    g,
    a::AbstractArray{T,1},
    b::AbstractArray{T,1};
    tol         = T(1e-8),
    maxiter     = 100,
    check_every = 1,
) where {T}
    # This weirdness (generated) is needed for compatibility with Metal
    # which literally cannot compile even if we immediately cast compile 
    # time constant to Float32.
    CGOLD = T(0.3819660112501051)
    return quote
        @assert length(a) == length(b)

        a  = copy(a)
        b  = copy(b)

        x  = @. a + $CGOLD * (b - a)
        gx = g(x)
        w  = copy(x);  gw = copy(gx)
        v  = copy(x);  gv = copy(gx)

        # Algorithm state
        d         = similar(a); d .= zero(T)   # last step taken
        e         = similar(a); e .= zero(T)   # step before last (NR: start at 0)
        converged = similar(a, Bool); converged .= false

        # ── Scratch arrays: 10 float + 5 bool ────────────────────────────────
        #
        # Float scratch — reuse strategy noted inline:
        #   p     : holds r=(x-w)*(gx-gv) first, then p_final
        #   q     : holds q_final = 2*(q_raw - r)
        #   d_para: parabolic step p/q; reused as bound in bracket update
        #   gs_e  : golden-section sub-interval (a-x or b-x)
        #   d_chos: chosen step pre-clamp; also written to e
        #   mid   : (a+b)/2; reused for tol2 expression inline
        #   tol1  : tol*|x|+eps(T); negated inline where -tol1 needed
        #   u     : trial point x+d; reused as u_eval (masked in-place)
        #   gu    : g(u_eval); masked in-place for converged lanes
        #
        # Bool scratch:
        #   para_ok : parabola accepted flag; reused for use_edge mask
        #   x_right : x>=mid; reused as u_lt_x then update_a
        #   is_best : gu>gx; reused as gu_best for bracket update
        #   is_2nd, is_3rd: point bookkeeping
        mid     = similar(a)
        tol1    = similar(a)
        p       = similar(a)
        q       = similar(a)
        d_para  = similar(a)
        gs_e    = similar(a)
        d_chos  = similar(a)
        u       = similar(a)
        gu      = similar(a)
        para_ok = similar(a, Bool)
        x_right = similar(a, Bool)
        is_best = similar(a, Bool)
        is_2nd  = similar(a, Bool)
        is_3rd  = similar(a, Bool)

        for iter in 1:maxiter

            @. mid  = (a + b) / 2
            @. tol1 = tol * abs(x) + $eps(T)

            if check_every > 0 && mod(iter, check_every) == 0
                @. converged = abs(x - mid) <= 2*tol1 - (b - a) / 2
                all(converged) && break
            end

            # ── Parabolic interpolation ───────────────────────────────────────
            # Compute r into p, q_raw into q, then:
            #   q_final = 2*(q_raw - r)        [update q while p=r]
            #   p_final = (x-v)*q_final/2 + r*(w-v)  [algebraic rearrangement,
            #             avoids storing r separately; verified by substitution]
            @. p     = (x - w) * (gx - gv)              # r → p
            @. q     = (x - v) * (gx - gw)              # q_raw → q
            @. q     = 2 * (q - p)                       # q_final (p=r still valid)
            @. p     = (x - v) * q / 2 + p * (w - v)   # p_final

            # Flip sign so q > 0 (NR convention), then abs(q)
            @. p     = ifelse(q > zero(T), -p, p)
            @. q     = abs(q)

            # e holds step from 2 iters ago (saved before overwrite)
            # para_ok temporarily holds the acceptance test result
            @. para_ok = (abs(e) > tol1) &
                         (abs(p) < abs(q) * abs(e / 2)) &
                         (p > q * (a - x)) &
                         (p < q * (b - x))

            # Parabolic step (guarded against q==0)
            @. d_para  = ifelse(q != zero(T), p / q, zero(T))

            # Golden-section step into the larger sub-interval
            @. x_right = x >= mid
            @. gs_e    = ifelse(x_right, a - x, b - x)

            # Chosen step and e update (NR: e ← step before clamping)
            @. d_chos  = ifelse(para_ok, d_para, $CGOLD * gs_e)

            # Edge guard: parabolic u too close to bracket endpoints → use ±tol1
            # Reuse para_ok as the edge-use mask
            @. para_ok = para_ok &
                         (((x + d_para) - a < 2*tol1) | (b - (x + d_para) < 2*tol1))
            @. d_chos  = ifelse(para_ok, ifelse(x < mid, tol1, -tol1), d_chos)

            # e ← chosen step before clamping (NR: e = d, pre-clamp)
            @. e       = d_chos

            # Enforce minimum step of tol1 away from x
            @. d       = ifelse(abs(d_chos) >= tol1, d_chos,
                                ifelse(d_chos >= zero(T), tol1, -tol1))

            # Evaluate g at trial point (mask converged lanes to avoid stale calls)
            @. u       = x + d
            @. u       = ifelse(converged, x, u)   # u_eval in-place
            gu        .= g(u)
            @. gu      = ifelse(converged, gx, gu)  # mask converged result

            # ── Update bracket (gate on !converged) ──────────────────────────
            # NR rule (maximization, strict >):
            #   gu>gx, u<x  → b=x;  gu>gx, u>=x → a=x
            #   gu≤gx, u<x  → a=u;  gu≤gx, u>=x → b=u
            # XOR truth table verifies: update_a = (u<x) ⊻ (gu>gx)
            # is_best reused as gu_best; x_right reused as u_lt_x then update_a
            @. is_best = gu > gx
            @. d_para  = ifelse(is_best, x, u)      # bound (reuse d_para, done above)
            @. x_right = (u < x) ⊻ is_best          # update_a
            @. a       = ifelse(converged, a, ifelse(x_right, d_para, a))
            @. b       = ifelse(converged, b, ifelse(x_right, b, d_para))

            # ── Update best points x, w, v (gate on !converged) ──────────────
            @. v   = ifelse(converged, v,  ifelse(is_best, w,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_best, gw, gv))
            @. w   = ifelse(converged, w,  ifelse(is_best, x,  w))
            @. gw  = ifelse(converged, gw, ifelse(is_best, gx, gw))
            @. x   = ifelse(converged, x,  ifelse(is_best, u,  x))
            @. gx  = ifelse(converged, gx, ifelse(is_best, gu, gx))

            @. is_2nd = !is_best & ((gu >= gw) | (w == x))
            @. v   = ifelse(converged, v,  ifelse(is_2nd, w,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_2nd, gw, gv))
            @. w   = ifelse(converged, w,  ifelse(is_2nd, u,  w))
            @. gw  = ifelse(converged, gw, ifelse(is_2nd, gu, gw))

            @. is_3rd = !is_best & !is_2nd & ((gu >= gv) | (v == x) | (v == w))
            @. v   = ifelse(converged, v,  ifelse(is_3rd, u,  v))
            @. gv  = ifelse(converged, gv, ifelse(is_3rd, gu, gv))

        end

        # Final convergence check
        @. converged = abs(x - (a+b)/2) <= 2*(tol*abs(x)+$eps(T)) - (b-a)/2

        return x, gx, converged
    end
end