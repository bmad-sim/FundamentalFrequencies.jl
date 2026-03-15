#=

This entire file was written by iterating with Claude.

=#
 
 
"""
    brentb(g, a, b; tol, maxiter, check_every) -> (xmax, gmax, converged)
 
Batched Brent's method for maximization using only broadcast/vectorized
operations. Works on CPU (Array) or GPU (CuArray/MtlArray).
 
Direct mechanical translation of NR's scalar `brent` (ch 10.2),
negating comparison directions to maximize insteßßd of minimize.

Find the maximum of vectorized `g` over `[a[i], b[i]]` for each element i.
To minimize, pass `x -> -g(x)` and negate `gmax`.
 
`check_every`: how often to check `all(converged)` for early exit.
  - 1  = check every iteration (max CPU←→GPU syncs, best for cheap g)
  - N  = check every N iterations (fewer syncs, better for expensive g on GPU)
  - 0  = never check (always run maxiter, no sync overhead)
"""
@generated function brentb(
    g,
    a::AbstractArray{T,1},
    b::AbstractArray{T,1};
    tol         = T(1e-8),
    maxiter     = 100,
    check_every = 1,
) where {T}
    # This weirdness is for Metal compatibility (only Float32)
    CGOLD = T(0.3819660112501051)
    ZEPS  = T(1e-10)
    return quote
        @assert length(a) == length(b)
 
        a  = copy(a)
        b  = copy(b)
 
        x  = @. a + $CGOLD * (b - a)
        gx = g(x)
        w  = copy(x);  gw = copy(gx)
        v  = copy(x);  gv = copy(gx)
 
        # Pre-allocate all work arrays outside the loop to avoid
        # per-iteration allocation and extra GPU kernel launches.
        d         = similar(a); d .= zero(T)
        e         = similar(a); e .= zero(T)   # NR: start at 0, not b-a
        converged = similar(a, Bool); converged .= false
 
        # Scratch arrays — reused every iteration
        mid       = similar(a)
        tol1      = similar(a)
        tol2      = similar(a)
        r_saved   = similar(a)
        r_arr     = similar(a)
        q_arr     = similar(a)
        p_arr     = similar(a)
        p_neg     = similar(a)
        d_para    = similar(a)
        gs_left   = similar(a)
        gs_right  = similar(a)
        gs_e      = similar(a)
        d_golden  = similar(a)
        d_chosen  = similar(a)
        u_para    = similar(a)
        tol1_neg  = similar(a)
        d_edge    = similar(a)
        d_min     = similar(a)
        u         = similar(a)
        u_eval    = similar(a)
        gu_raw    = similar(a)
        gu        = similar(a)
        bound     = similar(a)
        # Bool scratch
        qpos      = similar(a, Bool)
        try_para  = similar(a, Bool)
        p_small   = similar(a, Bool)
        left_ok   = similar(a, Bool)
        right_ok  = similar(a, Bool)
        para_ok   = similar(a, Bool)
        x_right   = similar(a, Bool)
        near_a    = similar(a, Bool)
        near_b    = similar(a, Bool)
        x_lt_mid  = similar(a, Bool)
        d_big     = similar(a, Bool)
        d_pos     = similar(a, Bool)
        use_edge  = similar(a, Bool)
        u_lt_x    = similar(a, Bool)
        gu_best   = similar(a, Bool)
        update_a  = similar(a, Bool)
        is_best   = similar(a, Bool)
        is_2nd    = similar(a, Bool)
        is_3rd    = similar(a, Bool)
 
        for iter in 1:maxiter
 
            @. mid   = (a + b) / 2
            @. tol1  = tol * abs(x) + $ZEPS
            @. tol2  = 2 * tol1
 
            # Convergence check — sync to CPU every check_every iterations
            if check_every > 0 && mod(iter, check_every) == 0
                @. converged = abs(x - mid) <= tol2 - (b - a) / 2
                all(converged) && break
            end
 
            # ── Parabolic interpolation ───────────────────────────────────────
            @. r_arr = (x - w) * (gx - gv)
            @. q_arr = (x - v) * (gx - gw)
            @. p_arr = (x - v) * q_arr - (x - w) * r_arr
            @. q_arr = 2 * (q_arr - r_arr)
            # flip p so q ends up positive
            @. p_neg = -p_arr
            @. qpos  = q_arr > 0
            @. p_arr = ifelse(qpos, p_neg, p_arr)
            @. q_arr = abs(q_arr)
 
            # r_saved = current e (step from 2 iters ago)
            r_saved .= e
 
            # ── Parabola vs golden section ────────────────────────────────────
            @. try_para  = abs(e) > tol1
            @. p_small   = abs(p_arr) < abs(q_arr) * abs(r_saved / 2)
            @. left_ok   = p_arr > q_arr * (a - x)
            @. right_ok  = p_arr < q_arr * (b - x)
            @. para_ok   = try_para & p_small & left_ok & right_ok
 
            @. d_para    = p_arr / q_arr
 
            @. x_right   = x >= mid
            @. gs_left   = a - x
            @. gs_right  = b - x
            @. gs_e      = ifelse(x_right, gs_left, gs_right)
            @. d_golden  = $CGOLD * gs_e
 
            @. d_chosen  = ifelse(para_ok, d_para, d_golden)
 
            # edge guard: parabolic u too close to bracket endpoints
            @. u_para    = x + d_para
            @. near_a    = (u_para - a) < tol2
            @. near_b    = (b - u_para) < tol2
            @. x_lt_mid  = x < mid
            @. tol1_neg  = -tol1
            @. d_edge    = ifelse(x_lt_mid, tol1, tol1_neg)
            @. use_edge  = para_ok & (near_a | near_b)
            @. d_chosen  = ifelse(use_edge, d_edge, d_chosen)
 
            # NR: e ← d_chosen (the step about to be taken, before clamping)
            # This is what NR means by "e = d" — the unchosen step size used
            # two iterations hence to gate parabolic acceptance.
            @. e         = d_chosen
 
            # enforce minimum step of tol1
            @. d_big     = abs(d_chosen) >= tol1
            @. d_pos     = d_chosen >= 0
            @. d_min     = ifelse(d_pos, tol1, tol1_neg)
            @. d         = ifelse(d_big, d_chosen, d_min)
 
            @. u         = x + d
 
            # Mask converged lanes so g isn't called on stale/redundant points
            @. u_eval    = ifelse(converged, x, u)
            gu_raw      .= g(u_eval)
            @. gu        = ifelse(converged, gx, gu_raw)
 
            # ── Update bracket and best points (gate on !converged) ─────────────
            # NR rule (maximization, strict >):
            #   gu>gx,  u<x  → b=x;  gu>gx,  u>=x → a=x
            #   gu<=gx, u<x  → a=u;  gu<=gx, u>=x → b=u
            # update_a = u_lt_x XOR gu_best  (truth table verifies all 4 cases)
            @. u_lt_x    = u < x
            @. gu_best   = gu > gx
            @. bound     = ifelse(gu_best, x, u)
            @. update_a  = u_lt_x ⊻ gu_best
            @. a         = ifelse(converged, a, ifelse(update_a, bound, a))
            @. b         = ifelse(converged, b, ifelse(update_a, b,     bound))
 
            # ── Update best points x, w, v (gate on !converged) ─────────────────
            @. is_best   = gu > gx   # strict: ties leave x unchanged (NR: fu < fx)
            @. v         = ifelse(converged, v,  ifelse(is_best, w,  v))
            @. gv        = ifelse(converged, gv, ifelse(is_best, gw, gv))
            @. w         = ifelse(converged, w,  ifelse(is_best, x,  w))
            @. gw        = ifelse(converged, gw, ifelse(is_best, gx, gw))
            @. x         = ifelse(converged, x,  ifelse(is_best, u,  x))
            @. gx        = ifelse(converged, gx, ifelse(is_best, gu, gx))
 
            # NR: if fu >= fw or w==x: shift w→v, u→w
            @. is_2nd    = !is_best & ((gu >= gw) | (w == x))
            @. v         = ifelse(converged, v,  ifelse(is_2nd, w,  v))
            @. gv        = ifelse(converged, gv, ifelse(is_2nd, gw, gv))
            @. w         = ifelse(converged, w,  ifelse(is_2nd, u,  w))
            @. gw        = ifelse(converged, gw, ifelse(is_2nd, gu, gw))
 
            # NR: if fu >= fv or v==x or v==w: u→v
            @. is_3rd    = !is_best & !is_2nd & ((gu >= gv) | (v == x) | (v == w))
            @. v         = ifelse(converged, v,  ifelse(is_3rd, u,  v))
            @. gv        = ifelse(converged, gv, ifelse(is_3rd, gu, gv))
 
        end
 
        # Final convergence check (catches the case where we exited via maxiter)
        @. mid       = (a + b) / 2
        @. tol1      = tol * abs(x) + $ZEPS
        @. tol2      = 2 * tol1
        @. converged = abs(x - mid) <= tol2 - (b - a) / 2
 
        return x, gx, converged
    end
end