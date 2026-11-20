# =============================================================================
# saddle_links.jl — which minima does an index-1 saddle separate?
#
# Analysis on a TRUE objective f (ForwardDiff gradients and Hessians), independent
# of any surrogate. The typical question it answers: given the minima and index-1
# saddles of a landscape, which pairs of minima are connected by a mountain pass?
#
# The pieces:
#   newton_critical_point  damped Newton on ∇f = 0 — backtracking on ‖∇f‖∞, so it
#                          converges to saddles and maxima as readily as to minima
#                          (an LM or Nelder–Mead polish can only return minima)
#   descend_to_minimum     backtracking gradient descent on f, confined to a box —
#                          decides which basin a point belongs to
#   morse_data             Morse index and spectrum of the symmetrised AD Hessian
#   saddle_links           from each index-1 saddle, step ±ε along the negative
#                          eigenvector, descend, and name the two minima reached
#
# For a square residual F with f = ‖F‖², the minima of f at value 0 are the zeros
# of F and every other critical point has JᵀF = 0 with F ≠ 0, hence a singular
# Jacobian: the saddles between two zeros lie on the fold hypersurface. A link
# (z_i, s, z_j) with f(s) the barrier is what "the landscape connecting two
# extremals" means operationally.
# =============================================================================

"""
    newton_critical_point(f, x0; lo, hi, iters=80, gtol=1e-10) -> (x, converged, gnorm)

Damped Newton on `∇f = 0`, confined to the box `[lo, hi]`, with backtracking on
`‖∇f‖∞` rather than on `f` so that the target is ANY critical point (saddles
and maxima included). A singular Hessian falls back to a gradient step for that
iteration. `converged` means `‖∇f‖∞ < gtol` (or `< 1e-7` after the last
iteration).
"""
function newton_critical_point(
    f,
    x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real},
    hi::AbstractVector{<:Real},
    iters::Int = 80,
    gtol::Float64 = 1e-10,
    grad = x -> ForwardDiff.gradient(f, x),
    hess = x -> ForwardDiff.hessian(f, x),
)
    x = clamp.(collect(float.(x0)), lo, hi)
    g = grad(x)
    all(isfinite, g) || return (x, false, Inf)
    for _ in 1:iters
        maximum(abs, g) < gtol && return (x, true, maximum(abs, g))
        H = hess(x)
        step = if all(isfinite, H)
            try
                -(LinearAlgebra.Symmetric(0.5 * (H + H')) \ g)
            catch
                -g
            end
        else
            -g
        end
        all(isfinite, step) || return (x, false, maximum(abs, g))
        alpha, accepted, gn = 1.0, false, maximum(abs, g)
        for _ in 1:40
            cand = clamp.(x .+ alpha .* step, lo, hi)
            gc = grad(cand)
            if all(isfinite, gc) && maximum(abs, gc) < gn
                x, g, accepted = cand, gc, true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    gn = maximum(abs, g)
    return (x, gn < max(gtol, 1e-7), gn)
end

"""
    lm_critical_point(f, x0; lo, hi, iters=100, gtol=1e-10, step_max=Inf, mu0=1e-3, grad, hess)
        -> (x, converged, gnorm)

Levenberg–Marquardt on the system `∇f = 0` with the Hessian as its Jacobian:
`step = −(H² + μI)⁻¹ H ∇f`, accepted when `‖∇f‖₂` decreases, `μ` adapted as in
Gauss–Newton/LM. Converges to saddles and maxima like
[`newton_critical_point`](@ref) but is damped: on landscapes where the quadratic
model is poor far from a critical point (a log-compressed residual norm, say)
pure Newton's first step can be orders of magnitude too long and run to the box
faces; LM shortens it. `step_max` caps the step length in the box coordinates.
A gradient that is not finite (a `NaN` from a plateau-aware `grad`) rejects the
candidate step.
"""
function lm_critical_point(
    f,
    x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real},
    hi::AbstractVector{<:Real},
    iters::Int = 100,
    gtol::Float64 = 1e-10,
    step_max::Float64 = Inf,
    mu0::Float64 = 1e-3,
    grad = x -> ForwardDiff.gradient(f, x),
    hess = x -> ForwardDiff.hessian(f, x),
)
    x = clamp.(collect(float.(x0)), lo, hi)
    g = grad(x)
    all(isfinite, g) || return (x, false, Inf)
    μ = mu0
    n = length(x)
    for _ in 1:iters
        gn = maximum(abs, g)
        gn < gtol && return (x, true, gn)
        H = hess(x)
        all(isfinite, H) || return (x, false, gn)
        Hs = LinearAlgebra.Symmetric(0.5 * (H + H'))
        HH = Hs * Hs
        Hg = Hs * g
        g2 = LinearAlgebra.norm(g)
        improved = false
        for _ in 1:14
            step = try
                -((HH + μ * LinearAlgebra.I(n)) \ Hg)
            catch
                μ *= 100
                continue
            end
            all(isfinite, step) || (μ *= 100; continue)
            sn = LinearAlgebra.norm(step)
            sn > step_max && (step .*= step_max / sn)
            cand = clamp.(x .+ step, lo, hi)
            gc = grad(cand)
            if all(isfinite, gc) && LinearAlgebra.norm(gc) < g2
                x, g = cand, gc
                μ = max(μ / 3, 1e-12)
                improved = true
                break
            else
                μ *= 10
            end
        end
        improved || break
    end
    gn = maximum(abs, g)
    return (x, gn < max(gtol, 1e-7), gn)
end

"""
    minmode_saddle(f, x0; lo, hi, iters=300, gtol=1e-10, step_max=0.05, grad, hess)
        -> (x, converged, gnorm, n_newton)

Index-1 saddle search by min-mode following with the exact Hessian (the
eigenvector-following / dimer idea of Henkelman and Jónsson, here with an AD
Hessian instead of a dimer estimate): the step direction is the gradient with
its component along the softest Hessian eigenvector `v` reversed,
`−∇f + 2(∇f·v)v`, so the iterate climbs along the one mode and descends in every
other, and is capped at `step_max` in box units. Once there is exactly one
negative eigenvalue and the Newton step is shorter than `step_max`, a Newton
step is tried and kept when it reduces `‖∇f‖₂` (quadratic endgame). Unlike a
gradient-norm minimiser this does not stall where `H ∇f = 0` with `∇f ≠ 0`. A
non-finite gradient (plateau) halves the step instead of being taken. Returns
the number of accepted Newton steps as `n_newton`.
"""
function minmode_saddle(
    f,
    x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real},
    hi::AbstractVector{<:Real},
    iters::Int = 300,
    gtol::Float64 = 1e-10,
    step_max::Float64 = 0.05,
    grad = x -> ForwardDiff.gradient(f, x),
    hess = x -> ForwardDiff.hessian(f, x),
)
    x = clamp.(collect(float.(x0)), lo, hi)
    g = grad(x)
    all(isfinite, g) || return (x, false, Inf, 0)
    α = step_max
    n_newton = 0
    for _ in 1:iters
        gn = maximum(abs, g)
        gn < gtol && return (x, true, gn, n_newton)
        H = hess(x)
        all(isfinite, H) || return (x, false, gn, n_newton)
        E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(0.5 * (H + H')))
        v = E.vectors[:, 1]
        # Newton endgame: exactly one negative mode and a short Newton step
        if E.values[1] < 0 && (length(E.values) == 1 || E.values[2] > 0)
            dN = -(E.vectors * ((E.vectors' * g) ./ E.values))
            if all(isfinite, dN) && LinearAlgebra.norm(dN) < step_max
                cand = clamp.(x .+ dN, lo, hi)
                gc = grad(cand)
                if all(isfinite, gc) && LinearAlgebra.norm(gc) < LinearAlgebra.norm(g)
                    x, g = cand, gc
                    n_newton += 1
                    continue
                end
            end
        end
        # eigenvector-following step (Cerjan–Miller / Baker): along each eigendirection e_k the
        # step is −(g·e_k)/|λ_k| e_k (descent in the quadratic model), except along the softest
        # mode e_1 where the sign is flipped (climb). Equals Newton when λ_1 < 0 < λ_2. The
        # |λ_k| are floored so a flat valley does not produce an infinite step; the whole step is
        # capped at the current trust radius α. This is what makes the walk usable on strongly
        # anisotropic landscapes, where a fixed-length step is either unstable along the stiff
        # modes or crawls along the soft ones.
        λ = E.values
        lam_floor = max(1e-3 * maximum(abs, λ), 1e-8)
        c = E.vectors' * g
        d = zeros(length(x))
        for k in eachindex(λ)
            s = k == 1 ? +1.0 : -1.0
            d .+= s * (c[k] / max(abs(λ[k]), lam_floor)) .* E.vectors[:, k]
        end
        dn = LinearAlgebra.norm(d)
        dn > 0 || break
        dn > α && (d .*= α / dn)
        cand = clamp.(x .+ d, lo, hi)
        gc = grad(cand)
        if !all(isfinite, gc)
            α = max(α / 2, 1e-6 * step_max)
            continue
        end
        x, g = cand, gc
        α = min(α * 1.2, step_max)
    end
    gn = maximum(abs, g)
    return (x, gn < max(gtol, 1e-7), gn, n_newton)
end

"""
    descend_to_minimum(f, x0; lo, hi, maxit=400, gtol=1e-9) -> x

Backtracking gradient descent on `f`, confined to the box. It deliberately
descends `f` (unlike [`newton_critical_point`](@ref)) so that the returned point
identifies the basin `x0` lies in.
"""
function descend_to_minimum(
    f,
    x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real},
    hi::AbstractVector{<:Real},
    maxit::Int = 400,
    gtol::Float64 = 1e-9,
    grad = x -> ForwardDiff.gradient(f, x),
)
    x = clamp.(collect(float.(x0)), lo, hi)
    fx = f(x)
    for _ in 1:maxit
        g = grad(x)
        all(isfinite, g) || break
        maximum(abs, g) < gtol && break
        step, moved = 1.0, false
        for _ in 1:50
            cand = clamp.(x .- step .* g, lo, hi)
            fc = f(cand)
            if fc < fx
                x, fx, moved = cand, fc, true
                break
            end
            step *= 0.5
        end
        moved || break
    end
    return x
end

"""
    morse_data(f, x) -> (index, eigenvalues, eigenvectors, asymmetry)

Morse index (number of negative Hessian eigenvalues) and the spectrum of the
TRUE Hessian at `x`. The AD Hessian is symmetrised before the eigendecomposition
and its relative asymmetry is returned: a silently asymmetric Hessian would
corrupt the index.
"""
function morse_data(f, x::AbstractVector{<:Real}; hess = x -> ForwardDiff.hessian(f, x))
    H = hess(collect(float.(x)))
    asym = maximum(abs, H - H') / max(1.0, maximum(abs, H))
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(0.5 * (H + H')))
    return (count(<(0), E.values), E.values, E.vectors, asym)
end

"""
    nearest_point_index(x, pts; tol=1e-3) -> Int

Index of the entry of `pts` nearest `x` in the max norm, or `0` when the nearest
is further than `tol`.
"""
function nearest_point_index(
    x::AbstractVector{<:Real},
    pts::AbstractVector;
    tol::Float64 = 1e-3,
)
    best, bestd = 0, Inf
    for (k, p) in enumerate(pts)
        d = maximum(abs, x .- p)
        if d < bestd
            best, bestd = k, d
        end
    end
    return bestd <= tol ? best : 0
end

"""
    saddle_links(f, saddles, minima; lo, hi, eps_frac=1e-3, match_tol=1e-3, descend=nothing)
        -> Vector{NamedTuple}

Which minima each index-1 saddle separates.

From a saddle, stepping `±ε` along the eigenvector `v` of its single negative
eigenvalue (`ε = eps_frac · min(hi − lo)`) and descending lands in the two basins
the saddle divides. Each entry is `(saddle, min_i, min_j, distinct, endpoints)`:
`min_i`, `min_j` index `minima` (`0` = the descent landed on no supplied minimum,
within `match_tol` in the max norm), `distinct` is true when both are nonzero
and differ, and `endpoints` holds the two landing points.

`saddles` are named tuples carrying at least `theta`, `eigenvalues` and
`eigenvectors` (as from [`morse_data`](@ref)); `minima` carry `theta`. A saddle
with no negative eigenvalue is skipped. `descend` overrides the descent: a
function `x -> landing point`, e.g. gradient descent followed by a problem-specific
polish such as Levenberg–Marquardt on a residual; the default is
[`descend_to_minimum`](@ref) on `f` in the box.
"""
function saddle_links(
    f,
    saddles,
    minima;
    lo::AbstractVector{<:Real},
    hi::AbstractVector{<:Real},
    eps_frac::Float64 = 1e-3,
    match_tol::Float64 = 1e-3,
    descend = nothing,
)
    mins = [collect(float.(m.theta)) for m in minima]
    step = eps_frac * minimum(hi .- lo)
    desc = descend === nothing ? (x -> descend_to_minimum(f, x; lo = lo, hi = hi)) : descend
    links = NamedTuple[]
    for (si, s) in enumerate(saddles)
        k = findfirst(<(0), s.eigenvalues)
        k === nothing && continue
        v = s.eigenvectors[:, k]
        ends = Int[]
        pts = Vector{Float64}[]
        for sgn in (+1.0, -1.0)
            x = clamp.(collect(float.(s.theta)) .+ sgn * step .* v, lo, hi)
            landed = collect(float.(desc(x)))
            push!(pts, landed)
            push!(ends, nearest_point_index(landed, mins; tol = match_tol))
        end
        push!(
            links,
            (
                saddle = si,
                min_i = ends[1],
                min_j = ends[2],
                distinct = (ends[1] != ends[2] && ends[1] != 0 && ends[2] != 0),
                endpoints = pts,
            ),
        )
    end
    return links
end
