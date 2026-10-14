# =============================================================================
# saddle_links.jl — which minima does an index-1 saddle separate?
#
# Analysis on a TRUE objective f (ForwardDiff gradients and Hessians), independent
# of any surrogate. Generalised from `pkg/BioObjectives/src/analysis.jl` (bead
# MARS-SADDLE-LINKS, globopt_merged-o6q5.4), where the same logic answers the GLV
# Gate-6 question; BioObjectives keeps its own copy to stay Globtim-free.
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
function newton_critical_point(f, x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real},
    iters::Int = 80, gtol::Float64 = 1e-10,
    grad = x -> ForwardDiff.gradient(f, x), hess = x -> ForwardDiff.hessian(f, x))
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
    descend_to_minimum(f, x0; lo, hi, maxit=400, gtol=1e-9) -> x

Backtracking gradient descent on `f`, confined to the box. It deliberately
descends `f` (unlike [`newton_critical_point`](@ref)) so that the returned point
identifies the basin `x0` lies in.
"""
function descend_to_minimum(f, x0::AbstractVector{<:Real};
    lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real},
    maxit::Int = 400, gtol::Float64 = 1e-9,
    grad = x -> ForwardDiff.gradient(f, x))
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
function nearest_point_index(x::AbstractVector{<:Real}, pts::AbstractVector; tol::Float64 = 1e-3)
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
function saddle_links(f, saddles, minima;
    lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real},
    eps_frac::Float64 = 1e-3, match_tol::Float64 = 1e-3, descend = nothing)
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
        push!(links, (saddle = si, min_i = ends[1], min_j = ends[2],
            distinct = (ends[1] != ends[2] && ends[1] != 0 && ends[2] != 0),
            endpoints = pts))
    end
    return links
end
