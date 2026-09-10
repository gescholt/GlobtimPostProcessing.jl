# Consensus validation of critical points across two independent polynomial fits.
#
# Motivation. Fitting an objective twice from different sample points (or noise
# draws) and keeping the critical points that appear in BOTH is a variance filter: it drops
# fit-specific (sampling-dependent) spurious CPs. On a SLOPPY landscape, however, the objective
# cannot localize a critical point along its flat valley, so the two fits' copies of a true CP spread
# apart — but only along the sloppy Hessian directions. The right output there is not a filtered
# point but the identifiable (stiff) combination plus a sloppy-direction uncertainty TUBE.
#
# `validate_consensus` matches two critical-point sets and, per matched pair, eigendecomposes the
# Hessian into stiff (identifiable) vs sloppy directions and measures how far the two copies disagree
# along each — the stiff disagreement is the validation residual, the sloppy disagreement is the tube.

"""
    ConsensusValidationConfig(; kwargs...)

Configuration for [`validate_consensus`](@ref).

# Fields
- `match_tol::Float64 = 0.1`: two fits' critical points are paired when within this Euclidean
  distance (parameter units).
- `sloppy_ratio::Float64 = 1e-2`: a Hessian eigen-direction is *sloppy* when `|λ| < sloppy_ratio·|λ|ₘₐₓ`;
  the complementary *stiff* directions span the identifiable subspace.
- `gradient_method::Symbol = :forwarddiff`: differentiation backend for gradients/Hessians. Use
  `:finitediff` for objectives that are not ForwardDiff-compatible (e.g. buffered ODE objectives).
- `resolution::Float64 = NaN`: length scale for the confidence `exp(-stiff_spread/resolution)`;
  defaults to `match_tol` when `NaN`.
"""
Base.@kwdef struct ConsensusValidationConfig
    match_tol::Float64 = 0.1
    sloppy_ratio::Float64 = 1e-2
    gradient_method::Symbol = :forwarddiff
    resolution::Float64 = NaN
end

"""
    ConsensusCP

A single consensus critical point: a matched pair from two fits, decomposed into its identifiable
(stiff) combination and its sloppy uncertainty tube.

# Fields
- `location::Vector{Float64}`: consensus point (midpoint of the matched pair).
- `cp_type::Symbol`: `:minimum` / `:maximum` / `:saddle` / `:degenerate` from the Hessian spectrum.
- `eigenvalues::Vector{Float64}`: Hessian eigenvalues at `location`, ordered by `|λ|` descending.
- `eigenvectors::Matrix{Float64}`: corresponding eigenvectors (columns), same order.
- `n_stiff::Int`: number of stiff (identifiable) directions.
- `identifiable_dirs::Matrix{Float64}`: stiff eigenvectors (columns) — the identifiable combinations.
- `identifiable_values::Vector{Float64}`: value of each identifiable combination at the CP
  (`vₖ · location`) — *what* the consensus validates.
- `sloppy_dirs::Matrix{Float64}`: sloppy eigenvectors (columns) — the axes of the uncertainty tube.
- `disagreement::Vector{Float64}`: `|vₖ · (x₁ − x₂)|`, how far the two copies differ along each
  eigen-direction (same order as `eigenvalues`).
- `stiff_spread::Float64`: disagreement confined to the stiff subspace — the validation residual
  (small ⇒ the identifiable combination is agreed).
- `tube_extent::Float64`: disagreement along the sloppy subspace — the uncertainty tube width.
- `confidence::Float64`: `exp(-stiff_spread/resolution) ∈ (0,1]`.
- `gradient_norm::Float64`: `‖∇f‖` at `location`.
- `newton_decrement::Float64`: `‖H⁻¹∇f‖` at `location` — small ⇒ a genuine stationary point; large
  (as for a non-critical point sitting in a flat valley) ⇒ not stationary.
"""
struct ConsensusCP
    location::Vector{Float64}
    cp_type::Symbol
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}
    n_stiff::Int
    identifiable_dirs::Matrix{Float64}
    identifiable_values::Vector{Float64}
    sloppy_dirs::Matrix{Float64}
    disagreement::Vector{Float64}
    stiff_spread::Float64
    tube_extent::Float64
    confidence::Float64
    gradient_norm::Float64
    newton_decrement::Float64
end

"""
    ConsensusValidationResult

Result of [`validate_consensus`](@ref).

# Fields
- `consensus::Vector{ConsensusCP}`: matched-and-analysed critical points.
- `n_fit1::Int`, `n_fit2::Int`: input critical-point counts.
- `n_matched::Int`: number of matched pairs (`length(consensus)`).
- `unmatched1::Vector{Vector{Float64}}`, `unmatched2::Vector{Vector{Float64}}`: fit-specific CPs with
  no partner within `match_tol` — the sampling-dependent (likely spurious) points consensus drops.
"""
struct ConsensusValidationResult
    consensus::Vector{ConsensusCP}
    n_fit1::Int
    n_fit2::Int
    n_matched::Int
    unmatched1::Vector{Vector{Float64}}
    unmatched2::Vector{Vector{Float64}}
end

"""
    match_critical_point_sets(cps1, cps2; tol) -> (pairs, unmatched1, unmatched2)

Bipartite nearest-neighbour matching of two critical-point sets (each a `Vector` of coordinate
vectors). Candidate pairs within Euclidean distance `tol` are matched greedily shortest-first with
mutual exclusion (each point used at most once). Returns the matched index pairs
`Vector{Tuple{Int,Int}}` and the unmatched coordinate vectors from each set.
"""
function match_critical_point_sets(cps1::AbstractVector, cps2::AbstractVector; tol::Float64)
    n1, n2 = length(cps1), length(cps2)
    cand = Tuple{Float64,Int,Int}[]
    for i in 1:n1, j in 1:n2
        d = norm(float.(cps1[i]) .- float.(cps2[j]))
        d <= tol && push!(cand, (d, i, j))
    end
    sort!(cand; by = first)
    used1, used2 = falses(n1), falses(n2)
    pairs = Tuple{Int,Int}[]
    for (_, i, j) in cand
        (used1[i] || used2[j]) && continue
        used1[i] = true
        used2[j] = true
        push!(pairs, (i, j))
    end
    unmatched1 = [collect(float.(cps1[i])) for i in 1:n1 if !used1[i]]
    unmatched2 = [collect(float.(cps2[j])) for j in 1:n2 if !used2[j]]
    return pairs, unmatched1, unmatched2
end

function _cp_type_from_eigs(λ::AbstractVector; tol::Float64 = 1e-8)
    λmax = maximum(abs, λ)
    any(abs.(λ) .< tol * max(λmax, eps())) && return :degenerate
    all(λ .> 0) && return :minimum
    all(λ .< 0) && return :maximum
    return :saddle
end

# Local Hessian spectrum + gradient diagnostics at a point: eigenpairs ordered stiff-first,
# stiff mask at `sloppy_ratio`, gradient norm, and the eigen-floored Newton decrement.
function _local_spectrum(
    objective,
    x::AbstractVector{Float64},
    method::Symbol,
    sloppy_ratio::Float64,
)
    H = _vw_hessian(objective, x, method)
    H = 0.5 .* (H .+ H')                       # symmetrise
    F = eigen(Symmetric(H))
    ord = sortperm(abs.(F.values); rev = true) # stiff (large |λ|) first
    λ = F.values[ord]
    V = F.vectors[:, ord]
    λmax = maximum(abs, λ)
    stiff = abs.(λ) .>= sloppy_ratio * λmax

    g = _vw_gradient(objective, x, method)
    gnorm = norm(g)
    floor = 1e-12 * max(λmax, eps())
    nd = sqrt(
        sum(eachindex(λ)) do k
            denom =
                abs(λ[k]) < floor ? sign(λ[k]) * floor + (λ[k] == 0 ? floor : 0.0) : λ[k]
            (dot(view(V, :, k), g) / denom)^2
        end,
    )
    return λ, V, stiff, gnorm, nd
end

"""
    validate_consensus(cps1, cps2, objective; config=ConsensusValidationConfig()) -> ConsensusValidationResult

Match two independent fits' critical-point sets and, for each matched pair, decompose the local
Hessian into stiff (identifiable) vs sloppy directions and measure how far the two copies disagree
along each. `cps1`/`cps2` are `Vector`s of coordinate vectors (a `DataFrame` overload with `x1,x2,…`
columns is also provided). `objective` is a callable `f(x)::Real`; its Hessian and gradient are taken
with `config.gradient_method`.

The stiff disagreement (`stiff_spread`) is the validation residual — small when the two fits agree on
the identifiable combination; the sloppy disagreement (`tube_extent`) is the uncertainty tube along
the flat directions. See [`ConsensusCP`](@ref).

# Example
```julia
f = x -> 10x[1]^2 + 0.01x[2]^2                    # stiff in x, sloppy in y
res = validate_consensus([[0.001, 0.2]], [[-0.001, -0.2]], f)
cp = res.consensus[1]
cp.n_stiff              # 1
cp.stiff_spread         # ≈ 0.002  (agree on the identifiable combination)
cp.tube_extent          # ≈ 0.40   (disagree along the sloppy y-axis: the tube)
```
"""
function validate_consensus(
    cps1::AbstractVector,
    cps2::AbstractVector,
    objective;
    config::ConsensusValidationConfig = ConsensusValidationConfig(),
)
    resolution = isnan(config.resolution) ? config.match_tol : config.resolution
    pairs, un1, un2 = match_critical_point_sets(cps1, cps2; tol = config.match_tol)
    out = ConsensusCP[]
    for (i, j) in pairs
        a = collect(float.(cps1[i]))
        b = collect(float.(cps2[j]))
        loc = 0.5 .* (a .+ b)

        λ, V, stiff, gnorm, nd =
            _local_spectrum(objective, loc, config.gradient_method, config.sloppy_ratio)

        Δ = a .- b
        disagree = [abs(dot(view(V, :, k), Δ)) for k in eachindex(λ)]
        stiff_spread =
            sqrt(sum((disagree[k]^2 for k in eachindex(λ) if stiff[k]); init = 0.0))
        tube_extent =
            sqrt(sum((disagree[k]^2 for k in eachindex(λ) if !stiff[k]); init = 0.0))

        idv = [dot(view(V, :, k), loc) for k in eachindex(λ) if stiff[k]]
        push!(
            out,
            ConsensusCP(
                loc,
                _cp_type_from_eigs(λ),
                λ,
                V,
                count(stiff),
                V[:, stiff],
                idv,
                V[:, .!stiff],
                disagree,
                stiff_spread,
                tube_extent,
                exp(-stiff_spread / resolution),
                gnorm,
                nd,
            ),
        )
    end
    return ConsensusValidationResult(out, length(cps1), length(cps2), length(out), un1, un2)
end

"DataFrame overload: extracts coordinate columns `x1, x2, …` from each frame."
function validate_consensus(
    df1::DataFrame,
    df2::DataFrame,
    objective;
    config::ConsensusValidationConfig = ConsensusValidationConfig(),
)
    validate_consensus(_extract_xcols(df1), _extract_xcols(df2), objective; config = config)
end

function _extract_xcols(df::DataFrame)
    cols = sort(
        [c for c in names(df) if occursin(r"^x\d+$", c)];
        by = c -> parse(Int, c[2:end]),
    )
    isempty(cols) && error("no coordinate columns matching x1,x2,… found in DataFrame")
    [Float64[df[r, c] for c in cols] for r in 1:nrow(df)]
end

"""
    EnsembleCluster

One τ-cluster of the pooled K-fit critical-point ensemble (see [`ensemble_consensus`](@ref)).

# Fields
- `centroid::Vector{Float64}`: mean of the member points.
- `cp_type::Symbol`: `:minimum` / `:maximum` / `:saddle` / `:degenerate` from the Hessian at the centroid.
- `n_members::Int`: pooled points in the cluster (one fit may contribute several).
- `fits::Vector{Int}`: sorted indices of the distinct fits represented.
- `recurrence::Float64`: `length(fits)/K` — how systematically this location recurs across fits.
  **This is a systematicity score, not a truth score** (see [`ensemble_consensus`](@ref)).
- `eigenvalues::Vector{Float64}`: Hessian eigenvalues at the centroid, `|λ|` descending.
- `n_stiff::Int`: stiff (identifiable) directions at the centroid.
- `identifiable_values::Vector{Float64}`: stiff-direction coordinates `vₖ · centroid`.
- `stiff_extent::Float64`, `sloppy_extent::Float64`: maximal member deviation from the centroid
  inside the stiff / sloppy subspace — the cluster's own shape split by identifiability.
- `gradient_norm::Float64`, `newton_decrement::Float64`: oracle diagnostics at the centroid
  (`newton_decrement` small ⇒ genuine stationary point of `objective`).
"""
struct EnsembleCluster
    centroid::Vector{Float64}
    cp_type::Symbol
    n_members::Int
    fits::Vector{Int}
    recurrence::Float64
    eigenvalues::Vector{Float64}
    n_stiff::Int
    identifiable_values::Vector{Float64}
    stiff_extent::Float64
    sloppy_extent::Float64
    gradient_norm::Float64
    newton_decrement::Float64
end

"""
    EnsembleConsensusResult

Result of [`ensemble_consensus`](@ref).

# Fields
- `clusters::Vector{EnsembleCluster}`: recurrence-descending.
- `K::Int`: number of fits pooled; `n_pool::Int`: total pooled critical points.
- `frame_point::Vector{Float64}`: where the shared stiff frame was computed (highest-recurrence
  centroid unless overridden).
- `frame_dirs::Matrix{Float64}`: stiff eigenvectors of the Hessian at `frame_point` (columns) —
  the shared identifiable directions the estimate below is expressed in.
- `identifiable_mean::Vector{Float64}`, `identifiable_std::Vector{Float64}`: ensemble estimate of
  each identifiable combination — mean/std of `frame_dirs[:,k] · centroid` over the estimating
  clusters. **This is the headline output** on sloppy landscapes.
- `n_estimating::Int`: clusters with `recurrence ≥ min_recurrence` used for the estimate.
"""
struct EnsembleConsensusResult
    clusters::Vector{EnsembleCluster}
    K::Int
    n_pool::Int
    frame_point::Vector{Float64}
    frame_dirs::Matrix{Float64}
    identifiable_mean::Vector{Float64}
    identifiable_std::Vector{Float64}
    n_estimating::Int
end

"""
    ensemble_consensus(cp_sets, objective; config=ConsensusValidationConfig(),
                       min_recurrence=0.5, frame_point=nothing) -> EnsembleConsensusResult

Pool the critical-point sets of `K = length(cp_sets)` independent fits, greedily τ-cluster them
(`config.match_tol`; a point joins the nearest cluster centroid within τ, else opens a new one),
and score each cluster by **recurrence** = fraction of distinct fits represented.

# Semantics — measured, not aspirational (FHN3D_1 / LV2D, 2026-08-20)

On sloppy landscapes recurrence is a **systematicity** score, *not* a validity score: noise-pinned
valley structures recur across every fit and take the top recurrence, while the true minimizer's
copies scatter along the flat basin and split their votes (measured: true CP at recurrence 0.38
*below* 19 valley clusters at 0.62–1.00). Do **not** threshold on recurrence to accept critical
points; `m ≥ K/2` voting rejects the true minimizer in the PE regime. Use recurrence to *localize
systematic structure* (high) and fit-specific artifacts (`1/K` singletons), and validate points
with the extrinsic `newton_decrement` oracle instead — cheapest applied only inside the
consensus tube.

The **headline output** is the ensemble estimate of the identifiable combination(s):
`identifiable_mean ± identifiable_std`, the statistics of the stiff-frame coordinates over all
clusters with `recurrence ≥ min_recurrence`. Measured on FHN3D_1: the stiff coordinate is
recovered to ~2e-6 at σ=0.02 and ~5e-7 at σ=0.2 objective noise from K=8 fits, with zero
true-objective gradient evaluations at the fitting stage.

The shared stiff frame is the Hessian eigenframe at `frame_point` (default: the
highest-recurrence cluster's centroid; pass a vector to override, e.g. a refined minimizer).

`cp_sets` is a vector of K critical-point sets, each a vector of coordinate vectors (a
`DataFrame` overload is provided). Requires `K ≥ 2` and a nonempty pool.
"""
function ensemble_consensus(
    cp_sets::AbstractVector,
    objective;
    config::ConsensusValidationConfig = ConsensusValidationConfig(),
    min_recurrence::Float64 = 0.5,
    frame_point::Union{Nothing,AbstractVector} = nothing,
)
    K = length(cp_sets)
    K >= 2 || error("ensemble_consensus requires at least 2 critical-point sets, got $K")

    # greedy τ-clustering of the pooled ensemble (running-mean centroids)
    centroids = Vector{Float64}[]
    members = Vector{Vector{Float64}}[]
    fitsets = Vector{Set{Int}}()
    n_pool = 0
    for (k, S) in enumerate(cp_sets), c0 in S
        c = collect(float.(c0))
        n_pool += 1
        best, hit = config.match_tol, 0
        for (ci, cent) in enumerate(centroids)
            d = norm(c .- cent)
            d < best && ((best, hit) = (d, ci))
        end
        if hit == 0
            push!(centroids, copy(c))
            push!(members, [c])
            push!(fitsets, Set(k))
        else
            push!(members[hit], c)
            push!(fitsets[hit], k)
            centroids[hit] = vec(mean(reduce(hcat, members[hit]); dims = 2))
        end
    end
    n_pool > 0 || error("ensemble_consensus: all $K critical-point sets are empty")

    clusters = EnsembleCluster[]
    for ci in eachindex(centroids)
        cent = centroids[ci]
        λ, V, stiff, gnorm, nd =
            _local_spectrum(objective, cent, config.gradient_method, config.sloppy_ratio)
        devs = [m .- cent for m in members[ci]]
        stiff_ext = maximum(
            (
                sqrt(
                    sum(
                        (dot(view(V, :, k), d)^2 for k in eachindex(λ) if stiff[k]);
                        init = 0.0,
                    ),
                ) for d in devs
            );
            init = 0.0,
        )
        sloppy_ext = maximum(
            (
                sqrt(
                    sum(
                        (dot(view(V, :, k), d)^2 for k in eachindex(λ) if !stiff[k]);
                        init = 0.0,
                    ),
                ) for d in devs
            );
            init = 0.0,
        )
        idv = [dot(view(V, :, k), cent) for k in eachindex(λ) if stiff[k]]
        push!(
            clusters,
            EnsembleCluster(
                cent,
                _cp_type_from_eigs(λ),
                length(members[ci]),
                sort!(collect(fitsets[ci])),
                length(fitsets[ci]) / K,
                λ,
                count(stiff),
                idv,
                stiff_ext,
                sloppy_ext,
                gnorm,
                nd,
            ),
        )
    end
    sort!(clusters; by = c -> (c.recurrence, c.n_members), rev = true)

    # shared identifiable frame + ensemble estimate over the recurrent clusters
    fp = frame_point === nothing ? clusters[1].centroid : collect(float.(frame_point))
    λf, Vf, stifff, _, _ =
        _local_spectrum(objective, fp, config.gradient_method, config.sloppy_ratio)
    frame_dirs = Vf[:, stifff]
    est = [c.centroid for c in clusters if c.recurrence >= min_recurrence]
    n_est = length(est)
    nstiff_f = size(frame_dirs, 2)
    id_mean = fill(NaN, nstiff_f)
    id_std = fill(NaN, nstiff_f)
    if n_est >= 1
        for k in 1:nstiff_f
            vals = [dot(view(frame_dirs, :, k), c) for c in est]
            id_mean[k] = mean(vals)
            id_std[k] = n_est >= 2 ? std(vals) : 0.0
        end
    end

    return EnsembleConsensusResult(
        clusters,
        K,
        n_pool,
        fp,
        frame_dirs,
        id_mean,
        id_std,
        n_est,
    )
end

"DataFrame overload: each element's coordinate columns `x1, x2, …` are extracted."
function ensemble_consensus(dfs::AbstractVector{DataFrame}, objective; kwargs...)
    ensemble_consensus([_extract_xcols(df) for df in dfs], objective; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", r::EnsembleConsensusResult)
    println(
        io,
        "EnsembleConsensusResult: $(r.n_pool) pooled CPs from $(r.K) fits -> ",
        "$(length(r.clusters)) clusters; identifiable estimate from $(r.n_estimating)",
    )
    for k in axes(r.frame_dirs, 2)
        @printf(
            io,
            "  identifiable[%d] = %.6g ± %.2g  (dir %s)\n",
            k,
            r.identifiable_mean[k],
            r.identifiable_std[k],
            string(round.(r.frame_dirs[:, k]; digits = 3)),
        )
    end
    for (k, c) in enumerate(r.clusters)
        @printf(
            io,
            "  [%d] rec=%.2f (%d/%d fits, %d pts)  %s  loc=%s  nd=%.2g  sloppy_ext=%.2g\n",
            k,
            c.recurrence,
            length(c.fits),
            r.K,
            c.n_members,
            c.cp_type,
            string(round.(c.centroid; digits = 4)),
            c.newton_decrement,
            c.sloppy_extent,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", r::ConsensusValidationResult)
    println(
        io,
        "ConsensusValidationResult: $(r.n_matched) matched of $(r.n_fit1)×$(r.n_fit2) CPs; ",
        "$(length(r.unmatched1))+$(length(r.unmatched2)) fit-specific dropped",
    )
    for (k, cp) in enumerate(r.consensus)
        @printf(
            io,
            "  [%d] loc=%s  %s  stiff=%d  stiff_spread=%.3g  tube=%.3g  conf=%.2f  newton=%.2g\n",
            k,
            string(round.(cp.location; digits = 4)),
            cp.cp_type,
            cp.n_stiff,
            cp.stiff_spread,
            cp.tube_extent,
            cp.confidence,
            cp.newton_decrement
        )
    end
end
