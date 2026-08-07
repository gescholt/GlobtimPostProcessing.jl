# Consensus validation of critical points across two independent polynomial fits.
#
# Motivation (beads n1hx / pxvr). Fitting an objective twice from different sample points (or noise
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

        H = _vw_hessian(objective, loc, config.gradient_method)
        H = 0.5 .* (H .+ H')                       # symmetrise
        F = eigen(Symmetric(H))
        ord = sortperm(abs.(F.values); rev = true) # stiff (large |λ|) first
        λ = F.values[ord]
        V = F.vectors[:, ord]
        λmax = maximum(abs, λ)

        stiff = abs.(λ) .>= config.sloppy_ratio * λmax
        Δ = a .- b
        disagree = [abs(dot(view(V, :, k), Δ)) for k in eachindex(λ)]
        stiff_spread =
            sqrt(sum((disagree[k]^2 for k in eachindex(λ) if stiff[k]); init = 0.0))
        tube_extent =
            sqrt(sum((disagree[k]^2 for k in eachindex(λ) if !stiff[k]); init = 0.0))

        g = _vw_gradient(objective, loc, config.gradient_method)
        gnorm = norm(g)
        floor = 1e-12 * max(λmax, eps())
        nd = sqrt(
            sum(eachindex(λ)) do k
                denom =
                    abs(λ[k]) < floor ? sign(λ[k]) * floor + (λ[k] == 0 ? floor : 0.0) :
                    λ[k]
                (dot(view(V, :, k), g) / denom)^2
            end,
        )

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
