"""
    Capture Analysis

Analyzes how well computed polynomial critical points capture known critical points
of the objective function. Given a set of known critical points (with types: min, max,
saddle) and a set of computed points, computes capture rates at multiple adaptive
tolerance levels with per-type breakdowns.

The key metric is: what fraction of ALL known critical points in the domain have
at least one computed polynomial critical point nearby (set-based matching)?

Created: 2026-02-04 (Capture analysis for multi-critical-point objectives)
"""

# Note: LinearAlgebra, Statistics, PrettyTables, Printf are imported in main module

"""
    KnownCriticalPoints

A set of known critical points of an objective function, with their values and types.

# Fields
- `points::Vector{Vector{Float64}}`: Coordinates of each known critical point
- `values::Vector{Float64}`: Objective function value f(x) at each point
- `types::Vector{Symbol}`: Type of each critical point (`:min`, `:max`, `:saddle`)
- `domain_diameter::Float64`: Euclidean diameter of the domain (norm of diagonal)

# Constructor
    KnownCriticalPoints(points, values, types, bounds)

Computes `domain_diameter = norm(upper - lower)` from the bounds.

# Example
```julia
points = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
values = [0.0, 2.0, 1.0]
types = [:min, :max, :saddle]
bounds = [(-2.0, 2.0), (-2.0, 2.0)]
known = KnownCriticalPoints(points, values, types, bounds)
```
"""
struct KnownCriticalPoints
    points::Vector{Vector{Float64}}
    values::Vector{Float64}
    types::Vector{Symbol}
    domain_diameter::Float64
end

const VALID_CP_TYPES = Set([:min, :max, :saddle])

"""
    KnownCriticalPoints(points, values, types, bounds)

Construct `KnownCriticalPoints` from bounds, computing domain diameter automatically.

# Arguments
- `points::Vector{Vector{Float64}}`: Known critical point coordinates
- `values::Vector{Float64}`: f(x) at each known critical point
- `types::Vector{Symbol}`: Type of each CP (`:min`, `:max`, `:saddle`)
- `bounds::Vector{Tuple{Float64,Float64}}`: Domain bounds as (lo, hi) tuples

# Errors
- If `points` is empty
- If `points`, `values`, `types` have different lengths
- If any type is not in `{:min, :max, :saddle}`
"""
function KnownCriticalPoints(
    points::Vector{Vector{Float64}},
    values::Vector{Float64},
    types::Vector{Symbol},
    bounds::Vector{Tuple{Float64,Float64}},
)
    isempty(points) && error("KnownCriticalPoints: points must be non-empty")

    n = length(points)
    length(values) == n || error(
        "KnownCriticalPoints: values length ($(length(values))) must match points length ($n)",
    )
    length(types) == n || error(
        "KnownCriticalPoints: types length ($(length(types))) must match points length ($n)",
    )

    lb, ub = split_bounds(bounds)
    ndim = length(bounds)
    for (i, pt) in enumerate(points)
        length(pt) == ndim || error(
            "KnownCriticalPoints: point $i has dimension $(length(pt)), expected $ndim (from bounds)",
        )
    end

    for (i, t) in enumerate(types)
        t in VALID_CP_TYPES || error(
            "KnownCriticalPoints: invalid type $(repr(t)) at index $i. Must be one of $VALID_CP_TYPES",
        )
    end

    domain_diameter = LinearAlgebra.norm(ub - lb)

    return KnownCriticalPoints(points, values, types, domain_diameter)
end

"""
    CaptureResult

Result of capture analysis: how well computed critical points cover known critical points.

Contains per-known-CP distances, multi-tolerance capture rates, and per-type breakdowns.

# Fields
- `distances::Vector{Float64}`: Min distance from each known CP to nearest computed CP
- `nearest_indices::Vector{Int}`: Index of nearest computed CP for each known CP
- `tolerance_fractions::Vector{Float64}`: Tolerance levels as fractions of domain diameter (sorted ascending)
- `tolerance_values::Vector{Float64}`: Absolute tolerance distances
- `captured_at::Vector{BitVector}`: `captured_at[t][k]` = was known CP k captured at tolerance t?
- `capture_rates::Vector{Float64}`: Fraction of known CPs captured at each tolerance
- `type_capture_rates::Dict{Symbol, Vector{Float64}}`: Per-type capture rates at each tolerance
- `n_known::Int`: Number of known critical points
- `n_computed::Int`: Number of computed critical points
- `domain_diameter::Float64`: Domain diameter used for adaptive tolerance
- `type_counts::Dict{Symbol, Int}`: Count of each CP type in known set
"""
struct CaptureResult
    distances::Vector{Float64}
    nearest_indices::Vector{Int}
    tolerance_fractions::Vector{Float64}
    tolerance_values::Vector{Float64}
    captured_at::Vector{BitVector}
    capture_rates::Vector{Float64}
    type_capture_rates::Dict{Symbol,Vector{Float64}}
    n_known::Int
    n_computed::Int
    domain_diameter::Float64
    type_counts::Dict{Symbol,Int}
end

"""
    capture_rate_at(cr::CaptureResult, fraction::Float64) -> Float64

Look up the capture rate at a specific tolerance fraction.

# Arguments
- `cr::CaptureResult`: Capture analysis result
- `fraction::Float64`: Tolerance fraction (e.g., 0.05 for 5% of domain diameter)

# Returns
- `Float64`: Capture rate at the given tolerance, or `NaN` if that fraction is not present

# Example
```julia
result = compute_capture_analysis(known, computed)
rate = capture_rate_at(result, 0.05)  # capture rate at 5% tolerance
```
"""
function capture_rate_at(cr::CaptureResult, fraction::Float64)::Float64
    idx = findfirst(f -> f ≈ fraction, cr.tolerance_fractions)
    idx === nothing && return NaN
    return cr.capture_rates[idx]
end

"""
    _count_types(types::Vector{Symbol}) -> Dict{Symbol,Int}

Number of occurrences of each critical-point type.
"""
function _count_types(types::Vector{Symbol})
    counts = Dict{Symbol,Int}()
    for tp in types
        counts[tp] = get(counts, tp, 0) + 1
    end
    return counts
end

"""
    compute_capture_analysis(known, computed_points; tolerance_fractions) -> CaptureResult

Compute capture analysis: for each known critical point, find the nearest computed point
and determine capture rates at multiple adaptive tolerance levels.

Uses set-based matching: a known CP is "captured" at tolerance `tol` if ANY computed
critical point is within Euclidean distance `tol` of it.

# Arguments
- `known::KnownCriticalPoints`: The set of known critical points with types
- `computed_points::Vector{Vector{Float64}}`: Computed polynomial critical points
- `tolerance_fractions::Vector{Float64}`: Tolerance levels as fractions of domain diameter.
  Default: `[0.01, 0.025, 0.05, 0.1]` (1%, 2.5%, 5%, 10% of domain diameter)

# Returns
- `CaptureResult`: Full capture analysis with per-type breakdowns

# Example
```julia
known = KnownCriticalPoints(
    [[0.0, 0.0], [1.0, 1.0]],
    [0.0, 2.0],
    [:min, :saddle],
    [-2.0, -2.0], [2.0, 2.0]
)
computed = [[0.01, -0.01], [0.5, 0.5], [0.99, 1.01]]
result = compute_capture_analysis(known, computed)
println("Capture rate at 5%: ", result.capture_rates[3])
```

# Notes
- If `computed_points` is empty, all distances are `Inf` and all capture rates are `0.0`
- Tolerance fractions are sorted ascending internally
- Domain diameter is taken from `known.domain_diameter`
"""
function compute_capture_analysis(
    known::KnownCriticalPoints,
    computed_points::Vector{Vector{Float64}};
    tolerance_fractions::Vector{Float64} = [0.01, 0.025, 0.05, 0.1],
)::CaptureResult
    n_known = length(known.points)
    n_computed = length(computed_points)

    # Sort tolerance fractions ascending
    tol_fracs = sort(tolerance_fractions)
    tol_values = tol_fracs .* known.domain_diameter

    # Handle empty computed points: all known CPs are missed
    if n_computed == 0
        distances = fill(Inf, n_known)
        nearest_indices = zeros(Int, n_known)
        captured_at = [falses(n_known) for _ in tol_fracs]
        capture_rates = zeros(Float64, length(tol_fracs))

        # Per-type rates (all zero)
        type_counts = _count_types(known.types)
        type_capture_rates = Dict{Symbol,Vector{Float64}}()
        for (tp, _) in type_counts
            type_capture_rates[tp] = zeros(Float64, length(tol_fracs))
        end

        return CaptureResult(
            distances,
            nearest_indices,
            tol_fracs,
            tol_values,
            captured_at,
            capture_rates,
            type_capture_rates,
            n_known,
            n_computed,
            known.domain_diameter,
            type_counts,
        )
    end

    # For each known CP, find minimum distance and nearest computed CP
    # Computed per-row to avoid O(n_known * n_computed) matrix allocation
    distances = Vector{Float64}(undef, n_known)
    nearest_indices = Vector{Int}(undef, n_known)
    for k in 1:n_known
        best_dist = Inf
        best_idx = 0
        for c in 1:n_computed
            d = LinearAlgebra.norm(known.points[k] - computed_points[c])
            if d < best_dist
                best_dist = d
                best_idx = c
            end
        end
        distances[k] = best_dist
        nearest_indices[k] = best_idx
    end

    # Compute capture at each tolerance level
    n_tol = length(tol_fracs)
    captured_at = Vector{BitVector}(undef, n_tol)
    capture_rates = Vector{Float64}(undef, n_tol)
    for t in 1:n_tol
        captured_at[t] = distances .<= tol_values[t]
        capture_rates[t] = sum(captured_at[t]) / n_known
    end

    # Per-type capture rates
    type_counts = _count_types(known.types)
    type_capture_rates = Dict{Symbol,Vector{Float64}}()
    for (tp, count) in type_counts
        type_mask = known.types .== tp
        rates = Vector{Float64}(undef, n_tol)
        for t in 1:n_tol
            n_captured_of_type = sum(captured_at[t] .& type_mask)
            rates[t] = n_captured_of_type / count
        end
        type_capture_rates[tp] = rates
    end

    return CaptureResult(
        distances,
        nearest_indices,
        tol_fracs,
        tol_values,
        captured_at,
        capture_rates,
        type_capture_rates,
        n_known,
        n_computed,
        known.domain_diameter,
        type_counts,
    )
end

# Use StatsBase.countmap for type counting (already a dependency of the module)

"""
    missed_critical_points(result, known; tolerance_index) -> Vector{NamedTuple}

Get details of known critical points that were NOT captured at a given tolerance level.

# Arguments
- `result::CaptureResult`: Output from `compute_capture_analysis`
- `known::KnownCriticalPoints`: The known critical points
- `tolerance_index::Int`: Which tolerance level to check (default: last/largest)

# Returns
Vector of named tuples with fields:
- `index::Int`: Index in the known points array
- `point::Vector{Float64}`: Coordinates of the missed CP
- `value::Float64`: f(x) at the missed CP
- `type::Symbol`: Type of the missed CP
- `nearest_distance::Float64`: Distance to the nearest computed CP

# Example
```julia
missed = missed_critical_points(result, known)
for m in missed
    println("Missed CP #\$(m.index) at \$(m.point), type=\$(m.type), nearest=\$(m.nearest_distance)")
end
```
"""
function missed_critical_points(
    result::CaptureResult,
    known::KnownCriticalPoints;
    tolerance_index::Int = length(result.tolerance_fractions),
)::Vector{
    @NamedTuple{
        index::Int,
        point::Vector{Float64},
        value::Float64,
        type::Symbol,
        nearest_distance::Float64,
    }
}
    1 <= tolerance_index <= length(result.tolerance_fractions) || error(
        "tolerance_index $tolerance_index out of range [1, $(length(result.tolerance_fractions))]",
    )

    captured = result.captured_at[tolerance_index]
    missed = @NamedTuple{
        index::Int,
        point::Vector{Float64},
        value::Float64,
        type::Symbol,
        nearest_distance::Float64,
    }[]

    for k in 1:result.n_known
        if !captured[k]
            push!(
                missed,
                (
                    index = k,
                    point = known.points[k],
                    value = known.values[k],
                    type = known.types[k],
                    nearest_distance = result.distances[k],
                ),
            )
        end
    end

    return missed
end

"""
    print_capture_summary(result, known; io=stdout)

Print a formatted summary of capture analysis results.

Outputs three tables:
1. **Overall capture rates** at each tolerance level
2. **Per-type capture rates** (min/max/saddle breakdown)
3. **Missed critical points** at the largest tolerance level

# Arguments
- `result::CaptureResult`: Output from `compute_capture_analysis`
- `known::KnownCriticalPoints`: The known critical points
- `io::IO`: Output stream (default: `stdout`)

Requires PrettyTables.jl: load it with `using PrettyTables` (package extension
`GlobtimPostProcessingPrettyTablesExt`).
"""
print_capture_summary(args...; kwargs...) =
    _missing_extension(print_capture_summary, args, "PrettyTables")

"""
    print_degree_capture_convergence(degree_results; io=stdout)

Print how capture rate improves with polynomial degree.

# Arguments
- `degree_results::Vector{Tuple{Int, CaptureResult}}`: Pairs of (degree, capture_result)
- `io::IO`: Output stream (default: `stdout`)

# Example
```julia
results = [(4, result4), (6, result6), (8, result8)]
print_degree_capture_convergence(results)
```

Requires PrettyTables.jl: load it with `using PrettyTables` (package extension
`GlobtimPostProcessingPrettyTablesExt`).
"""
print_degree_capture_convergence(args...; kwargs...) =
    _missing_extension(print_degree_capture_convergence, args, "PrettyTables")

# --- Mapping from CriticalPointClassification.jl String types to our Symbol types ---

const _CLASSIFICATION_TO_SYMBOL = Dict{String,Symbol}(
    "minimum" => :min,
    "maximum" => :max,
    "saddle" => :saddle,
    "degenerate" => :saddle,          # treat degenerate as saddle for capture purposes
    "degenerate_min" => :min,         # valley minimum → treat as min for capture
    "degenerate_max" => :max,         # valley maximum → treat as max for capture
    "degenerate_saddle" => :saddle,   # mixed → treat as saddle for capture
)

"""
    _classification_string_to_symbol(s::String) -> Symbol

Map CriticalPointClassification.jl output (`"minimum"`, `"maximum"`, `"saddle"`, `"degenerate"`)
to capture analysis symbols (`:min`, `:max`, `:saddle`).
"""
function _classification_string_to_symbol(s::String)::Symbol
    haskey(_CLASSIFICATION_TO_SYMBOL, s) || error("Unknown classification: $(repr(s))")
    return _CLASSIFICATION_TO_SYMBOL[s]
end

"""
    _map_cp_type_for_capture(cp_type::Symbol) -> Symbol

Map a CP type (possibly a degenerate sub-type) to a capture-analysis-compatible
type (`:min`, `:max`, or `:saddle`).

Degenerate sub-types are mapped to their "effective" type for capture analysis:
- `:degenerate_min` → `:min`
- `:degenerate_max` → `:max`
- `:degenerate_saddle` → `:saddle`
- `:degenerate` → `:saddle`
- `:unknown` → `:saddle`

Non-degenerate types pass through unchanged.
"""
function _map_cp_type_for_capture(cp_type::Symbol)::Symbol
    cp_type == :min && return :min
    cp_type == :max && return :max
    cp_type == :saddle && return :saddle
    cp_type == :degenerate_min && return :min
    cp_type == :degenerate_max && return :max
    cp_type == :degenerate_saddle && return :saddle
    cp_type == :degenerate && return :saddle
    cp_type == :unknown && return :saddle
    return :saddle  # conservative default
end

"""
    build_known_cps_from_2d_product(
        objective_2d,
        points_2d::Vector{Vector{Float64}},
        bounds::Vector{Tuple{Float64,Float64}};
        hessian_tol::Float64 = 1e-6
    ) -> KnownCriticalPoints

Build 4D known critical points from 2D critical points for a separable function
`f(x₁,x₂,x₃,x₄) = g(x₁,x₂) + g(x₃,x₄)`.

Takes the Cartesian product of the 2D critical points to form all N² 4D critical points,
classifies each by combining the Hessian eigenvalue analysis of both 2D components.

Uses `classify_critical_point` from `CriticalPointClassification.jl` for Hessian-based
classification of each 2D component, then combines types:
- min + min → min
- max + max → max
- everything else → saddle

# Arguments
- `objective_2d`: The 2D component function g(x) where f = g(x₁₂) + g(x₃₄)
- `points_2d::Vector{Vector{Float64}}`: Known 2D critical points of g
- `bounds::Vector{Tuple{Float64,Float64}}`: Domain bounds as (lo, hi) tuples (must be 4D)
- `hessian_tol::Float64`: Tolerance for eigenvalue sign classification (default: 1e-6)

# Returns
- `KnownCriticalPoints`: All N² 4D critical points with values and types

# Example
```julia
deuflhard_2d(x) = (exp(x[1]^2 + x[2]^2) - 3)^2 + (x[1] + x[2] - sin(3(x[1] + x[2])))^2
pts_2d = [[0.0, 0.0], [0.741, 0.741], ...]  # from CSV
known_4d = build_known_cps_from_2d_product(
    deuflhard_2d, pts_2d,
    [(-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)]
)
```
"""
function build_known_cps_from_2d_product(
    objective_2d,
    points_2d::Vector{Vector{Float64}},
    bounds::Vector{Tuple{Float64,Float64}};
    hessian_tol::Float64 = 1e-6,
)::KnownCriticalPoints
    isempty(points_2d) && error("points_2d must be non-empty")
    length(bounds) == 4 || error("bounds must be 4D, got $(length(bounds))D")

    n_2d = length(points_2d)

    # Classify each 2D point using Hessian eigenvalues
    types_2d = Vector{Symbol}(undef, n_2d)
    vals_2d = Vector{Float64}(undef, n_2d)
    for i in 1:n_2d
        vals_2d[i] = objective_2d(points_2d[i])
        H = ForwardDiff.hessian(objective_2d, points_2d[i])
        eigs = eigvals(Symmetric(H))
        class_str = classify_critical_point(collect(eigs); tol = hessian_tol)
        types_2d[i] = _classification_string_to_symbol(class_str)
    end

    # Build all N^2 4D combinations
    n_4d = n_2d^2
    points_4d = Vector{Vector{Float64}}(undef, n_4d)
    values_4d = Vector{Float64}(undef, n_4d)
    types_4d = Vector{Symbol}(undef, n_4d)

    idx = 0
    for i in 1:n_2d, j in 1:n_2d
        idx += 1
        points_4d[idx] = vcat(points_2d[i], points_2d[j])
        values_4d[idx] = vals_2d[i] + vals_2d[j]

        # Combine types: min+min=min, max+max=max, else saddle
        if types_2d[i] == :min && types_2d[j] == :min
            types_4d[idx] = :min
        elseif types_2d[i] == :max && types_2d[j] == :max
            types_4d[idx] = :max
        else
            types_4d[idx] = :saddle
        end
    end

    return KnownCriticalPoints(points_4d, values_4d, types_4d, bounds)
end

"""
    build_known_cps_from_refinement(
        objective,
        raw_points::Vector{Vector{Float64}},
        bounds::Vector{Tuple{Float64,Float64}};
        method::RefinementMethod = OptimNelderMead(),
        dedup_fraction = 0.01,
        top_k = nothing,
        trace = false,
    ) -> Union{NamedTuple, Nothing}

Build a reference set of known critical points by refining raw polynomial critical
points to true critical points of `f` using the specified `method`.

# Arguments
- `objective`: Objective function `f(x::Vector{Float64}) -> Float64`
- `raw_points`: Raw polynomial CPs (typically from the highest degree)
- `bounds`: Domain bounds as `(lo, hi)` tuples

# Keyword Arguments
- `method::RefinementMethod`: Refinement method (default: `OptimNelderMead()`)
- `dedup_fraction::Float64`: Fraction of domain diameter for deduplication (default: 0.01)
- `top_k::Union{Int, Nothing}`: Pre-filter to `top_k` CPs with smallest `f(x)` (default: `nothing` = all)
- `trace::Bool`: Record iteration trajectories (default: `false`)

# Returns
- Named tuple `(known_cps, refinement_results, all_refinement_results, refined_raw_points)`,
  or `nothing` if no CPs survived filtering.
"""
function build_known_cps_from_refinement(
    objective,
    raw_points::Vector{Vector{Float64}},
    bounds::Vector{Tuple{Float64,Float64}};
    method::RefinementMethod = OptimNelderMead(),
    dedup_fraction::Float64 = 0.01,
    top_k::Union{Int,Nothing} = nothing,
    trace::Bool = false,
)
    isempty(raw_points) && error("raw_points must be non-empty")
    lb, ub = split_bounds(bounds)
    n = length(bounds)
    for (i, pt) in enumerate(raw_points)
        length(pt) == n || error("raw_points[$i] has dimension $(length(pt)), expected $n")
    end
    0 < dedup_fraction < 1 || error("dedup_fraction must be in (0, 1), got $dedup_fraction")

    domain_diameter = LinearAlgebra.norm(ub .- lb)
    dedup_tol = dedup_fraction * domain_diameter

    # Step 0 (optional): Pre-filter by objective value — keep only top_k CPs with smallest f(x)
    # This is a cheap pre-filter (one f evaluation per CP) that avoids spending refinement
    # iterations on CPs with large objective values far from any minimum.
    if top_k !== nothing
        top_k > 0 || error("top_k must be positive, got $top_k")
        if top_k < length(raw_points)
            n_raw = length(raw_points)
            @printf("    Pre-filtering: evaluating f(x) at %d raw CPs...\n", n_raw)
            t_prefilter = time()
            obj_values = Vector{Float64}(undef, n_raw)
            for i in 1:n_raw
                obj_values[i] = objective(raw_points[i])
            end
            order = sortperm(obj_values)
            selected = order[1:top_k]
            @printf(
                "    Selected top %d by f(x): f ∈ [%.4e, %.4e]  (%.1fs)\n",
                top_k,
                obj_values[selected[1]],
                obj_values[selected[end]],
                time() - t_prefilter
            )
            @printf(
                "    Discarded %d CPs with f > %.4e\n",
                n_raw - top_k,
                obj_values[selected[end]]
            )
            raw_points = raw_points[selected]
        else
            @printf(
                "    top_k=%d >= %d raw CPs — refining all\n",
                top_k,
                length(raw_points)
            )
        end
    end

    # Snapshot the (possibly pre-filtered) raw points for downstream diagnostics.
    # After top_k filtering, raw_points may be a subset of the original input.
    # Indices align 1:1 with the refinement_results vector below.
    refined_raw_points = copy(raw_points)

    # Step 1: Refine selected points using unified dispatch
    refinement_results =
        refine_points(method, objective, raw_points; bounds = bounds, trace = trace)

    # Step 2: Accept CPs that converged or have meaningful classification
    accepted = filter(r -> r.converged || r.cp_type != :unknown, refinement_results)

    n_strict = count(r -> r.converged, refinement_results)
    n_accepted = count(r -> !r.converged && r.cp_type != :unknown, refinement_results)

    if isempty(accepted)
        best_grad = minimum(r -> r.gradient_norm, refinement_results)
        best_f = minimum(r -> r.objective_value, refinement_results)
        msg =
            "  No critical points found: $(length(raw_points)) polynomial CPs refined, " *
            "0 converged, 0 accepted. " *
            "Best |∇f|=$(@sprintf("%.2e", best_grad)), best f=$(@sprintf("%.2e", best_f))."
        println(msg)
        return nothing
    end

    # Report acceptance breakdown when there are non-strict acceptances
    if n_accepted > 0
        println("  $n_strict strictly converged, $n_accepted accepted (classified)")
    end

    # Step 3: Deduplicate — sort by gradient norm (best first), greedy keep
    sorted = sort(accepted, by = r -> r.gradient_norm)

    unique_results = CriticalPointRefinementResult[]
    for r in sorted
        is_duplicate = false
        for kept in unique_results
            if LinearAlgebra.norm(r.point .- kept.point) < dedup_tol
                is_duplicate = true
                break
            end
        end
        if !is_duplicate
            push!(unique_results, r)
        end
    end

    # Step 4: Build KnownCriticalPoints
    # Map degenerate sub-types to capture-compatible types (:min, :max, :saddle)
    points = [r.point for r in unique_results]
    values = [r.objective_value for r in unique_results]
    types = Symbol[_map_cp_type_for_capture(r.cp_type) for r in unique_results]

    n_deduped = length(accepted) - length(unique_results)
    n_min = count(==(:min), types)
    n_max = count(==(:max), types)
    n_saddle = count(==(:saddle), types)
    if n_deduped > 0
        println(
            "  After dedup: $(length(unique_results)) unique CPs ($n_deduped duplicates removed) — $n_min min, $n_max max, $n_saddle saddle",
        )
    else
        println(
            "  $(length(unique_results)) unique CPs — $n_min min, $n_max max, $n_saddle saddle",
        )
    end

    known_cps = KnownCriticalPoints(points, values, types, bounds)
    return (
        known_cps = known_cps,
        refinement_results = unique_results,
        all_refinement_results = refinement_results,
        refined_raw_points = refined_raw_points,
    )
end

# ─── Degree Convergence Summary and Verdict ─────────────────────────────────

"""
    DegreeConvergenceInfo

Per-degree data for the convergence summary table. Collects polynomial approximation
quality metrics alongside capture analysis results.

# Fields
- `degree::Int`: Polynomial degree
- `l2_error::Float64`: L2 approximation error of the polynomial
- `n_critical_points::Int`: Number of raw polynomial critical points found
- `min_gradient_norm::Union{Float64, Nothing}`: Minimum ||∇f|| at raw CPs (nothing if not computed)
- `best_objective::Union{Float64, Nothing}`: Best objective value (raw or refined)
"""
struct DegreeConvergenceInfo
    degree::Int
    l2_error::Float64
    relative_l2_error::Float64  # l2_error / ||f||_L2 (dimensionless)
    n_critical_points::Int
    min_gradient_norm::Union{Float64,Nothing}
    best_objective::Union{Float64,Nothing}
end

"""
    print_degree_convergence_summary(
        degree_capture_results::Vector{Tuple{Int, CaptureResult}},
        degree_info::Vector{DegreeConvergenceInfo};
        io::IO = stdout,
    )

Print a combined convergence summary table showing L2 error, gradient quality,
capture rates at key tolerances (1%, 5%, 10%), and best objective per degree.

# Arguments
- `degree_capture_results`: Vector of `(degree, CaptureResult)` tuples.
- `degree_info`: Vector of `DegreeConvergenceInfo` with per-degree metrics.
- `io::IO`: Output stream (default: `stdout`).

# Example
```julia
info = [DegreeConvergenceInfo(4, 0.1, 0.05, 50, 1e-3, 0.5),
        DegreeConvergenceInfo(6, 0.01, 0.003, 120, 1e-5, 0.3)]
print_degree_convergence_summary(degree_capture_results, info)
```

Requires PrettyTables.jl: load it with `using PrettyTables` (package extension
`GlobtimPostProcessingPrettyTablesExt`).
"""
print_degree_convergence_summary(args...; kwargs...) =
    _missing_extension(print_degree_convergence_summary, args, "PrettyTables")

"""
    CaptureVerdict

Result of the capture verdict analysis: which degree achieves the best capture rate,
with per-type breakdown and trend across degrees.

# Fields
- `best_degree::Int`: Degree with highest capture rate at reference tolerance
- `capture_rate::Float64`: Best capture rate (0.0 to 1.0)
- `n_captured::Int`: Number of known CPs captured at the best degree
- `n_known::Int`: Total number of known CPs
- `tolerance_fraction::Float64`: The reference tolerance fraction used
- `tolerance_absolute::Float64`: The absolute tolerance value
- `type_breakdown::Vector{@NamedTuple{type::Symbol, captured::Int, total::Int, rate::Float64}}`:
  Per-type capture counts and rates
- `degree_trend::Vector{@NamedTuple{degree::Int, rate::Float64}}`:
  Capture rate at each degree (for trend display)
- `label::String`: Verdict label ("EXCELLENT", "GOOD", "POOR")
"""
struct CaptureVerdict
    best_degree::Int
    capture_rate::Float64
    n_captured::Int
    n_known::Int
    tolerance_fraction::Float64
    tolerance_absolute::Float64
    type_breakdown::Vector{
        @NamedTuple{type::Symbol, captured::Int, total::Int, rate::Float64}
    }
    degree_trend::Vector{@NamedTuple{degree::Int, rate::Float64}}
    label::String
end

"""
    compute_capture_verdict(
        degree_capture_results::Vector{Tuple{Int, CaptureResult}};
        reference_tolerance_fraction::Float64 = 0.05,
    ) -> CaptureVerdict

Compute the capture verdict: which degree achieves the best capture rate,
with per-type breakdown and trend across degrees.

# Arguments
- `degree_capture_results`: Vector of `(degree, CaptureResult)` tuples.
- `reference_tolerance_fraction`: Which tolerance to use for the verdict (default: 5%).

# Returns
- `CaptureVerdict`: Structured result with verdict label, best degree, and breakdowns.

# Example
```julia
verdict = compute_capture_verdict(degree_capture_results)
verdict.label       # "EXCELLENT"
verdict.best_degree # 8
verdict.capture_rate # 1.0
```
"""
function compute_capture_verdict(
    degree_capture_results::Vector{Tuple{Int,CaptureResult}};
    reference_tolerance_fraction::Float64 = 0.05,
)::CaptureVerdict
    isempty(degree_capture_results) && error("degree_capture_results must be non-empty")

    ref_cr = degree_capture_results[1][2]
    tol_fracs = ref_cr.tolerance_fractions

    ref_tol_idx = findfirst(f -> f ≈ reference_tolerance_fraction, tol_fracs)
    ref_tol_idx !== nothing || error(
        "reference_tolerance_fraction=$reference_tolerance_fraction not found in " *
        "tolerance_fractions=$(tol_fracs)",
    )

    # Find best degree by capture rate at reference tolerance
    best_deg_idx = 1
    best_capture_rate = 0.0
    for (i, (_, cr)) in enumerate(degree_capture_results)
        rate = cr.capture_rates[ref_tol_idx]
        if rate > best_capture_rate
            best_capture_rate = rate
            best_deg_idx = i
        end
    end

    best_deg, best_cr = degree_capture_results[best_deg_idx]
    n_captured = count(best_cr.captured_at[ref_tol_idx])
    tol_abs = best_cr.tolerance_values[ref_tol_idx]

    # Per-type breakdown at best degree
    type_breakdown = @NamedTuple{type::Symbol, captured::Int, total::Int, rate::Float64}[]
    for sym in sort(collect(keys(best_cr.type_capture_rates)))
        rates = best_cr.type_capture_rates[sym]
        type_count = best_cr.type_counts[sym]
        type_captured = round(Int, rates[ref_tol_idx] * type_count)
        push!(
            type_breakdown,
            (
                type = sym,
                captured = type_captured,
                total = type_count,
                rate = rates[ref_tol_idx],
            ),
        )
    end

    # Trend across degrees
    degree_trend = @NamedTuple{degree::Int, rate::Float64}[]
    for (deg, cr) in degree_capture_results
        push!(degree_trend, (degree = deg, rate = cr.capture_rates[ref_tol_idx]))
    end

    # Verdict label
    label =
        best_capture_rate >= 0.80 ? "EXCELLENT" :
        best_capture_rate >= 0.50 ? "GOOD" : "POOR"

    return CaptureVerdict(
        best_deg,
        best_capture_rate,
        n_captured,
        best_cr.n_known,
        reference_tolerance_fraction,
        tol_abs,
        type_breakdown,
        degree_trend,
        label,
    )
end

const _CP_TYPE_DISPLAY = Dict(
    :min => "Minima",
    :max => "Maxima",
    :saddle => "Saddles",
    :degenerate => "Degenerate",
    :degenerate_min => "Degen. Min",
    :degenerate_max => "Degen. Max",
    :degenerate_saddle => "Degen. Saddle",
)

"""
    print_capture_verdict(
        verdict::CaptureVerdict;
        io::IO = stdout,
    )

Print a colored capture verdict banner with per-type breakdown and degree trend.

# Arguments
- `verdict::CaptureVerdict`: Computed verdict from `compute_capture_verdict`.
- `io::IO`: Output stream (default: `stdout`).

# Example
```julia
verdict = compute_capture_verdict(degree_capture_results)
print_capture_verdict(verdict)
```
"""
function print_capture_verdict(verdict::CaptureVerdict; io::IO = stdout)
    W = 80
    verdict_col =
        verdict.capture_rate >= 0.80 ? :green :
        verdict.capture_rate >= 0.50 ? :yellow : :red

    printstyled(io, "="^W * "\n"; color = verdict_col, bold = true)
    printstyled(
        io,
        " CAPTURE RESULT [$(verdict.label)]  —  Degree $(verdict.best_degree)\n";
        color = verdict_col,
        bold = true,
    )
    printstyled(io, "="^W * "\n"; color = verdict_col, bold = true)

    tol_pct = @sprintf("%.1f%%", 100 * verdict.tolerance_fraction)
    printstyled(
        io,
        "  Captured: $(verdict.n_captured) / $(verdict.n_known) " *
        "($(@sprintf("%.1f%%", 100 * verdict.capture_rate)))  " *
        "at $tol_pct tolerance (abs = $(round(verdict.tolerance_absolute, digits=4)))\n";
        color = verdict_col,
        bold = true,
    )

    # Per-type breakdown
    type_parts = String[]
    for tb in verdict.type_breakdown
        type_label = get(_CP_TYPE_DISPLAY, tb.type, string(tb.type))
        pct = tb.total > 0 ? @sprintf("%.1f%%", 100 * tb.rate) : "N/A"
        push!(type_parts, "$type_label: $(tb.captured)/$(tb.total) ($pct)")
    end
    printstyled(io, "  " * join(type_parts, "  |  ") * "\n"; color = verdict_col)

    # Trend
    trend_parts =
        [@sprintf("deg %d: %.1f%%", t.degree, 100 * t.rate) for t in verdict.degree_trend]
    printstyled(io, "  Trend: " * join(trend_parts, " → ") * "\n"; color = :white)

    printstyled(io, "="^W * "\n"; color = verdict_col, bold = true)
end
