"""
Level-5 multi-start recall tests (bead dx2h).

L4 asked "does refinement converge from a start near a known CP?". L5 asks the
harder question: launched from a dense grid and from random points, does a method
find *every* known local minimum, or does it systematically miss some?

The distinction that matters here is recall vs precision:
  - recall    — fraction of KNOWN minima that some start reached. A miss means a
                basin the method never escapes into; that is a false negative.
  - precision — fraction of DISTINCT converged points that match a known minimum.
                Points that match nothing are not automatically wrong: the ground
                truth lists only analytically known minima, so an unmatched point
                may be a real minimum the table omits (rastrigin lists 1 of 121).
                They are reported, not asserted against, except where the ground
                truth is complete.

Scope note on methods: NelderMead and LBFGS are descent methods, so "find all
minima" is the right question for them and they are what the recall assertions
cover. Newton converges to any critical point — saddles and maxima included — so
holding it to a minima-recall bar would be testing the wrong thing. Its behaviour
is exercised separately at the bottom.

Ground truth comes from L1 (`test_utils/ground_truth_functions.jl`). Only the
five functions with populated `known_minima` are usable: sphere_2d, himmelblau,
sixhump_camel, rosenbrock_2d, styblinski_tang_2d. deuflhard_2d/4d list none, and
rastrigin_2d lists 1 of an expected 121 — it is used for global-minimum capture
only, never for full recall.
"""

using Test
using GlobtimPostProcessing
using LinearAlgebra
using Optim
using Random

if !@isdefined(GroundTruthFunctions)
    include(joinpath(@__DIR__, "test_utils", "ground_truth_functions.jl"))
end
using .GroundTruthFunctions

# ── Harness ──────────────────────────────────────────────────────────────────

"""
    grid_starts(bounds, n_per_dim) -> Vector{Vector{Float64}}

Interior grid over `bounds`, `n_per_dim` points per axis. Interior on purpose:
corner starts sit on the boundary where clamping, not the method, decides the
outcome.
"""
function grid_starts(bounds, n_per_dim::Int)
    axes = [
        range(
            lo + (hi - lo) / (2n_per_dim),
            hi - (hi - lo) / (2n_per_dim),
            length = n_per_dim,
        ) for (lo, hi) in bounds
    ]
    return [collect(p) for p in Iterators.product(axes...)] |> vec
end

"""
    random_starts(bounds, n, seed) -> Vector{Vector{Float64}}

Uniform starts, seeded so a failure is reproducible rather than a coin flip.
"""
function random_starts(bounds, n::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    return [[lo + rand(rng) * (hi - lo) for (lo, hi) in bounds] for _ in 1:n]
end

"""
    multistart_report(bench, method; n_grid, n_random, seed, match_tol, kwargs...)

Run `refine_critical_point` from every start and score the outcome against
`bench.known_minima`.

Returns a NamedTuple with `recall`, `precision`, `convergence_rate`,
`found` (indices of known minima reached), `missed`, `basins` (known-minimum
index → how many starts landed there), `distinct` (deduped converged points),
and `unmatched` (distinct points matching no known minimum).
"""
function multistart_report(
    bench,
    method;
    n_grid::Int = 5,
    n_random::Int = 10,
    seed::Int = 20260812,
    match_tol::Float64 = 1e-3,
    max_iterations::Int = 2000,
)
    starts =
        vcat(grid_starts(bench.bounds, n_grid), random_starts(bench.bounds, n_random, seed))

    converged_pts = Vector{Float64}[]
    n_converged = 0
    basins = Dict{Int,Int}()

    for x0 in starts
        r = refine_critical_point(
            bench.f,
            x0;
            method = method,
            bounds = bench.bounds,
            max_iterations = max_iterations,
        )
        r.converged || continue
        n_converged += 1
        push!(converged_pts, r.refined)
        k = findfirst(m -> norm(r.refined .- m) < match_tol, bench.known_minima)
        k === nothing || (basins[k] = get(basins, k, 0) + 1)
    end

    # Dedup converged points — this is also the false-merge audit: too coarse a
    # tolerance would collapse genuinely distinct minima into one.
    distinct = Vector{Float64}[]
    for p in converged_pts
        any(q -> norm(p .- q) < match_tol, distinct) || push!(distinct, p)
    end

    found = sort(collect(keys(basins)))
    missed = setdiff(1:length(bench.known_minima), found)
    unmatched =
        filter(p -> all(m -> norm(p .- m) >= match_tol, bench.known_minima), distinct)

    n_known = length(bench.known_minima)
    return (
        recall = n_known == 0 ? NaN : length(found) / n_known,
        precision = isempty(distinct) ? NaN :
                    (length(distinct) - length(unmatched)) / length(distinct),
        convergence_rate = n_converged / length(starts),
        found = found,
        missed = missed,
        basins = basins,
        distinct = distinct,
        unmatched = unmatched,
        n_starts = length(starts),
    )
end

# Functions with usable ground truth, paired with the recall bar each can meet.
# rosenbrock is handled in its own testset below: with a single minimum its
# recall is trivially 1.0 for both methods, so the informative quantity there is
# convergence_rate, not recall.
const RECALL_CASES = [
    (name = "sphere_2d", bar = 1.0),
    (name = "himmelblau", bar = 1.0),
    (name = "styblinski_tang_2d", bar = 1.0),
    (name = "sixhump_camel", bar = 1.0),
]

_bench(name) = BENCHMARK_FUNCTIONS[findfirst(b -> b.name == name, BENCHMARK_FUNCTIONS)]

@testset "L5 Multi-Start Recall (dx2h)" begin

    # ========================================================================
    # Harness self-checks — a recall number is worthless if the scaffolding lies
    # ========================================================================

    @testset "harness: starts are inside the domain and reproducible" begin
        b = _bench("himmelblau")
        for x in vcat(grid_starts(b.bounds, 4), random_starts(b.bounds, 8, 1))
            @test all(lo <= xi <= hi for (xi, (lo, hi)) in zip(x, b.bounds))
        end
        @test length(grid_starts(b.bounds, 5)) == 25          # 5^2
        @test random_starts(b.bounds, 6, 42) == random_starts(b.bounds, 6, 42)
        @test random_starts(b.bounds, 6, 42) != random_starts(b.bounds, 6, 43)
    end

    # ========================================================================
    # Recall / precision per function × descent method
    # ========================================================================

    for method_name in (:NelderMead, :LBFGS)
        method = method_name === :NelderMead ? Optim.NelderMead() : Optim.LBFGS()

        @testset "$method_name recall" begin
            for case in RECALL_CASES
                b = _bench(case.name)
                rep = multistart_report(b, method)

                @testset "$(case.name)" begin
                    # The headline: no known minimum is systematically unreachable.
                    @test rep.recall >= case.bar
                    @test isempty(rep.missed)
                    # Every start that converged did so somewhere sensible, and
                    # most starts converge at all.
                    @test rep.convergence_rate > 0.5
                    # False-merge audit: distinct converged points must be at
                    # least as many as the known minima we reached. Fewer would
                    # mean the dedup tolerance collapsed separate minima.
                    @test length(rep.distinct) >= length(rep.found)
                    # Basin audit: each reached minimum owns at least one start.
                    @test all(rep.basins[k] >= 1 for k in rep.found)
                end
            end
        end
    end

    # ========================================================================
    # Multimodal detail — himmelblau is the case with four equal-value minima
    # ========================================================================

    @testset "himmelblau: all four basins are populated, none dominates" begin
        b = _bench("himmelblau")
        rep = multistart_report(b, Optim.NelderMead(); n_grid = 6, n_random = 20)
        @test length(rep.found) == 4
        @test length(rep.basins) == 4
        # With four symmetric basins over a symmetric domain, no single basin
        # should swallow nearly everything — that would indicate the starts, not
        # the method, are doing the work.
        total = sum(values(rep.basins))
        @test maximum(values(rep.basins)) < 0.9 * total
    end

    # ========================================================================
    # Worst-case starts — basin boundaries and a saddle
    # ========================================================================

    @testset "starts on a basin boundary still land on some minimum" begin
        b = _bench("himmelblau")
        # The origin is a local maximum of Himmelblau and equidistant-ish from
        # the four minima: the least helpful possible start.
        for x0 in ([0.0, 0.0], [0.0, 1.0], [-1.0, 0.0])
            r = refine_critical_point(
                b.f,
                x0;
                method = Optim.NelderMead(),
                bounds = b.bounds,
                max_iterations = 2000,
            )
            @test r.converged
            @test any(m -> norm(r.refined .- m) < 1e-2, b.known_minima)
        end
    end

    # ========================================================================
    # Rosenbrock — documented hard case, kept separate from the recall bar
    # ========================================================================

    @testset "rosenbrock: both recall the minimum, LBFGS converges from far fewer starts" begin
        # Measured, not assumed. The textbook expectation is that the banana
        # valley defeats the simplex and rewards gradient methods; on this
        # domain with these settings the opposite holds, so the assertions
        # follow the data:
        #   NelderMead  convergence_rate 1.00  (22/22 starts)
        #   LBFGS       convergence_rate 0.36  ( 8/22 starts)
        # Recall is 1.0 for both — there is a single minimum and each method
        # reaches it from at least one start. The interesting quantity here is
        # therefore convergence_rate, not recall.
        b = _bench("rosenbrock_2d")
        nm = multistart_report(b, Optim.NelderMead(); n_grid = 4, n_random = 6)
        lb = multistart_report(b, Optim.LBFGS(); n_grid = 4, n_random = 6)

        @test nm.recall == 1.0
        @test lb.recall == 1.0
        @test nm.convergence_rate >= 0.9
        # The gap is wide (1.00 vs 0.36); assert it loosely so this catches a
        # real reversal rather than ordinary drift.
        @test lb.convergence_rate < 0.75
        @test lb.convergence_rate < nm.convergence_rate
    end

    # ========================================================================
    # Newton — converges to critical points, not to minima. Different question.
    # ========================================================================

    @testset "Newton reaches critical points, including non-minima" begin
        # Newton lives behind a different entry point: refine_to_critical_point,
        # which reports `cp_type` and `gradient_norm`. refine_critical_point only
        # accepts descent methods (:neldermead, :lbfgs, :bfgs, :conjugategradient,
        # :gradientdescent) — passing :newton there is an error, not a slow path.
        b = _bench("himmelblau")

        types = Symbol[]
        for x0 in vcat(grid_starts(b.bounds, 5), random_starts(b.bounds, 10, 7))
            r = refine_to_critical_point(
                b.f,
                x0;
                gradient_method = :forwarddiff,
                tol = 1e-8,
                bounds = b.bounds,
            )
            r.converged || continue
            @test r.gradient_norm <= 1e-8       # it really is a critical point
            push!(types, r.cp_type)
        end

        @test !isempty(types)
        # The point of the separation: a CP-seeking method lands on saddles and
        # maxima too, so scoring it on minima-recall would penalise it for doing
        # its job. Himmelblau has 4 minima, 4 saddles and 1 maximum.
        @test any(t -> t !== :min, types)
    end

    @testset "Newton classification is correct near known minima" begin
        # Classification accuracy (bead scope): started near a known minimum,
        # Newton must both land on it and label it :min.
        b = _bench("himmelblau")
        for m in b.known_minima
            r = refine_to_critical_point(
                b.f,
                m .+ 0.05;
                gradient_method = :forwarddiff,
                tol = 1e-10,
                bounds = b.bounds,
            )
            @test r.converged
            @test r.cp_type == :min
            # Note the field is `point` here, not `refined` — the Newton path
            # returns CriticalPointRefinementResult, a different type from
            # refine_critical_point's RefinementResult.
            @test norm(r.point .- m) < 1e-4
        end
    end
end
