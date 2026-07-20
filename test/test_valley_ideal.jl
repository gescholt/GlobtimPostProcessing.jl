"""
    test_valley_ideal.jl

Tests for approximate vanishing-ideal recovery (ValleyIdeal.jl): given point
samples of a positive-dimensional set, recover its defining polynomial system.

Validated targets (all with known ideals; deterministic parametric samples):
- unit circle in 2D            — ⟨x₁² + x₂² − 1⟩, one degree-2 generator
- shifted/scaled circle in 2D  — physical-coordinate back-conversion
- hyperbola x₁x₂ = 2           — the identifiable combination of the product-form
                                 non-identifiability model
- circle in the x₃=0 plane, 3D — ⟨x₃, x₁² + x₂² − 1⟩: generators at two different
                                 degrees, exercises the minimal-generator logic
- unit sphere in 3D            — a k=2 valley: interpolation is dimension-agnostic
                                 even though curve walking cannot trace it
- trace_valley integration     — walker samples → recovered ideal
"""

using Test
using LinearAlgebra

# ─── helpers ────────────────────────────────────────────────────────────────

# Evaluate a (exponents, coeffs) polynomial at a point.
function eval_poly(exps, coeffs, x)
    s = 0.0
    for (e, c) in zip(exps, coeffs)
        s += c * prod(x[i]^e[i] for i in eachindex(x))
    end
    return s
end

# Extract the coefficient of a given exponent vector (0.0 if absent).
coeff_of(exps, coeffs, e) = begin
    i = findfirst(==(e), exps)
    i === nothing ? 0.0 : coeffs[i]
end

circle_points(n; r = 1.0, cx = 0.0, cy = 0.0) =
    [[cx + r * cos(t), cy + r * sin(t)] for t in range(0, 2π; length = n + 1)[1:n]]

# ─── unit circle ────────────────────────────────────────────────────────────

@testset "circle: single degree-2 generator" begin
    pts = circle_points(200)
    r = recover_vanishing_ideal(pts; config = ValleyIdealConfig(degree_max = 3))

    # Truncated Hilbert function of a principal ideal with deg g = 2:
    # nothing at degree 1; g at degree 2; {g, x₁g, x₂g} at degree 3.
    @test r.kernel_dims == [0, 1, 3]
    @test r.multiple_dims == [0, 0, 3]
    @test length(r.generators) == 1

    g = r.generators[1]
    @test g.degree == 2
    @test g.residual < 1e-10

    # Physical coordinates: proportional to x₁² + x₂² − 1.
    exps, coeffs = physical_generator(g, r)
    scale = coeff_of(exps, coeffs, [2, 0])
    @test abs(scale) > 0.1
    @test coeff_of(exps, coeffs, [0, 2]) / scale ≈ 1.0 atol = 1e-8
    @test coeff_of(exps, coeffs, [0, 0]) / scale ≈ -1.0 atol = 1e-8
    @test abs(coeff_of(exps, coeffs, [1, 0]) / scale) < 1e-8
    @test abs(coeff_of(exps, coeffs, [0, 1]) / scale) < 1e-8
end

@testset "shifted circle: physical back-conversion" begin
    # (x − 3)² + (y + 1)² = 4
    pts = circle_points(200; r = 2.0, cx = 3.0, cy = -1.0)
    r = recover_vanishing_ideal(pts; config = ValleyIdealConfig(degree_max = 2))
    @test length(r.generators) == 1

    exps, coeffs = physical_generator(r.generators[1], r)
    # Vanishes on the physical circle, and is genuinely quadratic there.
    for t in range(0, 2π; length = 17)
        @test abs(eval_poly(exps, coeffs, [3 + 2cos(t), -1 + 2sin(t)])) < 1e-6
    end
    scale = coeff_of(exps, coeffs, [2, 0])
    @test coeff_of(exps, coeffs, [0, 2]) / scale ≈ 1.0 atol = 1e-6
    @test coeff_of(exps, coeffs, [1, 0]) / scale ≈ -6.0 atol = 1e-6
    @test coeff_of(exps, coeffs, [0, 1]) / scale ≈ 2.0 atol = 1e-6
    @test coeff_of(exps, coeffs, [0, 0]) / scale ≈ 6.0 atol = 1e-6
end

# ─── hyperbola (identifiable combination) ───────────────────────────────────

@testset "hyperbola: recovers the identifiable combination" begin
    pts = [[a, 2.0 / a] for a in range(0.4, 5.0; length = 150)]
    r = recover_vanishing_ideal(pts; config = ValleyIdealConfig(degree_max = 3))

    @test r.kernel_dims == [0, 1, 3]
    @test length(r.generators) == 1
    g = r.generators[1]
    @test g.degree == 2
    @test g.residual < 1e-8

    exps, coeffs = physical_generator(g, r)
    scale = coeff_of(exps, coeffs, [1, 1])
    @test abs(scale) > 0.1
    @test coeff_of(exps, coeffs, [0, 0]) / scale ≈ -2.0 atol = 1e-6
    @test abs(coeff_of(exps, coeffs, [2, 0]) / scale) < 1e-6
    @test abs(coeff_of(exps, coeffs, [0, 2]) / scale) < 1e-6
end

# ─── circle in 3D: two generators, different degrees ────────────────────────

@testset "circle in x₃=0 plane: minimal generators ⟨x₃, x₁²+x₂²−1⟩" begin
    pts = [[cos(t), sin(t), 0.0] for t in range(0, 2π; length = 201)[1:200]]
    r = recover_vanishing_ideal(pts; config = ValleyIdealConfig(degree_max = 2))

    # Degree ≤1: just x₃. Degree ≤2: x₃, x₁x₃, x₂x₃, x₃² (multiples) + circle.
    @test r.kernel_dims == [1, 5]
    @test r.multiple_dims == [0, 4]
    @test length(r.generators) == 2
    @test r.generators[1].degree == 1
    @test r.generators[2].degree == 2

    # First generator ∝ x₃ (in normalized coordinates the degenerate axis
    # is exactly the third coordinate).
    g1 = r.generators[1]
    c1 = g1.coeffs ./ g1.coeffs[argmax(abs.(g1.coeffs))]
    @test coeff_of(g1.exponents, c1, [0, 0, 1]) ≈ 1.0 atol = 1e-8
    @test abs(coeff_of(g1.exponents, c1, [1, 0, 0])) < 1e-8
    @test abs(coeff_of(g1.exponents, c1, [0, 1, 0])) < 1e-8

    # Second generator: vanishes on the circle and involves x₁², x₂².
    exps, coeffs = physical_generator(r.generators[2], r)
    for t in range(0, 2π; length = 13)
        @test abs(eval_poly(exps, coeffs, [cos(t), sin(t), 0.0])) < 1e-6
    end
    scale = coeff_of(exps, coeffs, [2, 0, 0])
    @test abs(scale) > 0.1
    @test coeff_of(exps, coeffs, [0, 2, 0]) / scale ≈ 1.0 atol = 1e-6
end

# ─── sphere: a k=2 valley ───────────────────────────────────────────────────

@testset "sphere: interpolation handles intrinsic dimension 2" begin
    # Deterministic covering: latitude-longitude grid (poles excluded).
    pts = Vector{Vector{Float64}}()
    for θ in range(0.2, π - 0.2; length = 12), φ in range(0, 2π; length = 13)[1:12]
        push!(pts, [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)])
    end
    r = recover_vanishing_ideal(pts; config = ValleyIdealConfig(degree_max = 2))

    @test r.kernel_dims == [0, 1]
    @test length(r.generators) == 1
    g = r.generators[1]
    @test g.degree == 2
    @test g.residual < 1e-8

    exps, coeffs = physical_generator(g, r)
    scale = coeff_of(exps, coeffs, [2, 0, 0])
    @test coeff_of(exps, coeffs, [0, 2, 0]) / scale ≈ 1.0 atol = 1e-6
    @test coeff_of(exps, coeffs, [0, 0, 2]) / scale ≈ 1.0 atol = 1e-6
    @test coeff_of(exps, coeffs, [0, 0, 0]) / scale ≈ -1.0 atol = 1e-6
end

# ─── integration with the valley walker ─────────────────────────────────────

@testset "trace_valley → recover_vanishing_ideal" begin
    f_circle(x) = (x[1]^2 + x[2]^2 - 1)^2
    cfg = ValleyWalkConfig(
        eigenvalue_threshold = 1e-2,
        initial_step_size = 0.05,
        max_steps = 300,
        method = :tangent_projection,
    )
    trace = trace_valley(f_circle, [1.0, 0.0], cfg)
    @test trace.converged

    r = recover_vanishing_ideal(trace; config = ValleyIdealConfig(degree_max = 2))
    @test r.kernel_dims == [0, 1]
    @test length(r.generators) == 1
    @test r.generators[1].residual < 1e-6

    exps, coeffs = physical_generator(r.generators[1], r)
    scale = coeff_of(exps, coeffs, [2, 0])
    @test coeff_of(exps, coeffs, [0, 2]) / scale ≈ 1.0 atol = 1e-4
    @test coeff_of(exps, coeffs, [0, 0]) / scale ≈ -1.0 atol = 1e-4
end

# ─── guardrails ─────────────────────────────────────────────────────────────

@testset "guardrails" begin
    # Too few samples for the requested degree must raise, not hallucinate.
    pts = circle_points(10)
    @test_throws ErrorException recover_vanishing_ideal(
        pts;
        config = ValleyIdealConfig(degree_max = 4),
    )

    # Inconsistent dimensions raise.
    @test_throws ErrorException recover_vanishing_ideal([[1.0, 2.0], [1.0, 2.0, 3.0]])

    # A full-dimensional cloud (no algebraic relation) yields no generators.
    grid = [[x, y] for x in range(-1, 1; length = 8) for y in range(-1, 1; length = 8)]
    r = recover_vanishing_ideal(grid; config = ValleyIdealConfig(degree_max = 2))
    @test isempty(r.generators)
    @test r.kernel_dims == [0, 0]
end
