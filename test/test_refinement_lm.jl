# Levenberg–Marquardt refinement on residual vectors (bead rai5.9).
#
# The pipeline's Phase 3 refined every critical point with Nelder–Mead; on the
# squared ODE loss that refiner stalls at objective ~1e-10 and loses narrow
# minima. `LevenbergMarquardt` takes the residual vector and its exact
# ForwardDiff Jacobian. These tests pin the contract on algebraic residuals:
# exact recovery of a linear least-squares solution, Rosenbrock from the
# standard start, box projection, non-finite residuals, the factory, and the
# result's post-hoc classification.

using Test
using GlobtimPostProcessing
using LinearAlgebra
using Random

@testset "linear least squares: one accepted step lands on the LS solution" begin
    rng = MersenneTwister(7)
    A = randn(rng, 12, 3); xstar = [0.3, -1.2, 2.5]; b = A * xstar .+ 0.01 .* randn(rng, 12)
    xls = A \ b
    r(x) = A * x .- b
    f(x) = sum(abs2, r(x))
    m = LevenbergMarquardt(; residual = r)
    res = refine_point(m, f, [0.0, 0.0, 0.0])
    @test res.converged
    @test isapprox(res.point, xls; atol = 1e-8)
    @test isapprox(res.objective_value, f(xls); rtol = 1e-10)
    @test res.cp_type == :min
    @test res.gradient_norm <= m.g_abstol
    @test res.iterations >= 1
    @test length(res.eigenvalues) == 3 && all(>(0), res.eigenvalues)
    @test res.gradient_method == :forwarddiff
end

@testset "Rosenbrock residuals from (-1.2, 1) reach (1, 1)" begin
    r(x) = [10.0 * (x[2] - x[1]^2), 1.0 - x[1]]
    f(x) = sum(abs2, r(x))
    res = refine_point(LevenbergMarquardt(; residual = r), f, [-1.2, 1.0])
    @test res.converged
    @test isapprox(res.point, [1.0, 1.0]; atol = 1e-7)
    @test res.objective_value < 1e-14
    @test res.cp_type == :min
end

@testset "box projection: minimiser outside the box lands on its edge, unconverged" begin
    r(x) = [x[1] - 2.0, x[2] + 1.0]
    f(x) = sum(abs2, r(x))
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    res = refine_point(LevenbergMarquardt(; residual = r), f, [0.0, 0.0]; bounds = bounds)
    @test all(-1.0 .<= res.point .<= 1.0)
    @test isapprox(res.point, [1.0, -1.0]; atol = 1e-12)
    @test !res.converged                       # gradient is not zero at the edge
    @test res.gradient_norm > 1e-3
end

@testset "non-finite residual at the start returns unconverged, no throw" begin
    r(x) = [NaN, x[1]]
    f(x) = Inf
    res = refine_point(LevenbergMarquardt(; residual = r), f, [0.5])
    @test !res.converged
    @test res.iterations == 0
    @test res.point == [0.5]
    @test !isfinite(res.objective_value)
end

@testset "factory and required residual" begin
    r(x) = x .- 1.0
    m = refinement_method(:levenberg_marquardt; residual = r, max_iterations = 5)
    @test m isa LevenbergMarquardt
    @test m.max_iterations == 5
    @test GlobtimPostProcessing.method_name(m) == :levenberg_marquardt
    @test GlobtimPostProcessing.needs_gradient(m)
    @test_throws Exception LevenbergMarquardt()
    res = refine_point(m, x -> sum(abs2, r(x)), [3.0, -2.0])
    @test isapprox(res.point, [1.0, 1.0]; atol = 1e-10)
end

@testset "stall on a rank-deficient valley is reported, not hidden" begin
    # r depends on x1 only; x2 is a flat direction. LM converges in x1 (gradient
    # zero along both axes at the valley floor) and the Hessian has a zero
    # eigenvalue, so the point is typed degenerate rather than a strict minimum.
    r(x) = [x[1] - 0.5]
    f(x) = sum(abs2, r(x))
    res = refine_point(LevenbergMarquardt(; residual = r), f, [0.0, 0.7])
    @test res.converged
    @test isapprox(res.point[1], 0.5; atol = 1e-10)
    @test res.point[2] == 0.7                  # never moved along the flat axis
    @test res.cp_type == :degenerate_min      # the flat direction is reported, not typed away
end

@testset "batch refine_points dispatches LM" begin
    r(x) = [x[1] - 1.0, x[2] - 2.0]
    f(x) = sum(abs2, r(x))
    results = refine_points(LevenbergMarquardt(; residual = r), f, [[0.0, 0.0], [5.0, 5.0]]; io = devnull)
    @test length(results) == 2
    @test all(res -> isapprox(res.point, [1.0, 2.0]; atol = 1e-10), results)
end
