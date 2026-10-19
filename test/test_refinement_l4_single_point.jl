# Level-4 single-point refinement tests.
#
# test_capture_analysis.jl already covers Newton min/max/saddle/box/batch/
# Hessian-skip/accept_tol/f_accept_tol. This file fills the remaining
# scope: NelderMead behavior near a saddle, LBFGS near a minimum, tolerance
# and trust-region sensitivity, early-exit + f-value safeguard semantics,
# parameter validation, and trace structure.

using Test
using GlobtimPostProcessing
using LinearAlgebra
using Optim

@testset "L4 Single-Point Refinement" begin
    @testset "NelderMead converges near a minimum" begin
        f(x) = (x[1] - 1.0)^2 + (x[2] + 0.5)^2
        result = refine_critical_point(
            f,
            [0.9, -0.4];
            method = Optim.NelderMead(),
            f_abstol = 1e-10,   # the simplex-spread stop; NelderMead has no x test
        )
        @test result.converged
        @test isapprox(result.refined, [1.0, -0.5]; atol = 1e-4)
    end

    @testset "NelderMead near saddle slides to nearby minimum" begin
        # Minima at (±1, 0); saddle at origin (∂²f/∂x² = -4, ∂²f/∂y² = 2).
        # Off-axis start breaks symmetry so the simplex descends one basin.
        f(x) = (x[1]^2 - 1.0)^2 + x[2]^2
        result = refine_critical_point(
            f,
            [0.1, 0.05];
            method = Optim.NelderMead(),
            f_abstol = 1e-10,
            max_iterations = 5000,
        )
        @test result.converged
        @test result.value_refined < 1e-6
        @test abs(abs(result.refined[1]) - 1.0) < 1e-3
        @test abs(result.refined[2]) < 1e-3
        # Did NOT settle on the saddle at origin.
        @test norm(result.refined) > 0.5
    end

    @testset "LBFGS converges near a minimum (symbol dispatch)" begin
        f(x) = (x[1] - 1.0)^2 + (x[2] + 0.5)^2
        result = refine_critical_point(f, [0.9, -0.4]; method = :lbfgs)
        @test result.converged
        @test isapprox(result.refined, [1.0, -0.5]; atol = 1e-6)
    end

    @testset "LBFGS converges near a minimum (Optim object)" begin
        f(x) = sum((x .- [1.0, -0.5]) .^ 2)
        result = refine_critical_point(f, [0.9, -0.4]; method = Optim.LBFGS())
        @test result.converged
        @test isapprox(result.refined, [1.0, -0.5]; atol = 1e-6)
    end

    @testset "Newton tolerance sweep — gradient_norm respects tol" begin
        f(x) = sum(x .^ 2)
        for tol in (1e-4, 1e-8, 1e-12)
            r = refine_to_critical_point(
                f,
                [0.5, 0.3];
                gradient_method = :forwarddiff,
                tol = tol,
            )
            @test r.converged
            @test r.gradient_norm <= tol
            @test r.cp_type == :min
        end
    end

    @testset "Newton tolerance: tighter tol ⇒ ≥ iterations" begin
        f(x) = sum(x .^ 2)
        r_loose = refine_to_critical_point(
            f,
            [0.5, 0.3];
            gradient_method = :forwarddiff,
            tol = 1e-4,
        )
        r_tight = refine_to_critical_point(
            f,
            [0.5, 0.3];
            gradient_method = :forwarddiff,
            tol = 1e-12,
        )
        @test r_loose.iterations <= r_tight.iterations
    end

    @testset "Newton early exit on stalling start" begin
        # Rosenbrock from a far-bowl start with min_improvement_ratio=0.01
        # demands ≥99% gradient reduction within `patience` iterations.
        # Newton's first iterations do not deliver that, so the early-exit
        # heuristic terminates before max_iterations.
        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        r = refine_to_critical_point(
            f,
            [-3.0, 9.0];
            gradient_method = :forwarddiff,
            tol = 1e-8,
            max_iterations = 200,
            patience = 10,
            min_improvement_ratio = 0.01,
        )
        @test !r.converged
        @test r.iterations < 200
    end

    @testset "Trust-region: smaller radius ⇒ ≥ iterations" begin
        f(x) = sum(x .^ 2)
        r_big = refine_to_critical_point(
            f,
            [10.0, 8.0];
            gradient_method = :forwarddiff,
            trust_radius_fraction = 1.0,
            max_iterations = 500,
        )
        r_small = refine_to_critical_point(
            f,
            [10.0, 8.0];
            gradient_method = :forwarddiff,
            trust_radius_fraction = 1e-3,
            max_iterations = 500,
        )
        @test r_big.iterations <= r_small.iterations
    end

    @testset "Trust-region diagnostics are well-formed" begin
        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        r = refine_to_critical_point(f, [-1.0, 1.0]; gradient_method = :forwarddiff)
        @test r.rejected_steps >= 0
        @test r.f_safeguard_count >= 0
        @test isfinite(r.final_trust_radius)
        @test r.final_trust_radius > 0
    end

    @testset "f-value safeguard: stricter factor ⇒ same-or-more safeguarding" begin
        # Same Rosenbrock start under a strict vs. loose plateau guard. Strict
        # cannot record fewer safeguard hits than loose; this is a structural
        # invariant independent of the exact trajectory.
        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        r_strict = refine_to_critical_point(
            f,
            [-2.0, 5.0];
            gradient_method = :forwarddiff,
            max_f_increase_factor = 1.001,
            max_iterations = 100,
        )
        r_loose = refine_to_critical_point(
            f,
            [-2.0, 5.0];
            gradient_method = :forwarddiff,
            max_f_increase_factor = 1e8,
            max_iterations = 100,
        )
        @test r_strict.f_safeguard_count >= r_loose.f_safeguard_count
    end

    @testset "Parameter validation rejects bad inputs" begin
        f(x) = sum(x .^ 2)
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            trust_radius_fraction = 0.0,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            trust_radius_fraction = 1.5,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            trust_expand = 1.0,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            trust_shrink = 1.0,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            max_f_increase_factor = 1.0,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            mode = :unknown_mode,
        )
        @test_throws ErrorException refine_to_critical_point(
            f,
            [1.0, 1.0];
            gradient_method = :unknown_grad,
        )
    end

    @testset "Newton trace structure on quadratic" begin
        f(x) = sum(x .^ 2)
        r = refine_to_critical_point(
            f,
            [0.5, 0.3];
            gradient_method = :forwarddiff,
            trace = true,
        )
        @test r.converged
        @test length(r.trace) == r.iterations + 1
        @test length(r.trace_gradient_norms) == length(r.trace)
        @test r.trace[1] == [0.5, 0.3]
        @test r.trace[end] == r.point
        @test isapprox(r.trace_gradient_norms[end], r.gradient_norm; atol = 1e-12)
        @test isapprox(r.trace_gradient_norms[1], r.initial_gradient_norm; atol = 1e-12)
        # Quadratic descent: gradient norm is monotonic.
        for i in 2:length(r.trace_gradient_norms)
            @test r.trace_gradient_norms[i] <= r.trace_gradient_norms[i-1] + 1e-10
        end
    end

    @testset "Newton trace empty when trace=false (default)" begin
        f(x) = sum(x .^ 2)
        r = refine_to_critical_point(f, [0.5, 0.3]; gradient_method = :forwarddiff)
        @test isempty(r.trace)
        @test isempty(r.trace_gradient_norms)
    end

    @testset "Mode dispatch :minimize avoids saddle classification" begin
        # f = x² − y² has a saddle at origin and is unbounded below in y.
        # :critical_point converges to the saddle; :minimize uses |λ| in the
        # Newton step so y diverges and the iterate does NOT classify as
        # :saddle (either non-converged, or pinned elsewhere).
        f(x) = x[1]^2 - x[2]^2
        r_cp = refine_to_critical_point(
            f,
            [0.1, 0.05];
            gradient_method = :forwarddiff,
            mode = :critical_point,
        )
        @test r_cp.converged
        @test r_cp.cp_type == :saddle

        r_min = refine_to_critical_point(
            f,
            [0.1, 0.05];
            gradient_method = :forwarddiff,
            mode = :minimize,
            max_iterations = 50,
        )
        @test !(r_min.converged && r_min.cp_type == :saddle)
    end
end
