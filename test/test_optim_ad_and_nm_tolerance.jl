# Optim 2 differentiates with central finite
# differences unless told otherwise, and Optim's NelderMead stops on the simplex
# spread of f only (it never reads f_abstol / x_abstol). These tests pin the fix:
# gradient methods get an exact AD backend, and the NelderMead tolerance we expose
# is the one the method actually uses.

using Test
using GlobtimPostProcessing
using Optim
using LinearAlgebra

const AutoFD = GlobtimPostProcessing.AutoFiniteDiff
const AutoFwd = GlobtimPostProcessing.AutoForwardDiff

# Counting objective: every call is tallied by the element type it was called with.
function make_counting_quadratic(m::Int)
    counts = Dict{Symbol,Int}(:float => 0, :dual => 0)
    f = function (x)
        counts[eltype(x) === Float64 ? :float : :dual] += 1
        s = zero(eltype(x))
        for i in 1:m
            s += (i / m) * (x[i] - 1 / i)^2
        end
        return s
    end
    return f, counts
end

@testset "Optim AD backend and NelderMead tolerance" begin
    m = 10
    x0 = fill(0.5, m)

    @testset "LBFGS defaults to ForwardDiff, not 2m finite-difference evaluations" begin
        f_ad, c_ad = make_counting_quadratic(m)
        r_ad = refine_critical_point(f_ad, copy(x0); method = :lbfgs)
        @test r_ad.converged
        @test all(abs.(r_ad.refined .- [1 / i for i in 1:m]) .< 1e-4)
        # Every gradient is one Dual evaluation; the Float64 calls are the line-search
        # values, so they do not scale with the dimension.
        @test c_ad[:dual] > 0
        @test c_ad[:float] < 2m * max(r_ad.iterations, 1)

        f_fd, c_fd = make_counting_quadratic(m)
        r_fd = refine_critical_point(
            f_fd,
            copy(x0);
            method = :lbfgs,
            autodiff = AutoFD(; fdtype = Val(:central)),
        )
        @test r_fd.converged
        @test c_fd[:dual] == 0
        # Central differences: 2m objective calls per gradient.
        @test c_fd[:float] >= 2m * r_fd.g_calls
        @test c_fd[:float] > c_ad[:float]
    end

    @testset "gradient_method on the Optim structs drives the iteration backend" begin
        f_ad, c_ad = make_counting_quadratic(m)
        res = refine_point(OptimLBFGS(), f_ad, copy(x0))
        @test res.converged
        @test c_ad[:dual] > 0

        # A Float64-pinned objective must still be usable through :finitediff ...
        pinned(x) = (eltype(x) === Float64 || error("Dual reached a Float64-pinned objective"); sum(abs2, x .- 0.25))
        res_fd = refine_point(OptimLBFGS(; gradient_method = :finitediff), pinned, copy(x0))
        @test res_fd.converged
        @test all(abs.(res_fd.point .- 0.25) .< 1e-5)
        # ... and :forwarddiff must not be silently downgraded for it: the Dual error
        # surfaces (from the post-hoc ForwardDiff classification, which uses the same
        # backend), it is not quietly finite-differenced away.
        @test_throws ErrorException refine_point(
            OptimLBFGS(; gradient_method = :forwarddiff),
            pinned,
            copy(x0),
        )
    end

    @testset "NelderMead: f_abstol is the simplex-spread stop; x_abstol / g_abstol refused" begin
        g(x) = (x[1] - 1.0)^2 + 3 * (x[2] + 0.5)^2
        loose = refine_critical_point(g, [0.3, 0.2]; f_abstol = 1e-2)
        tight = refine_critical_point(g, [0.3, 0.2]; f_abstol = 1e-12, max_iterations = 5000)
        @test loose.converged && tight.converged
        @test loose.convergence_reason == :f_tol
        @test tight.convergence_reason == :f_tol
        @test tight.iterations > loose.iterations
        @test norm(tight.refined .- [1.0, -0.5]) < norm(loose.refined .- [1.0, -0.5])
        @test norm(tight.refined .- [1.0, -0.5]) < 1e-5

        @test_throws ArgumentError refine_critical_point(g, [0.3, 0.2]; x_abstol = 1e-6)
        @test_throws ArgumentError refine_critical_point(g, [0.3, 0.2]; g_abstol = 1e-6)
        @test_throws ArgumentError refine_critical_points_batch(
            g,
            [[0.3, 0.2]];
            x_abstol = 1e-6,
            show_progress = false,
        )
        # The struct has no x_abstol field any more.
        @test_throws MethodError OptimNelderMead(; x_abstol = 1e-6)
        # The legacy config refuses an x_abstol for NelderMead at conversion time.
        cfg = RefinementConfig(; method = Optim.NelderMead(), x_abstol = 1e-6)
        @test_throws ArgumentError GlobtimPostProcessing.RefinementMethod(cfg)
        # ... but a gradient-method config keeps both tolerances.
        cfg_lbfgs = RefinementConfig(; method = Optim.LBFGS(), f_abstol = 1e-9, x_abstol = 1e-9)
        mth = GlobtimPostProcessing.RefinementMethod(cfg_lbfgs)
        @test mth isa OptimLBFGS && mth.f_abstol == 1e-9 && mth.x_abstol == 1e-9
    end

    @testset "bounded NelderMead keeps a Float64-pinned objective on Float64" begin
        # Fminbox differentiates the barrier objective in its outer loop; the
        # NelderMead default backend is finite differences so that gradient never
        # hands the objective a Dual.
        pinned(x) = (eltype(x) === Float64 || error("Dual reached a Float64-pinned objective"); sum(abs2, x .- 0.25))
        r = refine_critical_point(
            pinned,
            [0.6, -0.1];
            bounds = [(-1.0, 1.0), (-1.0, 1.0)],
            f_abstol = 1e-10,
            max_iterations = 2000,
        )
        @test r.converged
        @test r.error_message === nothing
        @test all(abs.(r.refined .- 0.25) .< 1e-4)
        # Overriding to ForwardDiff on the same objective is an explicit choice and fails loudly.
        r_fwd = refine_critical_point(
            pinned,
            [0.6, -0.1];
            bounds = [(-1.0, 1.0), (-1.0, 1.0)],
            autodiff = AutoFwd(),
        )
        @test !r_fwd.converged
        @test occursin("Float64-pinned", something(r_fwd.error_message, ""))
    end

    # With bounds, Optim.g_converged is Fminbox's OUTER projected-gradient
    # test (against outer_g_abstol), not NelderMead's simplex spread.
    @testset "bounded NelderMead reports the projected-gradient stop as :g_tol" begin
        q(x) = (x[1] - 0.3)^2 + 2 * (x[2] + 0.2)^2
        box = [(-1.0, 1.0), (-1.0, 1.0)]
        # At the default outer_g_abstol = 1e-8 the central-difference projected gradient
        # at a NelderMead optimum (~1e-6) rarely passes; the outer loop then stops when an
        # inner solve leaves x and f exactly unchanged (outer x/f tolerances are 0), which
        # is reported as :f_tol. A reachable outer_g_abstol exercises the g stop.
        r = refine_critical_point(
            q,
            [0.7, 0.5];
            bounds = box,
            f_abstol = 1e-12,
            outer_g_abstol = 1e-5,
        )
        @test r.converged
        @test r.g_converged
        @test r.convergence_reason == :g_tol
        @test norm(r.refined .- [0.3, -0.2]) < 1e-4
        # The unbounded NelderMead spread stop stays :f_tol.
        @test refine_critical_point(q, [0.7, 0.5]; f_abstol = 1e-12).convergence_reason ==
              :f_tol
        # Same label through the batch entry point.
        rb = refine_critical_points_batch(
            q,
            [[0.7, 0.5]];
            bounds = box,
            f_abstol = 1e-12,
            outer_g_abstol = 1e-5,
            show_progress = false,
        )
        @test only(rb).convergence_reason == :g_tol

        # outer_iterations / outer_g_abstol reach Fminbox: one outer solve against an
        # unreachable projected-gradient tolerance runs the outer loop out, and that is
        # recorded rather than reported as a tolerance stop.
        r1 = refine_critical_point(
            q,
            [0.7, 0.5];
            bounds = box,
            f_abstol = 1e-12,
            outer_g_abstol = 0.0,
            outer_iterations = 1,
        )
        @test !r1.converged
        @test r1.iterations == 1
        @test r1.convergence_reason == :outer_iterations
        r3 = only(
            refine_critical_points_batch(
                q,
                [[0.7, 0.5]];
                bounds = box,
                f_abstol = 1e-12,
                outer_g_abstol = 0.0,
                outer_iterations = 3,
                show_progress = false,
            ),
        )
        @test r3.iterations == 3
        @test r3.convergence_reason == :outer_iterations
    end
end
