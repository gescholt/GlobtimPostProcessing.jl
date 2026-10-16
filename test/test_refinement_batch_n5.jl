# N5: threaded batch refinement with a per-thread objective pool, and representatives-only
# refinement driven by the cluster columns Globtim writes into critical_points_raw_deg_X.csv.

using Test
using GlobtimPostProcessing
using CSV, DataFrames

@testset "N5 batch refinement" begin
    dw(x) = (x[1]^2 - 0.25)^2 + x[2]^2
    starts = [[0.45, 0.02], [-0.52, -0.03], [0.48, -0.01], [-0.47, 0.02], [0.51, 0.03], [-0.5, 0.01]]
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    @testset "threads = true gives the sequential answers" begin
        seq = refine_critical_points_batch(dw, starts; bounds = bounds, show_progress = false)
        thr = refine_critical_points_batch(dw, starts; bounds = bounds, show_progress = false, threads = true)
        @test length(thr) == length(seq) == 6
        @test all(r.converged for r in thr)
        for (a, b) in zip(seq, thr)
            @test a.refined ≈ b.refined atol = 1e-5
            @test abs(a.value_refined - b.value_refined) < 1e-10
        end
        # a per-thread pool is indexed by threadid; a short pool is refused
        pool = [dw for _ in 1:Threads.maxthreadid()]
        thr2 = refine_critical_points_batch(dw, starts; bounds = bounds, show_progress = false, threads = true, objective_pool = pool)
        @test all(r.converged for r in thr2)
        if Threads.nthreads() > 1
            @test_throws ArgumentError refine_critical_points_batch(
                dw, starts; bounds = bounds, show_progress = false, threads = true, objective_pool = [dw],
            )
        end
    end

    @testset "representatives only from the CSV columns" begin
        dir = mktempdir()
        # six raw roots in three clusters (two representatives per cluster marked by Globtim)
        df = DataFrame(
            :index => 1:6,
            :p1 => [0.5, 0.5000001, -0.5, -0.4999999, 0.0, 1e-7],
            :p2 => [0.0, 1e-7, 0.0, -1e-7, 0.0, 0.0],
            :objective => [dw([0.5, 0.0]), dw([0.5, 0.0]), dw([-0.5, 0.0]), dw([-0.5, 0.0]), dw([0.0, 0.0]), dw([0.0, 0.0])],
            :cluster_id => [1, 1, 2, 2, 3, 3],
            :cluster_size => [2, 2, 2, 2, 2, 2],
            :representative => [1, 0, 1, 0, 1, 0],
        )
        CSV.write(joinpath(dir, "critical_points_raw_deg_4.csv"), df)
        raw = load_raw_critical_points(dir)
        @test raw.n_points == 6
        @test raw.representative == [true, false, true, false, true, false]
        @test raw.cluster_size == fill(2, 6)

        cfg = RefinementConfig(; bounds = bounds, show_progress = false)
        r_rep = refine_experiment_results(dir, dw, cfg)
        @test r_rep.n_raw == 3                      # representatives only (default)
        @test r_rep.n_converged == 3
        r_all = refine_experiment_results(dir, dw, cfg; representatives_only = false)
        @test r_all.n_raw == 6
        r_thr = refine_experiment_results(dir, dw, cfg; threads = true)
        @test r_thr.n_raw == 3 && r_thr.n_converged == 3

        # a CSV without the columns: every point is its own representative
        dir2 = mktempdir()
        CSV.write(joinpath(dir2, "critical_points_raw_deg_4.csv"), df[:, [:index, :p1, :p2, :objective]])
        raw2 = load_raw_critical_points(dir2)
        @test all(raw2.representative) && all(raw2.cluster_size .== 1)
        @test refine_experiment_results(dir2, dw, cfg).n_raw == 6
    end
end
