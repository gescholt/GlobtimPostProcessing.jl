# test_metrics_loader.jl
#
# Round-trip test for GlobtimPostProcessing.load_metrics.
# Writes a JSONL fixture via Globtim.Metrics.MetricsLogger, then reads it
# back and asserts:
#   - meta returns as a Dict with the expected fields
#   - phases / leaves / cps come back as DataFrames with correct counts
#   - bounds get tuple-converted on the leaves DataFrame
#   - missing optional fields fill with `missing` (forward-compatible union
#     of keys)

using Test
using DataFrames
using GlobtimPostProcessing
using Globtim
import Globtim: Metrics as MET

@testset "load_metrics — round-trip with Globtim.Metrics" begin
    mktempdir() do dir
        path = joinpath(dir, "m.jsonl")
        logger = MET.MetricsLogger(path; run_id = "load-metrics-test")

        MET.log_phase!(logger, "start"; extra = Dict(:degree => 3, :note => "hi"))
        MET.log_phase!(logger, "done"; extra = Dict(:wall_s => 1.25, :n_leaves => 2))

        # Two leaves: one with optional fields populated, one without.
        # This exercises the union-of-keys column behavior.
        MET.log_leaf!(logger, 1;
            depth = 0, bounds = [(0.0, 1.0), (-1.0, 1.0)],
            degree = 3, rel_l2 = 1.5e-2, n_raw_cps = 4,
            n_eval_failures = 1, n_grid_evals = 25,
            extra = Dict(:status => "active", :parent_id => nothing),
        )
        MET.log_leaf!(logger, 2;
            depth = 1, bounds = [(0.0, 0.5), (-1.0, 0.0)],
            degree = 3, rel_l2 = 8.0e-3, n_raw_cps = 2,
            extra = Dict(:status => "converged", :parent_id => 1),
        )

        MET.log_cp!(logger, "raw";
            leaf_id = 1, dep_day = 100.0, tof_day = 250.0, dv = 5.6,
            type_label = "Type I", is_boundary = false,
            dedup_group = 1, dedup_canonical = true,
        )

        r = load_metrics(path)

        # Top-level shape
        @test r.path == path
        @test r.n_rows == 1 + 2 + 2 + 1   # meta + 2 phases + 2 leaves + 1 cp

        # meta is a Dict (singleton row)
        @test r.meta isa Dict{String,Any}
        @test r.meta["run_id"] == "load-metrics-test"
        @test r.meta["kind"] == "meta"
        @test haskey(r.meta, "julia_version")
        @test r.meta["n_threads"] isa Integer

        # phases: 2 rows, distinct phase names
        @test r.phases isa DataFrame
        @test nrow(r.phases) == 2
        @test Set(r.phases.phase) == Set(["start", "done"])
        # union of keys: degree (from start) and wall_s (from done) both present
        @test :degree in propertynames(r.phases)
        @test :wall_s in propertynames(r.phases)
        # the row that didn't carry `degree` gets `missing`
        degree_col = r.phases.degree
        @test count(ismissing, degree_col) == 1
        @test 3 in skipmissing(degree_col)

        # leaves: 2 rows, bounds as tuples
        @test r.leaves isa DataFrame
        @test nrow(r.leaves) == 2
        @test eltype(r.leaves.bounds[1]) <: Tuple
        @test r.leaves.bounds[1] == [(0.0, 1.0), (-1.0, 1.0)]
        @test Set(r.leaves.status) == Set(["active", "converged"])
        # n_eval_failures present on leaf 1, missing on leaf 2 (union-of-keys)
        @test :n_eval_failures in propertynames(r.leaves)
        nef = r.leaves.n_eval_failures
        @test count(ismissing, nef) == 1
        @test 1 in skipmissing(nef)
        # parent_id: null on leaf 1 (root), 1 on leaf 2
        @test :parent_id in propertynames(r.leaves)
        @test r.leaves.parent_id[r.leaves.leaf_id .== 2][1] == 1

        # cps
        @test r.cps isa DataFrame
        @test nrow(r.cps) == 1
        @test r.cps.cp_kind[1] == "raw"
        @test r.cps.dv[1] == 5.6
        @test r.cps.dedup_canonical[1] == true
    end
end

@testset "load_metrics — empty file" begin
    mktempdir() do dir
        path = joinpath(dir, "empty.jsonl")
        # Touch it (writer creates a meta row, so we write a totally empty
        # file by hand to exercise the empty-input branch).
        open(path, "w") do io
        end
        r = load_metrics(path)
        @test r.n_rows == 0
        @test r.meta == Dict{String,Any}()
        @test nrow(r.phases) == 0
        @test nrow(r.leaves) == 0
        @test nrow(r.cps) == 0
    end
end

@testset "load_metrics — missing file errors" begin
    @test_throws SystemError load_metrics("/tmp/does_not_exist_at_all_42.jsonl")
end

@testset "load_metrics — malformed row errors" begin
    mktempdir() do dir
        path = joinpath(dir, "bad.jsonl")
        open(path, "w") do io
            println(io, """{"kind":"meta","run_id":"x"}""")
            println(io, """{"no_kind_field":true}""")
        end
        @test_throws ArgumentError load_metrics(path)
    end
end
