# test_benefit_matrix.jl — Part A: assemble_benefit_matrix honest-missing semantics
#
# Builds a synthetic cluster_root + audit_root in a tempdir and asserts:
#   * subdivision=off cell picks the best (min) recovery_error across degrees/configs
#   * subdivision=on cell carries rel_l2 + n_leaves but recovery_error is `missing`
#     (audits never solve) — the headline gap
#   * sparsify cells are all `missing` (no ODE sparsification runs)
#   * a system with no single-domain run yields a `missing` off cell + a gap
#   * the cross-cutting comparability caveat is always present

using Test
using GlobtimPostProcessing
using JSON

function _write_json(path, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, obj)
    end
end

mktempdir() do root
    cluster_root = joinpath(root, "cluster")
    audit_root = joinpath(cluster_root, "per_axis_audit")
    spars_root = joinpath(root, "subdivision")   # intentionally has no ODE entries

    # ---- system "alpha": single-domain run with two degrees; best recovery at deg 6 ----
    _write_json(joinpath(cluster_root, "alpha_tight", "results_summary.json"), Dict(
        "results_summary" => Dict(
            "degree_4" => Dict("status" => "success", "recovery_error" => 1.0e-3,
                "best_objective" => 5.0, "relative_l2_error" => 2.0e-2,
                "critical_point_solving_time" => 0.4),
            "degree_6" => Dict("status" => "success", "recovery_error" => 2.0e-7,
                "best_objective" => 1.0, "relative_l2_error" => 3.0e-4,
                "critical_point_solving_time" => 0.9),
            "degree_8" => Dict("status" => "error", "recovery_error" => 1.0e-9),  # excluded: not success
        ),
    ))
    # subdivision-ON audit for "alpha": predicate replay only, no recovery
    _write_json(joinpath(audit_root, "alpha", "deg6_leaves64", "audit_summary.json"),
        Dict("n_leaves" => 64, "wall_s" => 123.4))
    open(joinpath(audit_root, "alpha", "deg6_leaves64", "predcall.jsonl"), "w") do io
        println(io, JSON.json(Dict("rel_l2" => 5.0e-4, "degree" => 6)))
        println(io, JSON.json(Dict("rel_l2" => 9.0e-5, "degree" => 6)))   # min
        println(io, JSON.json(Dict("rel_l2" => 7.0e-4, "degree" => 6)))
    end
    # a smaller audit run that must NOT be chosen (fewer leaves)
    _write_json(joinpath(audit_root, "alpha", "deg2_leaves16", "audit_summary.json"),
        Dict("n_leaves" => 16, "wall_s" => 10.0))

    # ---- system "beta": NO single-domain run, but an audit exists ----
    _write_json(joinpath(audit_root, "beta", "deg4_leaves32", "audit_summary.json"),
        Dict("n_leaves" => 32, "wall_s" => 50.0))
    open(joinpath(audit_root, "beta", "deg4_leaves32", "predcall.jsonl"), "w") do io
        println(io, JSON.json(Dict("rel_l2" => 1.5e-3)))
    end

    cells, gaps = assemble_benefit_matrix(;
        cluster_root = cluster_root,
        audit_root = audit_root,
        sparsification_root = spars_root,
        systems = ["alpha", "beta"],
    )

    getcell(sys, subdiv, spars) = only(filter(
        c -> c.system == sys && c.subdivision == subdiv && c.sparsify == spars, cells))

    @testset "4 cells per system" begin
        @test length(cells) == 8
    end

    @testset "alpha subdivision=off picks best recovery" begin
        c = getcell("alpha", false, false)
        @test c.recovery_error == 2.0e-7          # deg 6, not deg 4, not the errored deg 8
        @test c.best_objective == 1.0
        @test c.rel_l2 == 3.0e-4
        @test c.solve_time == 0.9
    end

    @testset "alpha subdivision=on: rel_l2 + n_leaves, recovery MISSING" begin
        c = getcell("alpha", true, false)
        @test c.n_leaves == 64                    # the larger run, not 16
        @test c.rel_l2 == 9.0e-5                   # min leaf rel_l2
        @test c.solve_time == 123.4                # wall_s
        @test ismissing(c.recovery_error)          # the headline gap
        @test ismissing(c.best_objective)
    end

    @testset "beta: no single-domain run" begin
        c = getcell("beta", false, false)
        @test ismissing(c.recovery_error)
        @test any(g -> occursin("beta", g) && occursin("no single-domain", g), gaps)
        # but beta's audit is present
        @test getcell("beta", true, false).n_leaves == 32
    end

    @testset "sparsify cells all missing" begin
        for sys in ("alpha", "beta"), subdiv in (false, true)
            c = getcell(sys, subdiv, true)
            @test ismissing(c.recovery_error)
            @test ismissing(c.rel_l2)
        end
    end

    @testset "gaps enumerate the real holes" begin
        @test any(g -> occursin("NEVER RECORDED", g), gaps)               # subdivision recovery
        @test any(g -> occursin("no sparsification run", g), gaps)         # ODE sparsify
        @test any(g -> occursin("[ALL]", g) && occursin("controlled", g), gaps)  # comparability caveat
    end
end
