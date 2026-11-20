# Optional dependencies as package extensions (bead 0do2).
#
#   GlobtimPostProcessingPrettyTablesExt  trigger PrettyTables   table printers
#   GlobtimPostProcessingJLD2Ext          trigger JLD2           load_subdivision_tree
#   GlobtimPostProcessingTUIExt           triggers REPL, PrettyTables  postprocess() + menus
#
# (a) and (b) run in a fresh julia process, because this test session has already
# loaded the triggers. PrettyTables is a dependency of DataFrames, so it is loaded by
# every `using GlobtimPostProcessing` and its extension is always present; its stub
# branch is unreachable while DataFrames is a hard dependency and is checked through
# the shared message helper instead.

using Test
using GlobtimPostProcessing
using PrettyTables
using JLD2
using REPL

const _GPP = GlobtimPostProcessing

function _run_fresh_julia(code::String)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return success(proc), String(take!(out))
end

const _FRESH_SESSION = raw"""
using GlobtimPostProcessing
const G = GlobtimPostProcessing
loaded(name) = any(k -> k.name == name, keys(Base.loaded_modules))
ext(name) = Base.get_extension(G, name) !== nothing
println("REPL_LOADED=", loaded("REPL"))
println("JLD2_LOADED=", loaded("JLD2"))
println("PT_LOADED=", loaded("PrettyTables"))
println("JLD2EXT=", ext(:GlobtimPostProcessingJLD2Ext))
println("TUIEXT=", ext(:GlobtimPostProcessingTUIExt))
println("PTEXT=", ext(:GlobtimPostProcessingPrettyTablesExt))
msg(f, args...) = try
    f(args...)
    "NO ERROR"
catch e
    sprint(showerror, e)
end
println("MSG_JLD2=", msg(load_subdivision_tree, "tree.jld2"))
println("MSG_TUI=", msg(postprocess))
println("MSG_MENU=", msg(G.UnifiedPipeline.confirm_selection, "summary"))

# (c) load the triggers in the same process: the features work
using JLD2
path = joinpath(mktempdir(), "tree.jld2")
jldsave(path; subdivision_tree = Dict("leaves" => 3), metadata = Dict("function" => "f"))
tree, meta = load_subdivision_tree(path)
println("JLD2_ROUNDTRIP=", tree == Dict("leaves" => 3) && meta == Dict("function" => "f"))
using REPL
println("TUIEXT_AFTER=", ext(:GlobtimPostProcessingTUIExt))
println("EMPTY_ROOT=", G.UnifiedPipeline.select_recent_experiment(mktempdir()) === nothing)
"""

field(out, key) = (m = match(Regex("^" * key * "=(.*)\$", "m"), out); m === nothing ? nothing : m[1])

@testset "Optional dependencies (package extensions)" begin
    @testset "fresh session: extensions load only with their triggers" begin
        ok, out = _run_fresh_julia(_FRESH_SESSION)
        ok || @info "fresh-session output" out
        @test ok
        # (a) not loaded by a plain `using GlobtimPostProcessing`
        @test field(out, "REPL_LOADED") == "false"
        @test field(out, "JLD2_LOADED") == "false"
        @test field(out, "JLD2EXT") == "false"
        @test field(out, "TUIEXT") == "false"
        # PrettyTables arrives with DataFrames; the extension follows its trigger
        @test field(out, "PTEXT") == field(out, "PT_LOADED")
        # (b) stubs name the missing package
        @test occursin("load_subdivision_tree requires JLD2.jl", field(out, "MSG_JLD2"))
        @test occursin("postprocess requires REPL", field(out, "MSG_TUI"))
        @test occursin("confirm_selection requires REPL", field(out, "MSG_MENU"))
        # (c) features work once the trigger is loaded
        @test field(out, "JLD2_ROUNDTRIP") == "true"
        @test field(out, "TUIEXT_AFTER") == "true"
        @test field(out, "EMPTY_ROOT") == "true"
    end

    @testset "stub message and MethodError when the extension is loaded" begin
        msg = _GPP._missing_extension_message(:styled_table, "PrettyTables")
        @test occursin("styled_table requires PrettyTables.jl", msg)
        @test occursin("using PrettyTables", msg)
        # Extension loaded but no method matches: a MethodError, not a misleading hint
        @test_throws MethodError styled_table(1, 2, 3)
        @test_throws MethodError load_subdivision_tree(1, 2)
        @test_throws MethodError postprocess(1)
    end

    @testset "every stubbed printer has an extension method" begin
        ptext = Base.get_extension(_GPP, :GlobtimPostProcessingPrettyTablesExt)
        jldext = Base.get_extension(_GPP, :GlobtimPostProcessingJLD2Ext)
        tuiext = Base.get_extension(_GPP, :GlobtimPostProcessingTUIExt)
        @test ptext !== nothing
        @test jldext !== nothing
        @test tuiext !== nothing
        has_ext_method(f, mod) = any(m -> m.module === mod, methods(f))
        for f in (
            styled_table,
            print_capture_summary,
            print_degree_capture_convergence,
            print_degree_convergence_summary,
            print_comparison_table,
            print_experiment_header,
            print_poly_summary_table,
            print_degree_analysis_table,
            print_parameter_recovery_table,
            print_sparsification_metrics_table,
            print_sparsification_capture_table,
            print_subdivision_tree_report,
            print_tree_comparison_table,
            _GPP.format_metrics_pretty_table,
        )
            @test has_ext_method(f, ptext)
        end
        @test has_ext_method(load_subdivision_tree, jldext)
        UP = _GPP.UnifiedPipeline
        for f in (
            UP.postprocess,
            UP.select_experiment_type,
            UP.select_results_source,
            UP.select_recent_experiment,
            UP.select_analysis_mode,
            UP.confirm_selection,
        )
            @test has_ext_method(f, tuiext)
        end
    end

    # Printers not exercised elsewhere in the suite. A colour IOContext makes
    # PrettyTables evaluate the highlighter closures.
    @testset "table printers render (PrettyTables extension)" begin
        buf = IOBuffer()
        io = IOContext(buf, :color => true)
        rendered() = String(take!(buf))

        print_experiment_header("Header", ["Function" "f"; "Dimension" "2"]; io = io)
        @test occursin("Dimension", rendered())

        dr = [
            (
                degree = d,
                n_critical_points = 3,
                l2_approx_error = 1e-3 / d,
                relative_l2_error = 1e-4 / d,
                best_objective = 0.1,
                status = d == 4 ? "success" : "failed",
                total_computation_time = 1.5,
                polynomial_construction_time = 0.5,
                critical_point_solving_time = 0.7,
            ) for d in (4, 6)
        ]
        print_poly_summary_table(dr; io = io)
        @test occursin("Rel L2", rendered())
        print_poly_summary_table(dr; show_timing_breakdown = true, io = io)
        @test occursin("HC Solve", rendered())

        gv = GradientValidationResult([1e-9, 1e-3], [true, false], 1, 1, 1e-6, 5e-4, 5e-4, 1e-3, 1e-9)
        refined = RefinedExperimentResult(
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0]],
            [1.0, 2.0],
            [0.5],
            [0.5],
            [true, false],
            [3, 0],
            2,
            1,
            1,
            0,
            0.5,
            0.5,
            1,
            1,
            1.0,
            0.5,
            mktempdir(),
            4,
            0.1,
            RefinementConfig(),
        )
        print_degree_analysis_table([(4, gv, refined)]; io = io)
        @test occursin("Converged", rendered())

        print_parameter_recovery_table(
            [1.0, 2.0],
            [1.0, 2.0005];
            raw_estimate = [1.1, 2.2],
            degree = 4,
            io = io,
        )
        @test occursin("raw est.", rendered())
        print_parameter_recovery_table([1.0, 2.0], [1.0 + 1e-8, 2.1]; io = io)
        @test occursin("refined est.", rendered())

        sc = [
            GlobtimPostProcessing.SparsifyCompare(
                d, t, "1e-$k", 20, 5, 0.8, 1.0, 12, 4, 0.8 + s, 0.4, 40.0, 1.1, 2.5,
            ) for d in (4, 6) for (k, t, s) in ((3, 1e-3, 0.0), (2, 1e-2, -0.1))
        ]
        print_sparsification_metrics_table(sc; io = io)
        @test occursin("Speedup", rendered())
        print_sparsification_capture_table(sc; io = io)
        @test occursin("Δ Cap", rendered())

        metrics = Dict(
            "total_timing" => Dict(
                "mean" => 1.0,
                "min" => 0.5,
                "max" => 2.0,
                "std" => 0.1,
                "num_experiments" => 2,
                "best_experiment" => "/runs/exp_deg4_GN8",
            ),
        )
        out = mktemp() do _, f
            redirect_stdout(f) do
                _GPP.format_metrics_pretty_table(metrics, Dict())
                print_comparison_table(refined, gv; raw_gradient_norms = [1e-2, 1e-1])
            end
            flush(f)
            seekstart(f)
            read(f, String)
        end
        @test occursin("Total Computation Time", out)
        @test occursin("Raw vs Refined", out)
    end
end
