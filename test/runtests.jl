using Test
using GlobtimPostProcessing

# Include shared test fixtures once (used by test_capture_analysis, test_refinement_phase1,
# test_integration_real_fixtures — previously each included it independently, causing
# duplicate method definition warnings)
include(joinpath(@__DIR__, "fixtures", "test_functions.jl"))

# Include shared test utility modules once (used by test_cp_verification, test_ground_truth,
# test_gradient_hessian_accuracy, test_hessian_classification_correctness — previously
# each included them independently, causing module redefinition errors)
include(joinpath(@__DIR__, "test_utils", "cp_verification.jl"))
include(joinpath(@__DIR__, "test_utils", "ground_truth_functions.jl"))
using .CPVerification
using .GroundTruthFunctions

# Shared constant used by test_ground_truth, test_gradient_hessian_accuracy,
# test_hessian_classification_correctness
const GT_DIR = joinpath(@__DIR__, "fixtures", "ground_truth")

@testset "GlobtimPostProcessing.jl" begin
    @testset "Module Loading" begin
        @test isdefined(GlobtimPostProcessing, :load_experiment_results)
        @test isdefined(GlobtimPostProcessing, :load_campaign_results)
        @test isdefined(GlobtimPostProcessing, :compute_statistics)
        @test isdefined(GlobtimPostProcessing, :ExperimentResult)
        @test isdefined(GlobtimPostProcessing, :CampaignResults)
    end

    @testset "Data Structures" begin
        # Test ExperimentResult construction
        exp_result = ExperimentResult(
            "test_experiment",
            Dict{String,Any}("test" => "value"),
            ["label1", "label2"],
            ["label1", "label2", "label3"],
            nothing,
            nothing,
            nothing,
            "/tmp/test",
        )

        @test exp_result.experiment_id == "test_experiment"
        @test length(exp_result.enabled_tracking) == 2
        @test length(exp_result.tracking_capabilities) == 3
    end

    # Run Data Loading tests (Issue #7, Phase 1)
    @testset "Data Loading (Issue #7, Phase 1)" begin
        include("test_data_loading.jl")
    end

    # Run Parameter Recovery tests (Issue #7)
    @testset "Parameter Recovery (Issue #7)" begin
        include("test_parameter_recovery.jl")
    end

    # Run Parameter Recovery Real Data tests (Issue #7, Phase 2)
    @testset "Parameter Recovery - Real Data (Issue #7, Phase 2)" begin
        include("test_parameter_recovery_real_data.jl")
    end

    # Run Quality Diagnostics tests (Issue #7, Phase 3)
    @testset "Quality Diagnostics (Issue #7, Phase 3)" begin
        include("test_quality_diagnostics.jl")
    end

    # Run Critical Point Classification tests
    @testset "Critical Point Classification" begin
        include("test_critical_point_classification.jl")
    end

    # Run Landscape Fidelity tests
    @testset "Landscape Fidelity" begin
        include("test_landscape_fidelity.jl")
    end

    # Run Phase 1 Refinement tests (simple functions, no Globtim dependency)
    @testset "Phase 1 Refinement" begin
        include("test_refinement_phase1.jl")
    end

    # Saddle links: Newton on ∇f=0, descent, Morse data, which minima a saddle separates
    @testset "Saddle Links" begin
        include("test_saddle_links.jl")
    end

    # Run Capture Analysis tests (known CP capture rates, no Globtim dependency)
    @testset "Capture Analysis" begin
        include("test_capture_analysis.jl")
    end

    # Level-4 single-point refinement — NM/LBFGS/Newton convergence,
    # tolerance + trust-region sensitivity, early-exit, f-safeguard, trace
    @testset "L4 Single-Point Refinement" begin
        include("test_refinement_l4_single_point.jl")
    end

    # Level-5 multi-start recall — from grid + seeded random starts,
    # does each descent method reach EVERY known local minimum? Recall/precision,
    # basin coverage, false-merge audit, worst-case starts, Newton classification.
    @testset "L5 Multi-Start Recall" begin
        include("test_multistart_recall_l5.jl")
    end

    # Run Integration tests with Real Fixtures (uses real globtim data)
    @testset "Integration: Real Fixtures" begin
        include("test_integration_real_fixtures.jl")
    end

    # Run Unified Pipeline tests (January 2026)
    @testset "Unified Pipeline" begin
        include("test_unified_pipeline.jl")
    end

    # Run Loader Consolidation tests (February 2026)
    @testset "Loader Consolidation" begin
        include("test_loader_consolidation.jl")
    end

    # Run LV4D TUI tests (January 2026 - domain filter fix)
    @testset "LV4D TUI" begin
        include("test_lv4d_tui.jl")
    end

    # Run LV4D Sweep tests (January 2026 - error message improvement)
    @testset "LV4D Sweep" begin
        include("test_lv4d_sweep.jl")
    end

    # Run Subdivision Tree Analysis tests (January 2026)
    @testset "Subdivision Tree Analysis" begin
        include("test_subdivision_analysis.jl")
    end

    # Run Valley Walking tests (February 2026)
    @testset "Valley Walking" begin
        include("test_valley_walking.jl")
    end

    # Run Valley Ideal tests (July 2026, VALLEY-IDEAL.1)
    @testset "Valley Ideal" begin
        include("test_valley_ideal.jl")
    end

    # Run Consensus Validation tests (August 2026)
    @testset "Consensus Validation" begin
        include("test_consensus_validation.jl")
    end

    # Run Postprocessing Pipeline tests (March 2026)
    @testset "Postprocessing Pipeline" begin
        include("test_postprocessing_pipeline.jl")
    end

    # Landscape capture data types (March 2026)
    @testset "Landscape Capture Types" begin
        include("test_landscape_capture_types.jl")
    end

    # CP verification oracle (Level 0 — March 2026)
    @testset "CP Verification Oracle" begin
        include("test_cp_verification.jl")
    end

    # Ground truth reference data (Level 1 — March 2026)
    @testset "Ground Truth Reference Data" begin
        include("test_ground_truth.jl")
    end

    # Gradient/Hessian accuracy (Level 2 — March 2026)
    @testset "Gradient/Hessian Accuracy" begin
        include("test_gradient_hessian_accuracy.jl")
    end

    # Hessian classification correctness (Level 3 — March 2026)
    @testset "Hessian Classification Correctness" begin
        include("test_hessian_classification_correctness.jl")
    end

    # Metrics JSONL loader (May 2026 — round-trip with Globtim.Metrics)
    @testset "Metrics JSONL Loader" begin
        include("test_metrics_loader.jl")
    end

    # Benefit matrix assembly (June 2026 — Part A baseline; honest-missing semantics)
    @testset "Benefit Matrix" begin
        include("test_benefit_matrix.jl")
    end

    # Levenberg–Marquardt refinement on residual vectors (September 2026)
    @testset "Levenberg-Marquardt Refinement" begin
        include("test_refinement_lm.jl")
    end

    # N5: threaded batch refinement, representatives-only refinement from the cluster columns
    @testset "N5 Batch Refinement" begin
        include("test_refinement_batch_n5.jl")
    end

    # N6: index-preserving Newton census (promoted from experiments/sandbox/lv_critical_census.jl)
    @testset "N6 Newton Census" begin
        include("test_newton_census.jl")
    end

    # Aqua.jl quality assurance tests
    include("test_aqua.jl")
end
