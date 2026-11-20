"""
    GlobtimPostProcessingPrettyTablesExt

Terminal table printers of GlobtimPostProcessing, loaded by `using PrettyTables`.

`src/` defines each printer as a documented catch-all stub `f(args...; kwargs...)` that
errors with "f requires PrettyTables.jl"; the methods here are more specific, so
dispatch picks them once this extension is loaded. Written against the PrettyTables 2
API (`Highlighter`, `ft_printf`, `tf_unicode_rounded`, `header_crayon`).
"""
module GlobtimPostProcessingPrettyTablesExt

using GlobtimPostProcessing
using GlobtimPostProcessing:
    CaptureResult,
    KnownCriticalPoints,
    DegreeConvergenceInfo,
    DegreeAnalysisResult,
    SparsifyCompare,
    RefinedExperimentResult,
    GradientValidationResult,
    missed_critical_points,
    analyze_subdivision_tree,
    tree_depth_summary,
    compare_subdivision_trees,
    extract_experiment_label,
    format_scientific,
    fmt_sci,
    fmt_time,
    fmt_pct,
    print_section,
    _count_thresholds
import GlobtimPostProcessing:
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
    format_metrics_pretty_table

using LinearAlgebra
using Printf
using PrettyTables
using PrettyTables: Crayon, @crayon_str, Highlighter, ft_printf, tf_borderless, tf_unicode_rounded

include("styled_table.jl")
include("capture_tables.jl")
include("refinement_tables.jl")
include("experiment_tables.jl")
include("subdivision_tables.jl")
include("campaign_tables.jl")

end # module GlobtimPostProcessingPrettyTablesExt
