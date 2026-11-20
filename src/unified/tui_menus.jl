"""
Shared menu utilities for unified TUI.

Provides reusable menu components and styling for the interactive interface.
"""

# ============================================================================
# ANSI Terminal Colors
# ============================================================================

const UNIFIED_TUI_CYAN = "\e[36m"
const UNIFIED_TUI_YELLOW = "\e[33m"
const UNIFIED_TUI_GREEN = "\e[32m"
const UNIFIED_TUI_RED = "\e[31m"
const UNIFIED_TUI_DIM = "\e[2m"
const UNIFIED_TUI_BOLD = "\e[1m"
const UNIFIED_TUI_RESET = "\e[0m"

# ============================================================================
# Header and Formatting
# ============================================================================

"""
    tui_header(title::String)

Print a styled TUI header.
"""
function tui_header(title::String)
    println()
    println("$(UNIFIED_TUI_BOLD)$(UNIFIED_TUI_CYAN)$title$(UNIFIED_TUI_RESET)")
    println(UNIFIED_TUI_DIM * ("─"^length(title)) * UNIFIED_TUI_RESET)
    println("$(UNIFIED_TUI_DIM)Use ↑/↓ to navigate, Enter to select$(UNIFIED_TUI_RESET)")
    println()
end

"""
    tui_section(title::String)

Print a section divider with title.
"""
function tui_section(title::String)
    println()
    println("$(UNIFIED_TUI_DIM)── $title ──$(UNIFIED_TUI_RESET)")
end

"""
    tui_success(msg::String)

Print a success message.
"""
function tui_success(msg::String)
    println("$(UNIFIED_TUI_GREEN)✓$(UNIFIED_TUI_RESET) $msg")
end

"""
    tui_warning(msg::String)

Print a warning message.
"""
function tui_warning(msg::String)
    println("$(UNIFIED_TUI_YELLOW)⚠ $msg$(UNIFIED_TUI_RESET)")
end

"""
    tui_error(msg::String)

Print an error message.
"""
function tui_error(msg::String)
    println("$(UNIFIED_TUI_RED)✗ $msg$(UNIFIED_TUI_RESET)")
end

"""
    tui_info(msg::String)

Print an info message.
"""
function tui_info(msg::String)
    println("$(UNIFIED_TUI_DIM)ℹ $msg$(UNIFIED_TUI_RESET)")
end

"""
    tui_running(msg::String)

Print a "running" status message.
"""
function tui_running(msg::String)
    println()
    println("$(UNIFIED_TUI_CYAN)▶ $msg$(UNIFIED_TUI_RESET)")
    println()
end

"""
    tui_divider()

Print a simple divider line.
"""
function tui_divider()
    println("$(UNIFIED_TUI_DIM)────────────────────────────────────$(UNIFIED_TUI_RESET)")
end

# ============================================================================
# Results Root Discovery
# ============================================================================

"""
    find_unified_results_root() -> String

Find the results directory using environment variable or default locations.

Checks the GLOBTIM_RESULTS_ROOT environment variable.
"""
function find_unified_results_root()::String
    # Check environment variable first
    results_root = get(ENV, "GLOBTIM_RESULTS_ROOT", nothing)
    if results_root !== nothing && isdir(results_root)
        return results_root
    end

    error(
        "Could not find results directory. Set the GLOBTIM_RESULTS_ROOT environment variable or pass the path as an argument.",
    )
end

# ============================================================================
# Experiment Type Menu
# ============================================================================

"""
    select_experiment_type() -> Union{ExperimentType, Nothing}

Show menu to select experiment type.

Returns the selected ExperimentType singleton or nothing if cancelled.

Requires REPL.jl and PrettyTables.jl (package extension `GlobtimPostProcessingTUIExt`).
In the Julia REPL, REPL is already loaded and `using PrettyTables` suffices; in a script
add `using REPL, PrettyTables`.
"""
select_experiment_type(args...; kwargs...) =
    _missing_tui_extension(select_experiment_type, args)

# ============================================================================
# Results Source Menu
# ============================================================================

"""
    ResultsSource

Enum-like struct for results source selection.
"""
struct ResultsSource
    type::Symbol  # :recent, :path
    value::String
end

"""
    select_results_source(results_root::String) -> Union{ResultsSource, Nothing}

Show menu to select results source.

Requires REPL.jl and PrettyTables.jl (package extension `GlobtimPostProcessingTUIExt`).
In the Julia REPL, REPL is already loaded and `using PrettyTables` suffices; in a script
add `using REPL, PrettyTables`.
"""
select_results_source(args...; kwargs...) =
    _missing_tui_extension(select_results_source, args)

# ============================================================================
# Recent Experiments Menu
# ============================================================================

"""
    select_recent_experiment(results_root::String; type_filter::Union{ExperimentType, Nothing}=nothing, limit::Int=15) -> Union{String, Nothing}

Show menu to select from recent experiments.

# Arguments
- `results_root`: Root directory to search
- `type_filter`: Optional type filter (show only experiments of this type)
- `limit`: Maximum number of experiments to show

# Returns
Path to selected experiment or nothing if cancelled.

Requires REPL.jl and PrettyTables.jl (package extension `GlobtimPostProcessingTUIExt`).
In the Julia REPL, REPL is already loaded and `using PrettyTables` suffices; in a script
add `using REPL, PrettyTables`.
"""
select_recent_experiment(args...; kwargs...) =
    _missing_tui_extension(select_recent_experiment, args)

"""
    _format_age(seconds::Real) -> String

Format a time duration in human-readable form.
"""
function _format_age(seconds::Real)::String
    if seconds < 60
        return "$(round(Int, seconds))s"
    elseif seconds < 3600
        return "$(round(Int, seconds/60))m"
    elseif seconds < 86400
        return "$(round(Int, seconds/3600))h"
    else
        return "$(round(Int, seconds/86400))d"
    end
end

# ============================================================================
# Analysis Mode Menus
# ============================================================================

"""
    AnalysisMode

Represents an analysis mode with its metadata.
"""
struct AnalysisMode
    id::Symbol
    name::String
    description::String
    supported_types::Vector{DataType}  # Which ExperimentType subtypes support this mode
end

"""
Standard analysis modes available for all types.
"""
const COMMON_ANALYSIS_MODES = [
    AnalysisMode(:quality, "Quality", "Single experiment diagnostics", [ExperimentType]),
    AnalysisMode(
        :convergence,
        "Convergence",
        "Degree convergence analysis",
        [ExperimentType],
    ),
]

"""
    get_analysis_modes(type::ExperimentType) -> Vector{AnalysisMode}

Get available analysis modes for an experiment type.
"""
function get_analysis_modes(::ExperimentType)
    return COMMON_ANALYSIS_MODES
end

"""
    select_analysis_mode(type::ExperimentType) -> Union{Symbol, Nothing}

Show menu to select analysis mode for given experiment type.

Requires REPL.jl and PrettyTables.jl (package extension `GlobtimPostProcessingTUIExt`).
In the Julia REPL, REPL is already loaded and `using PrettyTables` suffices; in a script
add `using REPL, PrettyTables`.
"""
select_analysis_mode(args...; kwargs...) =
    _missing_tui_extension(select_analysis_mode, args)

# ============================================================================
# Confirmation Menu
# ============================================================================

"""
    confirm_selection(summary::String) -> Bool

Show confirmation with summary of selections.

Requires REPL.jl and PrettyTables.jl (package extension `GlobtimPostProcessingTUIExt`).
In the Julia REPL, REPL is already loaded and `using PrettyTables` suffices; in a script
add `using REPL, PrettyTables`.
"""
confirm_selection(args...; kwargs...) = _missing_tui_extension(confirm_selection, args)
