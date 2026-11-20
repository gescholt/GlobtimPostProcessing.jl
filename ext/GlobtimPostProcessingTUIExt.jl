"""
    GlobtimPostProcessingTUIExt

Interactive terminal interface of GlobtimPostProcessing (`postprocess()` and the
`select_*` / `confirm_selection` menus), loaded when both REPL and PrettyTables are
loaded. In the Julia REPL, REPL is already loaded, so `using PrettyTables` suffices.

`GlobtimPostProcessing.UnifiedPipeline` defines each function as a documented catch-all
stub that errors with "f requires REPL.jl and PrettyTables.jl"; the methods here are
more specific. The non-interactive loaders (`load_experiment`, `detect_experiment_type`,
...) stay in the package itself.
"""
module GlobtimPostProcessingTUIExt

using GlobtimPostProcessing
using GlobtimPostProcessing.UnifiedPipeline:
    ExperimentType,
    SUPPORTED_TYPES,
    ResultsSource,
    type_name,
    detect_experiment_type,
    is_single_experiment,
    find_unified_results_root,
    get_analysis_modes,
    tui_header,
    tui_info,
    tui_warning,
    tui_success,
    tui_divider,
    _format_age,
    _resolve_experiment_path,
    _run_unified_analysis
import GlobtimPostProcessing.UnifiedPipeline:
    postprocess,
    select_experiment_type,
    select_results_source,
    select_recent_experiment,
    select_analysis_mode,
    confirm_selection

using REPL.TerminalMenus: RadioMenu, request
# PrettyTables is a trigger because the analysis modes print tables through
# GlobtimPostProcessing.styled_table, whose methods come from the PrettyTables extension.
import PrettyTables

# from src/unified/tui_main.jl
function postprocess(; results_root::Union{String,Nothing} = nothing)
    # Get results root
    root = results_root === nothing ? find_unified_results_root() : results_root

    tui_header("UNIFIED POST-PROCESSING")

    # Step 1: Select experiment type
    exp_type = select_experiment_type()

    # Step 2: Select results source and get experiment path
    source = select_results_source(root)
    source === nothing && return nothing

    exp_path = _resolve_experiment_path(source, root, exp_type)
    exp_path === nothing && return nothing

    # Auto-detect type if needed
    if exp_type === nothing
        exp_type = detect_experiment_type(exp_path)
        tui_info("Detected experiment type: $(type_name(exp_type))")
    end

    # Step 3: Select analysis mode
    analysis_mode = select_analysis_mode(exp_type)
    analysis_mode === nothing && return nothing

    # Step 4 & 5: Run analysis (dispatched by type and mode)
    return _run_unified_analysis(exp_type, analysis_mode, exp_path, root)
end

# from src/unified/tui_menus.jl
function select_experiment_type()::Union{ExperimentType,Nothing}
    # Build options from SUPPORTED_TYPES
    options = String[]
    for (typ, desc) in SUPPORTED_TYPES
        push!(options, "$(type_name(typ)) - $desc")
    end
    push!(options, "Auto-detect from path")

    menu = RadioMenu(options, pagesize = min(8, length(options)))
    choice = request("Select experiment type:", menu)

    choice == -1 && return nothing

    # Handle auto-detect option
    if choice == length(options)
        return nothing  # Signal auto-detect
    end

    return SUPPORTED_TYPES[choice][1]
end

# from src/unified/tui_menus.jl
function select_results_source(results_root::String)::Union{ResultsSource,Nothing}
    options = ["Browse recent experiments", "Enter path manually"]

    menu = RadioMenu(options, pagesize = 2)
    choice = request("Select results source:", menu)

    choice == -1 && return nothing

    if choice == 1
        return ResultsSource(:recent, results_root)
    else
        print("Enter experiment path: ")
        path = strip(readline())
        isempty(path) && return nothing
        return ResultsSource(:path, path)
    end
end

# from src/unified/tui_menus.jl
function select_recent_experiment(
    results_root::String;
    type_filter::Union{ExperimentType,Nothing} = nothing,
    limit::Int = 15,
)::Union{String,Nothing}
    # Build list of experiments
    experiments = String[]

    # Walk through results_root looking for experiment directories
    for entry in readdir(results_root, join = true)
        if isdir(entry) && is_single_experiment(entry)
            # Apply type filter if specified
            if type_filter !== nothing
                detected = detect_experiment_type(entry)
                typeof(detected) != typeof(type_filter) && continue
            end
            push!(experiments, entry)
        end

        # Also check subdirectories (e.g., lotka_volterra_4d/)
        if isdir(entry)
            for subentry in readdir(entry, join = true)
                if isdir(subentry) && is_single_experiment(subentry)
                    if type_filter !== nothing
                        detected = detect_experiment_type(subentry)
                        typeof(detected) != typeof(type_filter) && continue
                    end
                    push!(experiments, subentry)
                end
            end
        end
    end

    if isempty(experiments)
        tui_warning("No experiments found in: $results_root")
        return nothing
    end

    # Sort by modification time (most recent first)
    sort!(experiments, by = mtime, rev = true)
    experiments = experiments[1:min(limit, length(experiments))]

    # Build options with age info
    options = map(experiments) do exp
        name = basename(exp)
        age = time() - mtime(exp)
        age_str = _format_age(age)
        typ = detect_experiment_type(exp)
        type_str = type_name(typ)
        "[$type_str] $name ($age_str ago)"
    end

    menu = RadioMenu(options, pagesize = min(12, length(options)))
    choice = request("Select experiment:", menu)

    choice == -1 && return nothing
    return experiments[choice]
end

# from src/unified/tui_menus.jl
function select_analysis_mode(type::ExperimentType)::Union{Symbol,Nothing}
    modes = get_analysis_modes(type)

    options = ["$(m.name) - $(m.description)" for m in modes]
    menu = RadioMenu(options, pagesize = min(8, length(options)))
    choice = request("Select analysis mode:", menu)

    choice == -1 && return nothing
    return modes[choice].id
end

# from src/unified/tui_menus.jl
function confirm_selection(summary::String)::Bool
    println()
    tui_divider()
    tui_success(summary)
    tui_divider()
    println()

    options = ["Yes - Run analysis", "No - Cancel"]
    menu = RadioMenu(options, pagesize = 2)
    choice = request("Proceed?", menu)

    return choice == 1
end

end # module GlobtimPostProcessingTUIExt
