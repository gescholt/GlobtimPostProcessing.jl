# tree_display.jl
# Colored subdivision tree visualization using AbstractTrees + printstyled

using AbstractTrees
using Printf: @sprintf
using Globtim: SubdivisionTree, Subdomain, total_error, n_leaves, n_active, get_max_depth, volume

#=============================================================================
# Configuration
=============================================================================#

"""
    TreeDisplayConfig

Configuration for text-based tree display.
"""
Base.@kwdef struct TreeDisplayConfig
    show_split_position::Bool = true    # Show ASCII [---|------] visualization
    show_error_reduction::Bool = true   # Show n%↓ or n%↑ after splits
    show_depth::Bool = false            # Show depth indicator
    error_sigfigs::Int = 2              # Significant figures for error display
    split_viz_width::Int = 10           # Width of split position ASCII bar
    max_display_depth::Union{Int,Nothing} = nothing  # Truncate display at this depth
    compact::Bool = false               # Use compact one-line-per-leaf format
    use_colors::Bool = true             # Enable colored output
end

# Default configuration
const DEFAULT_CONFIG = TreeDisplayConfig()

#=============================================================================
# Tracker utility functions (pure, testable)
=============================================================================#

"""
    progress_bar(current, max, width=10) -> String

Create ASCII progress bar: "█████░░░░░" for 50% progress.
"""
function progress_bar(current::Int, max::Int, width::Int=10)
    progress = clamp(current / max, 0.0, 1.0)
    filled = round(Int, progress * width)
    return "█"^filled * "░"^(width - filled)
end

"""
    format_error(errors) -> String

Format error sum, returning "?" if no finite errors.
"""
function format_error(errors::Vector{Float64})
    finite = filter(isfinite, errors)
    isempty(finite) ? "?" : string(round(sum(finite), sigdigits=3))
end

"""
    format_time(seconds) -> String

Format elapsed time as "5.5s" or "1.5m".
"""
function format_time(seconds::Float64)
    seconds < 60 ? "$(round(seconds, digits=1))s" : "$(round(seconds/60, digits=1))m"
end

"""
    compute_display_error(tree) -> Float64

Compute error for display, handling newly-created leaves with Inf error
by using parent's error as estimate.
"""
function compute_display_error(tree::SubdivisionTree)
    total = 0.0
    for id in tree.converged_leaves
        total += tree.subdomains[id].l2_error
    end
    for id in tree.active_leaves
        err = tree.subdomains[id].l2_error
        if isfinite(err)
            total += err
        else
            # Use parent's error / 2 as estimate for this leaf
            parent_id = tree.subdomains[id].parent_id
            if parent_id !== nothing
                total += tree.subdomains[parent_id].l2_error / 2
            end
        end
    end
    return total
end

# Tree node wrapper for AbstractTrees interface
struct TreeNode
    tree::SubdivisionTree
    id::Int
end

AbstractTrees.children(n::TreeNode) = begin
    sd = n.tree.subdomains[n.id]
    sd.children === nothing ? TreeNode[] : [TreeNode(n.tree, sd.children[1]), TreeNode(n.tree, sd.children[2])]
end

# Dimension colors (cycle through for >6 dimensions)
const DIM_COLORS = [:red, :green, :blue, :yellow, :magenta, :cyan]
get_dim_color(dim::Int) = DIM_COLORS[mod1(dim, length(DIM_COLORS))]

# ASCII split position: [----|---------] shows where split occurs in [-1,1]
function split_position_viz(split_pos::Float64, width::Int=10)
    pos = round(Int, (split_pos + 1) / 2 * (width - 1))
    pos = clamp(pos, 0, width - 1)
    return "[" * "-"^pos * "|" * "-"^(width - 1 - pos) * "]"
end

# Error reduction from parent to children (percentage)
function get_error_reduction(tree, node_id::Int)
    sd = tree.subdomains[node_id]
    sd.children === nothing && return nothing
    left_id, right_id = sd.children
    left_err = tree.subdomains[left_id].l2_error
    right_err = tree.subdomains[right_id].l2_error
    children_total = left_err + right_err
    if sd.l2_error > 0 && sd.l2_error != Inf
        return (sd.l2_error - children_total) / sd.l2_error * 100
    end
    return nothing
end

# Node display for print_tree()
AbstractTrees.printnode(io::IO, n::TreeNode) = begin
    sd = n.tree.subdomains[n.id]
    if sd.split_dim !== nothing
        dim = sd.split_dim
        color = get_dim_color(dim)
        viz = split_position_viz(sd.split_pos)

        reduction = get_error_reduction(n.tree, n.id)
        if reduction !== nothing
            if reduction >= 0
                reduction_str = " $(round(Int, reduction))%↓"
                reduction_color = :cyan
            else
                reduction_str = " $(round(Int, -reduction))%↑"
                reduction_color = :red
            end
        else
            reduction_str = ""
            reduction_color = :white
        end

        printstyled(io, "x$(dim)", color=color)
        print(io, " $viz")
        printstyled(io, reduction_str, color=reduction_color)
    else
        if n.id in n.tree.converged_leaves
            printstyled(io, "✓", color=:green)
        else
            printstyled(io, "○", color=:yellow)
        end
        print(io, " [$(round(sd.l2_error, sigdigits=2))]")
    end
end

#=============================================================================
# Live progress trackers
=============================================================================#

"""
    count_splits_per_dim(tree) -> Dict{Int,Int}

Count how many splits occurred in each dimension.
"""
function count_splits_per_dim(tree::SubdivisionTree)
    counts = Dict{Int,Int}()
    for sd in tree.subdomains
        if sd.split_dim !== nothing
            counts[sd.split_dim] = get(counts, sd.split_dim, 0) + 1
        end
    end
    return counts
end

"""
    make_tracker(; max_leaves=50, show_dims=true) -> callback

Create a live progress tracker callback for adaptive_refine with progress bar and timing.

Output format: `[5] [████░░░░░░] 16/50 d=3 (4✓ 12○) [x1:3 x2:5] err=0.342 [1.2s]`

When `show_dims=true`, includes split dimension distribution.

Usage:
    tree = adaptive_refine(...,
        iteration_callback = make_tracker(max_leaves=50),
        ...
    )
"""
function make_tracker(; max_leaves::Int=50, show_dims::Bool=true)
    start_time = Ref(0.0)

    return function(tree, iteration)
        # Initialize timer on first call
        iteration == 1 && (start_time[] = time())

        leaves = n_leaves(tree)
        active = n_active(tree)
        conv = leaves - active
        depth = get_max_depth(tree)

        # Progress bar
        bar = progress_bar(leaves, max_leaves)

        # Split dimension distribution
        dim_str = ""
        if show_dims
            counts = count_splits_per_dim(tree)
            if !isempty(counts)
                parts = ["x$d:$c" for (d, c) in sort(collect(counts))]
                dim_str = " [" * join(parts, " ") * "]"
            end
        end

        # Error: hybrid sum (uses parent error for newly-created Inf leaves)
        err_val = compute_display_error(tree)
        err_str = isfinite(err_val) ? string(round(err_val, sigdigits=3)) : "?"

        # Elapsed time
        time_str = format_time(time() - start_time[])

        print("\r\e[K")
        print("[$iteration] [$bar] $leaves/$max_leaves d=$depth ($(conv)✓ $(active)○)$dim_str err=$err_str [$time_str]")
        flush(stdout)
    end
end

"""
    live_tracker(tree, iteration)

Simple live progress tracker (backward compatible, no progress bar or timing).
"""
function live_tracker(tree, iteration)
    depth = get_max_depth(tree)
    leaves = n_leaves(tree)
    active = n_active(tree)
    conv = leaves - active

    # Error: hybrid sum (uses parent error for newly-created Inf leaves)
    err_val = compute_display_error(tree)
    err_str = isfinite(err_val) ? string(round(err_val, sigdigits=3)) : "?"

    print("\r\e[K")
    print("[$iteration] d=$depth leaves=$leaves ($(conv)✓ $(active)○) err=$err_str")
    flush(stdout)
end

# Print summary after tree construction
function print_summary(tree::SubdivisionTree)
    n_total = n_leaves(tree)
    n_conv = length(tree.converged_leaves)
    n_act = length(tree.active_leaves)

    # Count splits per dimension
    split_counts = Dict{Int,Int}()
    for sd in tree.subdomains
        if sd.split_dim !== nothing
            split_counts[sd.split_dim] = get(split_counts, sd.split_dim, 0) + 1
        end
    end

    println("\nSummary: $n_total leaves ($n_conv converged, $n_act active)")
    if !isempty(split_counts)
        dims_str = join(["x$d=$(split_counts[d])" for d in sort(collect(keys(split_counts)))], " ")
        println("Splits: $dims_str")
    end
    println("Total L2 error: $(round(total_error(tree), sigdigits=4))")
end

# Print legend for dimension colors
function print_legend(n_dims::Int)
    println("\nLegend:")
    for i in 1:min(n_dims, 6)
        printstyled("  x$i", color=get_dim_color(i))
        println(" = Dimension $i")
    end
    printstyled("  ✓", color=:green); println(" = converged")
    printstyled("  ○", color=:yellow); println(" = active")
end

#=============================================================================
# Compact tree display (one line per leaf)
=============================================================================#

"""
    get_path_to_root(tree, node_id) -> Vector{Int}

Get path from node to root (returns [root_id, ..., node_id]).
"""
function get_path_to_root(tree::SubdivisionTree, node_id::Int)
    path = [node_id]
    current = node_id
    while tree.subdomains[current].parent_id !== nothing
        current = tree.subdomains[current].parent_id
        pushfirst!(path, current)
    end
    return path
end

"""
    format_path_compact(tree, path; config=DEFAULT_CONFIG) -> String

Format a path from root to leaf as a compact string.
Example: "x1@-0.3 → x2@0.1 → ✓[0.023]"
"""
function format_path_compact(tree::SubdivisionTree, path::Vector{Int}; config::TreeDisplayConfig=DEFAULT_CONFIG)
    parts = String[]

    for (i, id) in enumerate(path)
        sd = tree.subdomains[id]

        if i == length(path)
            # Leaf node
            symbol = id in tree.converged_leaves ? "✓" : "○"
            err_str = round(sd.l2_error, sigdigits=config.error_sigfigs)
            push!(parts, "$symbol[$err_str]")
        else
            # Internal node - show split info
            if sd.split_dim !== nothing
                pos_str = @sprintf("%.1f", sd.split_pos)
                push!(parts, "x$(sd.split_dim)@$pos_str")
            end
        end
    end

    return join(parts, " → ")
end

"""
    print_tree_compact(tree; config=DEFAULT_CONFIG, io=stdout)

Print tree in compact format: one line per leaf showing the path from root.

Example output:
```
x1@-0.3 → x2@0.1 → ✓[0.023]
x1@-0.3 → x2@0.1 → x3@0.5 → ○[0.045]
x1@0.2 → ✓[0.012]
```
"""
function print_tree_compact(tree::SubdivisionTree; config::TreeDisplayConfig=DEFAULT_CONFIG, io::IO=stdout)
    # Collect all leaf IDs and sort by error (largest first)
    all_leaves = vcat(collect(tree.converged_leaves), collect(tree.active_leaves))
    sort!(all_leaves, by=id -> -tree.subdomains[id].l2_error)

    for leaf_id in all_leaves
        path = get_path_to_root(tree, leaf_id)
        line = format_path_compact(tree, path; config=config)

        if config.use_colors
            # Color the whole line based on leaf status
            color = leaf_id in tree.converged_leaves ? :green : :yellow
            printstyled(io, line, "\n", color=color)
        else
            println(io, line)
        end
    end
end

#=============================================================================
# Large tree summarization
=============================================================================#

"""
    print_tree_summary_only(tree; top_n=10, io=stdout)

Print a summary view for large trees, showing statistics and top-N leaves by error.

Use this when `n_leaves(tree) > 30` and full tree display would be unwieldy.
"""
function print_tree_summary_only(tree::SubdivisionTree; top_n::Int=10, io::IO=stdout)
    n_total = n_leaves(tree)
    n_conv = length(tree.converged_leaves)
    n_act = length(tree.active_leaves)
    depth = get_max_depth(tree)

    # Header
    println(io, "Tree Summary ($n_total leaves)")
    println(io, "=" ^ 50)

    # Basic stats
    println(io, "Leaves: $n_conv converged, $n_act active")
    println(io, "Max depth: $depth")

    # Dimension split distribution
    split_counts = count_splits_per_dim(tree)
    if !isempty(split_counts)
        dims_str = join(["x$d=$c" for (d, c) in sort(collect(split_counts))], " ")
        println(io, "Splits: $dims_str")
    end

    # Total error
    total_err = total_error(tree)
    println(io, "Total L2 error: $(round(total_err, sigdigits=4))")

    # Top N leaves by error contribution
    println(io)
    println(io, "Top $top_n leaves by error:")
    println(io, "-" ^ 50)

    all_leaves = vcat(collect(tree.converged_leaves), collect(tree.active_leaves))
    sort!(all_leaves, by=id -> -tree.subdomains[id].l2_error)

    for (i, leaf_id) in enumerate(all_leaves[1:min(top_n, length(all_leaves))])
        sd = tree.subdomains[leaf_id]
        status = leaf_id in tree.converged_leaves ? "✓" : "○"
        err_pct = total_err > 0 ? round(sd.l2_error / total_err * 100, digits=1) : 0.0
        path = get_path_to_root(tree, leaf_id)
        path_str = join(["x$(tree.subdomains[id].split_dim)" for id in path[1:end-1] if tree.subdomains[id].split_dim !== nothing], "→")

        println(io, "  $i. $status err=$(round(sd.l2_error, sigdigits=3)) ($err_pct%) depth=$(sd.depth) path: $path_str")
    end

    # Show remaining error if truncated
    if length(all_leaves) > top_n
        remaining = length(all_leaves) - top_n
        remaining_err = sum(tree.subdomains[id].l2_error for id in all_leaves[top_n+1:end])
        remaining_pct = total_err > 0 ? round(remaining_err / total_err * 100, digits=1) : 0.0
        println(io, "  ... and $remaining more leaves ($remaining_pct% of error)")
    end

    # Volume distribution
    print_volume_distribution(tree; io=io)
end

"""
    print_tree_auto(tree; compact_threshold=20, summary_threshold=40, config=DEFAULT_CONFIG, io=stdout)

Automatically choose the best display format based on tree size:
- ≤ compact_threshold leaves: Full tree (print_tree)
- ≤ summary_threshold leaves: Compact format (print_tree_compact)
- > summary_threshold leaves: Summary only (print_tree_summary_only)
"""
function print_tree_auto(tree::SubdivisionTree;
                         compact_threshold::Int=20,
                         summary_threshold::Int=40,
                         config::TreeDisplayConfig=DEFAULT_CONFIG,
                         io::IO=stdout)
    n = n_leaves(tree)

    if n <= compact_threshold
        print_tree(TreeNode(tree, tree.root_id))
    elseif n <= summary_threshold
        println(io, "Compact view ($n leaves):")
        print_tree_compact(tree; config=config, io=io)
    else
        print_tree_summary_only(tree; io=io)
    end

    print_summary(tree)
end

#=============================================================================
# Volume distribution display
=============================================================================#

"""
    collect_leaf_volumes(tree) -> (all_volumes, active_volumes, converged_volumes)

Collect volumes for all leaves, separated by status.
"""
function collect_leaf_volumes(tree::SubdivisionTree)
    all_volumes = Float64[]
    active_volumes = Float64[]
    converged_volumes = Float64[]

    for id in tree.active_leaves
        v = volume(tree.subdomains[id])
        push!(all_volumes, v)
        push!(active_volumes, v)
    end
    for id in tree.converged_leaves
        v = volume(tree.subdomains[id])
        push!(all_volumes, v)
        push!(converged_volumes, v)
    end

    return (all_volumes, active_volumes, converged_volumes)
end

"""
    print_histogram(values, bins; bar_width=20, io=stdout)

Print ASCII histogram for given values and bin edges.
"""
function print_histogram(values::Vector{Float64}, bins::Vector{Float64};
                         bar_width::Int=20, io::IO=stdout)
    n = length(values)
    for i in 1:(length(bins)-1)
        lo, hi = bins[i], bins[i+1]
        count = sum(lo .<= values .< hi)
        count == 0 && continue  # Skip empty bins
        pct = n > 0 ? count / n * 100 : 0.0
        bar_len = round(Int, pct / 100 * bar_width)
        bar = "█"^bar_len * "░"^(bar_width - bar_len)

        # Format bin range as percentage
        range_str = @sprintf("[%5.1f%%-%5.1f%%]", lo * 100, hi * 100)
        count_str = @sprintf("%3d (%4.1f%%)", count, pct)

        println(io, "  $(rpad(range_str, 16)) $bar $count_str")
    end
end

"""
    print_volume_distribution(tree; io=stdout)

Print volume distribution of leaves as ASCII histogram, showing volumes as
fraction of root domain volume.

Example output:
```
Volume Distribution (50 leaves)
────────────────────────────────────────
All leaves:
  [  0.1%-  1.0%]   ████████████████████  35 (70.0%)
  [  1.0%- 10.0%]   ████████              12 (24.0%)
  [ 10.0%-100.0%]   ██                     3 (6.0%)

Active (not converged):
  [  0.1%-  1.0%]   ████████████████████   8 (100.0%)
```
"""
function print_volume_distribution(tree::SubdivisionTree; io::IO=stdout)
    all_vols, active_vols, _ = collect_leaf_volumes(tree)
    root_vol = volume(tree.subdomains[tree.root_id])

    # Convert to fractions of root volume
    vol_ratios = all_vols ./ root_vol
    min_ratio = minimum(vol_ratios)

    # Create bins in powers of 10 (10^0 = 1.0 is already included)
    min_exp = floor(Int, log10(min_ratio))
    bins = [10.0^i for i in min_exp:0]

    println(io, "\nVolume Distribution ($(n_leaves(tree)) leaves)")
    println(io, "─"^50)

    # All leaves
    println(io, "All leaves:")
    print_histogram(vol_ratios, bins, io=io)

    # Active leaves (if any)
    if !isempty(active_vols)
        println(io, "\nActive (not converged):")
        active_ratios = active_vols ./ root_vol
        print_histogram(active_ratios, bins, io=io)
    end
end
