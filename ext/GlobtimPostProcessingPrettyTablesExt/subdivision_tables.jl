# Table printers moved from src/SubdivisionTreeAnalysis.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function print_subdivision_tree_report(tree; io::IO = stdout)
    stats = analyze_subdivision_tree(tree)

    println(io, "="^70)
    println(io, "SUBDIVISION TREE ANALYSIS REPORT")
    println(io, "="^70)
    println(io)

    # Structure summary
    println(io, "STRUCTURE")
    println(io, "-"^70)
    @printf(io, "  Dimension:           %d\n", stats.dimension)
    @printf(io, "  Total leaves:        %d\n", stats.n_leaves)
    @printf(
        io,
        "  Converged:           %d (%.1f%%)\n",
        stats.n_converged,
        stats.convergence_rate * 100
    )
    @printf(io, "  Active:              %d\n", stats.n_active)
    @printf(io, "  Max depth:           %d\n", stats.max_depth)
    @printf(io, "  Mean depth:          %.2f\n", stats.mean_depth)
    println(io)

    # L2 error summary
    println(io, "L2 APPROXIMATION ERROR")
    println(io, "-"^70)
    @printf(io, "  Total:               %.4e\n", stats.total_l2_error)
    @printf(io, "  Mean:                %.4e\n", stats.mean_l2_error)
    @printf(io, "  Std:                 %.4e\n", stats.std_l2_error)
    @printf(io, "  Min:                 %.4e\n", stats.min_l2_error)
    @printf(io, "  Max:                 %.4e\n", stats.max_l2_error)
    @printf(io, "  Median:              %.4e\n", stats.median_l2_error)
    println(io)

    # Quantiles
    println(io, "  Quantiles:")
    @printf(io, "    10th:              %.4e\n", stats.error_quantiles.q10)
    @printf(io, "    25th:              %.4e\n", stats.error_quantiles.q25)
    @printf(io, "    50th:              %.4e\n", stats.error_quantiles.q50)
    @printf(io, "    75th:              %.4e\n", stats.error_quantiles.q75)
    @printf(io, "    90th:              %.4e\n", stats.error_quantiles.q90)
    println(io)

    # Depth distribution table
    println(io, "DEPTH DISTRIBUTION")
    println(io, "-"^70)
    depth_df = tree_depth_summary(tree)

    styled_table(
        io,
        depth_df;
        header = [
            "Depth",
            "Leaves",
            "Converged",
            "Mean L2",
            "Std L2",
            "Min L2",
            "Max L2",
            "Volume",
        ],
        formatters = (ft_printf("%.2e", [4, 5, 6, 7]), ft_printf("%.3f", [8])),
        alignment = [:c, :c, :c, :r, :r, :r, :r, :r],
    )

    println(io)
    println(io, "="^70)

    nothing
end

function print_tree_comparison_table(trees::Vector; labels = nothing, io::IO = stdout)
    df = compare_subdivision_trees(trees; labels = labels)

    println(io, "SUBDIVISION TREE COMPARISON")
    println(io, "="^70)

    styled_table(
        io,
        df;
        header = [
            "Label",
            "Leaves",
            "Conv",
            "Conv%",
            "Depth",
            "Mean L2",
            "Total L2",
            "Dim",
        ],
        formatters = (ft_printf("%.1f%%", [4]), ft_printf("%.2e", [6, 7])),
        alignment = [:l, :c, :c, :c, :c, :r, :r, :c],
    )

    nothing
end
