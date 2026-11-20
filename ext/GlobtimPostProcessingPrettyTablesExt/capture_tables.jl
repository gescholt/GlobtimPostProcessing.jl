# Table printers moved from src/refinement/capture_analysis.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function print_capture_summary(
    result::CaptureResult,
    known::KnownCriticalPoints;
    io::IO = stdout,
)
    n_tol = length(result.tolerance_fractions)

    # --- Table 1: Overall capture rates ---
    overall_data = Matrix{Any}(undef, n_tol, 4)
    for t in 1:n_tol
        n_captured = sum(result.captured_at[t])
        overall_data[t, 1] = @sprintf("%.1f%%", result.tolerance_fractions[t] * 100)
        overall_data[t, 2] = @sprintf("%.4f", result.tolerance_values[t])
        overall_data[t, 3] = "$n_captured / $(result.n_known)"
        overall_data[t, 4] = @sprintf("%.1f%%", result.capture_rates[t] * 100)
    end

    println(io)
    styled_table(
        io,
        overall_data;
        header = ["Tol (frac)", "Tol (abs)", "Captured", "Rate"],
        title = "Capture Analysis ($(result.n_computed) computed vs $(result.n_known) known CPs, domain diam = $(@sprintf("%.4f", result.domain_diameter)))",
        alignment = [:r, :r, :c, :r],
    )

    # --- Table 2: Per-type capture rates ---
    unique_types = sort(collect(keys(result.type_counts)))
    if length(unique_types) > 1
        type_header = ["Tol (frac)"]
        for tp in unique_types
            push!(type_header, "$(tp) ($(result.type_counts[tp]))")
        end

        type_data = Matrix{Any}(undef, n_tol, 1 + length(unique_types))
        for t in 1:n_tol
            type_data[t, 1] = @sprintf("%.1f%%", result.tolerance_fractions[t] * 100)
            for (j, tp) in enumerate(unique_types)
                rate = result.type_capture_rates[tp][t]
                type_data[t, 1+j] = @sprintf("%.1f%%", rate * 100)
            end
        end

        println(io)
        styled_table(
            io,
            type_data;
            header = type_header,
            title = "Per-Type Capture Rates",
            alignment = :r,
        )
    end

    # --- Table 3: Missed CPs at largest tolerance ---
    missed = missed_critical_points(result, known; tolerance_index = n_tol)
    if !isempty(missed)
        n_dims = length(known.points[1])
        missed_header = ["#", "Type", "f(x)", "Nearest dist"]
        for d in 1:n_dims
            push!(missed_header, "x$d")
        end

        max_display = 20
        display_missed = length(missed) <= max_display ? missed : missed[1:max_display]

        missed_data = Matrix{Any}(undef, length(display_missed), 4 + n_dims)
        for (i, m) in enumerate(display_missed)
            missed_data[i, 1] = m.index
            missed_data[i, 2] = string(m.type)
            missed_data[i, 3] = @sprintf("%.6f", m.value)
            missed_data[i, 4] = @sprintf("%.6f", m.nearest_distance)
            for d in 1:n_dims
                missed_data[i, 4+d] = @sprintf("%.6f", m.point[d])
            end
        end

        n_total_missed = length(missed)
        title_suffix =
            n_total_missed > max_display ?
            ", showing first $max_display of $n_total_missed" : ""
        println(io)
        styled_table(
            io,
            missed_data;
            header = missed_header,
            title = "Missed Critical Points (at tol = $(@sprintf("%.4f", result.tolerance_values[n_tol]))$title_suffix)",
            alignment = :r,
        )
    else
        println(io)
        println(
            io,
            "All $(result.n_known) known critical points captured at tolerance $(@sprintf("%.4f", result.tolerance_values[n_tol]))",
        )
    end
end

function print_degree_capture_convergence(
    degree_results::Vector{Tuple{Int,CaptureResult}};
    io::IO = stdout,
)
    isempty(degree_results) && error("degree_results must be non-empty")

    # Validate that all results used the same tolerance fractions
    tol_fracs = degree_results[1][2].tolerance_fractions
    for (i, (deg, res)) in enumerate(degree_results)
        res.tolerance_fractions == tol_fracs || error(
            "Inconsistent tolerance_fractions: degree $(deg) (entry $i) has " *
            "$(res.tolerance_fractions) but expected $tol_fracs (from first entry)",
        )
    end

    header = ["Degree", "# Computed"]
    for tf in tol_fracs
        push!(header, @sprintf("rate@%.1f%%", tf * 100))
    end

    n_rows = length(degree_results)
    n_cols = 2 + length(tol_fracs)
    data = Matrix{Any}(undef, n_rows, n_cols)

    for (i, (deg, res)) in enumerate(degree_results)
        data[i, 1] = deg
        data[i, 2] = res.n_computed
        for (t, rate) in enumerate(res.capture_rates)
            data[i, 2+t] = @sprintf("%.1f%%", rate * 100)
        end
    end

    println(io)
    styled_table(
        io,
        data;
        header = header,
        title = "Capture Rate vs Polynomial Degree",
        alignment = :r,
    )
end

function print_degree_convergence_summary(
    degree_capture_results::Vector{Tuple{Int,CaptureResult}},
    degree_info::Vector{DegreeConvergenceInfo};
    io::IO = stdout,
)
    isempty(degree_capture_results) && error("degree_capture_results must be non-empty")
    isempty(degree_info) && error("degree_info must be non-empty")

    # Build lookup maps
    cr_map = Dict(deg => cr for (deg, cr) in degree_capture_results)
    info_map = Dict(di.degree => di for di in degree_info)
    all_degrees = sort(unique([d for (d, _) in degree_capture_results]))

    # Determine tolerance indices for 1%, 5%, 10%
    ref_cr = degree_capture_results[1][2]
    tol_fracs = ref_cr.tolerance_fractions

    idx_1pct = findfirst(f -> f ≈ 0.01, tol_fracs)
    idx_5pct = findfirst(f -> f ≈ 0.05, tol_fracs)
    idx_10pct = findfirst(f -> f ≈ 0.1, tol_fracs)

    n_rows = length(all_degrees)
    conv_data = Matrix{Any}(undef, n_rows, 9)

    for (row, deg) in enumerate(all_degrees)
        di = get(info_map, deg, nothing)
        cr = get(cr_map, deg, nothing)

        conv_data[row, 1] = deg
        conv_data[row, 2] = di !== nothing ? @sprintf("%.*e", 2, di.l2_error) : "N/A"
        conv_data[row, 3] =
            di !== nothing && !isnan(di.relative_l2_error) ?
            @sprintf("%.*e", 2, di.relative_l2_error) : "N/A"
        conv_data[row, 4] = di !== nothing ? di.n_critical_points : 0
        conv_data[row, 5] =
            di !== nothing && di.min_gradient_norm !== nothing ?
            @sprintf("%.*e", 2, di.min_gradient_norm) : "N/A"

        # Capture rates at 1%, 5%, 10%
        if cr !== nothing
            conv_data[row, 6] =
                idx_1pct !== nothing ?
                @sprintf("%.1f%%", 100 * cr.capture_rates[idx_1pct]) : "N/A"
            conv_data[row, 7] =
                idx_5pct !== nothing ?
                @sprintf("%.1f%%", 100 * cr.capture_rates[idx_5pct]) : "N/A"
            conv_data[row, 8] =
                idx_10pct !== nothing ?
                @sprintf("%.1f%%", 100 * cr.capture_rates[idx_10pct]) : "N/A"
        else
            conv_data[row, 6] = "N/A"
            conv_data[row, 7] = "N/A"
            conv_data[row, 8] = "N/A"
        end

        # Best objective
        if di !== nothing && di.best_objective !== nothing
            conv_data[row, 9] = @sprintf("%.*e", 2, di.best_objective)
        else
            conv_data[row, 9] = "N/A"
        end
    end

    # Highlight best row by capture rate at 5%
    capture_5pct = Float64[]
    for deg in all_degrees
        cr = get(cr_map, deg, nothing)
        if cr !== nothing && idx_5pct !== nothing
            push!(capture_5pct, cr.capture_rates[idx_5pct])
        else
            push!(capture_5pct, 0.0)
        end
    end
    best_conv_row = argmax(capture_5pct)

    hl_deg = Highlighter((_, i, j) -> j == 1, bold = true, foreground = :cyan)
    hl_best_conv = Highlighter(
        (_, i, j) -> i == best_conv_row && (j >= 6 && j <= 8),
        foreground = :green,
        bold = true,
    )

    println(io)
    styled_table(
        io,
        conv_data;
        header = [
            "Deg",
            "L2 err",
            "Rel L2",
            "# CPs",
            "min ||∇f||",
            "Cap @1%",
            "Cap @5%",
            "Cap @10%",
            "Best f(x)",
        ],
        title = "Degree Convergence Summary",
        alignment = [:r, :r, :r, :r, :r, :r, :r, :r, :r],
        highlighters = (hl_deg, hl_best_conv),
    )
end
