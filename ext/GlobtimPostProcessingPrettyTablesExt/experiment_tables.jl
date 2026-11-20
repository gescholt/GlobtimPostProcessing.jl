# Table printers moved from src/experiment_display.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function print_experiment_header(
    title::String,
    setup_pairs::Matrix{String};
    io::IO = stdout,
)
    W = 80
    printstyled(io, "="^W * "\n"; color = :blue, bold = true)
    printstyled(io, " " * title * "\n"; color = :white, bold = true)
    printstyled(io, "="^W * "\n"; color = :blue, bold = true)

    pretty_table(
        io,
        setup_pairs;
        show_header = false,
        tf = tf_borderless,
        alignment = [:r, :l],
        highlighters = (Highlighter((_, i, j) -> j == 1, bold = true, foreground = :cyan),),
        vlines = [1],
    )
end

function print_poly_summary_table(
    degree_results;
    show_timing_breakdown::Bool = false,
    io::IO = stdout,
)
    print_section("Polynomial Approximation (Globtim)"; io = io)

    # Check if degree_results have relative_l2_error field
    has_rel_l2 =
        !isempty(degree_results) && hasproperty(first(degree_results), :relative_l2_error)

    n = length(degree_results)
    base_cols = has_rel_l2 ? 7 : 6
    n_cols = show_timing_breakdown ? base_cols + 2 : base_cols
    data = Matrix{Any}(undef, n, n_cols)

    for (row, dr) in enumerate(degree_results)
        col = 1
        data[row, col] = dr.degree
        col += 1
        data[row, col] = dr.n_critical_points
        col += 1
        data[row, col] = fmt_sci(dr.l2_approx_error)
        col += 1
        if has_rel_l2
            rel = dr.relative_l2_error
            data[row, col] = isnan(rel) ? "N/A" : fmt_sci(rel)
            col += 1
        end
        data[row, col] = dr.best_objective !== nothing ? fmt_sci(dr.best_objective) : "N/A"
        col += 1
        data[row, col] = dr.status
        col += 1
        if show_timing_breakdown
            data[row, col] = fmt_time(dr.polynomial_construction_time)
            col += 1
            data[row, col] = fmt_time(dr.critical_point_solving_time)
            col += 1
        end
        data[row, col] = fmt_time(dr.total_computation_time)
    end

    base_header =
        has_rel_l2 ? ["Degree", "# CPs", "L2 err", "Rel L2", "Best f(x)", "Status"] :
        ["Degree", "# CPs", "L2 err", "Best f(x)", "Status"]
    base_align = has_rel_l2 ? [:r, :r, :r, :r, :r, :c] : [:r, :r, :r, :r, :c]
    status_col = has_rel_l2 ? 6 : 5
    header = if show_timing_breakdown
        vcat(base_header, ["Poly time", "HC Solve", "Total"])
    else
        vcat(base_header, ["Time"])
    end
    alignment = if show_timing_breakdown
        vcat(base_align, [:r, :r, :r])
    else
        vcat(base_align, [:r])
    end

    styled_table(
        io,
        data;
        header = header,
        alignment = alignment,
        highlighters = (
            Highlighter(
                (d, i, j) -> j == status_col && d[i, status_col] == "success",
                foreground = :green,
                bold = true,
            ),
            Highlighter(
                (d, i, j) -> j == status_col && d[i, status_col] != "success",
                foreground = :red,
                bold = true,
            ),
        ),
    )
end

function print_degree_analysis_table(
    degree_analyses::Vector{DegreeAnalysisResult};
    io::IO = stdout,
)
    isempty(degree_analyses) && return

    n = length(degree_analyses)
    data = Matrix{Any}(undef, n, 8)

    for (row, (deg, gv, ref)) in enumerate(degree_analyses)
        data[row, 1] = deg
        data[row, 2] = length(gv.norms)
        data[row, 3] = fmt_sci(gv.min_norm)
        data[row, 4] = fmt_sci(gv.median_norm)
        data[
            row,
            5,
        ] = "$(ref.n_converged)/$(ref.n_raw) ($(fmt_pct(ref.n_converged, ref.n_raw)))"
        data[row, 6] = fmt_sci(ref.best_raw_value)
        data[row, 7] = ref.n_converged > 0 ? fmt_sci(ref.best_refined_value) : "N/A"
        data[row, 8] = fmt_time(ref.total_time)
    end

    best_grad_row = argmin([gv.min_norm for (_, gv, _) in degree_analyses])
    best_ref_row = argmin([
        ref.n_converged > 0 ? ref.best_refined_value : Inf for
        (_, _, ref) in degree_analyses
    ])

    styled_table(
        io,
        data;
        header = [
            "Deg",
            "# CPs",
            "min ||∇f||",
            "med ||∇f||",
            "Converged",
            "Best raw",
            "Best ref",
            "Time",
        ],
        alignment = [:r, :r, :r, :r, :r, :r, :r, :r],
        highlighters = (
            Highlighter(
                (_, i, j) -> i == best_grad_row && j == 3,
                foreground = :green,
                bold = true,
            ),
            Highlighter(
                (_, i, j) -> i == best_ref_row && j == 7,
                foreground = :green,
                bold = true,
            ),
        ),
    )
end

function print_parameter_recovery_table(
    true_params::Vector{Float64},
    best_estimate::Vector{Float64};
    raw_estimate::Union{Vector{Float64},Nothing} = nothing,
    param_labels::Union{Vector{String},Nothing} = nothing,
    source_label::String = "refined",
    degree::Union{Int,Nothing} = nothing,
    io::IO = stdout,
)
    DIM = length(true_params)
    labels = param_labels !== nothing ? param_labels : ["p_$i" for i in 1:DIM]
    abs_errors = abs.(best_estimate .- true_params)
    rel_errors = abs_errors ./ abs.(true_params)

    has_raw = raw_estimate !== nothing
    n_cols = has_raw ? 6 : 5

    # Section header
    deg_str = degree !== nothing ? ", deg $degree" : ""
    print_section("Parameter Recovery (best across degrees$deg_str)"; io = io)

    # Data matrix: rows = params + summary
    data = Matrix{Any}(undef, DIM + 1, n_cols)
    for i in 1:DIM
        col = 1
        data[i, col] = labels[i]
        col += 1
        data[i, col] = @sprintf("%.4f", true_params[i])
        col += 1
        if has_raw
            data[i, col] = @sprintf("%.6f", raw_estimate[i])
            col += 1
        end
        data[i, col] = @sprintf("%.6f", best_estimate[i])
        col += 1
        data[i, col] = fmt_sci(abs_errors[i])
        col += 1
        data[i, col] = fmt_sci(rel_errors[i])
    end

    # Summary row (L2 norm of errors)
    ref_l2 = LinearAlgebra.norm(best_estimate .- true_params)
    sr = DIM + 1
    col = 1
    data[sr, col] = "||.||_2"
    col += 1
    data[sr, col] = ""
    col += 1
    if has_raw
        data[sr, col] = fmt_sci(LinearAlgebra.norm(raw_estimate .- true_params))
        col += 1
    end
    data[sr, col] = fmt_sci(ref_l2)
    col += 1
    data[sr, col] = ""
    data[sr, n_cols] = ""

    header = if has_raw
        ["param", "true", "raw est.", "$source_label est.", "abs error", "rel error"]
    else
        ["param", "true", "$source_label est.", "abs error", "rel error"]
    end

    abs_err_col = has_raw ? 5 : 4

    styled_table(
        io,
        data;
        header = header,
        alignment = fill(:r, n_cols),
        highlighters = (
            Highlighter((_, i, j) -> j == 1 && i <= DIM, bold = true, foreground = :cyan),
            Highlighter((_, i, j) -> i == DIM + 1, bold = true, foreground = :white),
            Highlighter(
                (d, i, j) ->
                    j == abs_err_col &&
                        i <= DIM &&
                        isa(d[i, j], String) &&
                        tryparse(Float64, d[i, j]) !== nothing &&
                        parse(Float64, d[i, j]) < 1e-6,
                bold = true,
                foreground = :green,
            ),
            Highlighter(
                (d, i, j) ->
                    j == abs_err_col &&
                        i <= DIM &&
                        isa(d[i, j], String) &&
                        tryparse(Float64, d[i, j]) !== nothing &&
                        1e-6 <= parse(Float64, d[i, j]) < 1e-3,
                foreground = :yellow,
            ),
            Highlighter(
                (d, i, j) ->
                    j == abs_err_col &&
                        i <= DIM &&
                        isa(d[i, j], String) &&
                        tryparse(Float64, d[i, j]) !== nothing &&
                        parse(Float64, d[i, j]) >= 1e-3,
                bold = true,
                foreground = :red,
            ),
        ),
        body_hlines = [DIM],
    )
end

function print_sparsification_metrics_table(
    comparisons::Vector{SparsifyCompare};
    io::IO = stdout,
)
    print_section("Sparsification Metrics (degree x threshold)"; io = io)

    n_comp = length(comparisons)
    metrics_data = Matrix{Any}(undef, n_comp, 9)

    for (row, sc) in enumerate(comparisons)
        metrics_data[row, 1] = sc.degree
        metrics_data[row, 2] = sc.threshold_label
        metrics_data[row, 3] = sc.full_n_coeffs
        metrics_data[row, 4] = sc.sparse_n_coeffs
        metrics_data[row, 5] = @sprintf("%.1f%%", sc.sparsity_pct)
        metrics_data[row, 6] = fmt_time(sc.full_solve_time)
        metrics_data[row, 7] = fmt_time(sc.sparse_solve_time)
        metrics_data[row, 8] = @sprintf("%.1f×", sc.speedup)
        metrics_data[row, 9] = fmt_sci(sc.threshold)
    end

    # Degree-boundary hlines
    n_thresh = _count_thresholds(comparisons)
    n_degrees = length(unique(sc.degree for sc in comparisons))
    hlines = n_thresh > 0 ? [i * n_thresh for i in 1:(n_degrees-1)] : Int[]

    styled_table(
        io,
        metrics_data;
        header = [
            "Deg",
            "Threshold",
            "Full #c",
            "Sparse #c",
            "Zeroed %",
            "Full time",
            "Sparse time",
            "Speedup",
            "Thresh",
        ],
        alignment = [:r, :l, :r, :r, :r, :r, :r, :r, :r],
        highlighters = (
            Highlighter(
                (data, i, j) -> j == 5 && occursin(r"^[5-9]\d|^100", string(data[i, 5])),
                foreground = :yellow,
                bold = true,
            ),
            Highlighter(
                (data, i, j) -> j == 8 && (m = match(r"^(\d+\.?\d*)", string(data[i, 8]));
                m !== nothing && parse(Float64, m[1]) >= 1.5),
                foreground = :green,
                bold = true,
            ),
        ),
        body_hlines = hlines,
    )
end

function print_sparsification_capture_table(
    comparisons::Vector{SparsifyCompare};
    io::IO = stdout,
)
    print_section("Capture Comparison: Full vs Sparsified (@ 5% tolerance)"; io = io)

    n_comp = length(comparisons)
    capture_data = Matrix{Any}(undef, n_comp, 7)

    for (row, sc) in enumerate(comparisons)
        delta = sc.sparse_capture_rate_5pct - sc.full_capture_rate_5pct
        capture_data[row, 1] = sc.degree
        capture_data[row, 2] = sc.threshold_label
        capture_data[row, 3] = sc.full_n_cps
        capture_data[row, 4] = sc.sparse_n_cps
        capture_data[row, 5] = @sprintf("%.1f%%", 100 * sc.full_capture_rate_5pct)
        capture_data[row, 6] = @sprintf("%.1f%%", 100 * sc.sparse_capture_rate_5pct)
        capture_data[row, 7] = @sprintf("%+.1f pp", 100 * delta)
    end

    n_thresh = _count_thresholds(comparisons)
    n_degrees = length(unique(sc.degree for sc in comparisons))
    hlines = n_thresh > 0 ? [i * n_thresh for i in 1:(n_degrees-1)] : Int[]

    styled_table(
        io,
        capture_data;
        header = [
            "Deg",
            "Threshold",
            "Full #CPs",
            "Sparse #CPs",
            "Full Cap@5%",
            "Sparse Cap@5%",
            "Δ Cap",
        ],
        alignment = [:r, :l, :r, :r, :r, :r, :r],
        highlighters = (
            Highlighter(
                (data, i, j) -> j == 7 && startswith(string(data[i, 7]), "+0.0"),
                foreground = :white,
            ),
            Highlighter(
                (data, i, j) -> j == 7 && startswith(string(data[i, 7]), "+"),
                foreground = :green,
                bold = true,
            ),
            Highlighter(
                (data, i, j) -> j == 7 && startswith(string(data[i, 7]), "-"),
                foreground = :red,
                bold = true,
            ),
        ),
        body_hlines = hlines,
    )
end
