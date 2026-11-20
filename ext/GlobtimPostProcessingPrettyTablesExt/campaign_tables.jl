# Table printers moved from src/CampaignAnalysis.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function format_metrics_pretty_table(metrics::Dict, _metric_labels::Dict)
    # Metric name mappings for better readability
    metric_labels = Dict(
        "approximation_quality" => "Approximation Quality (L2 Error)",
        "parameter_recovery" => "Parameter Recovery Error",
        "numerical_stability" => "Numerical Stability (Cond. #)",
        "critical_points" => "Critical Points (Total)",
        "critical_point_count" => "Critical Points (Raw)",
        "refined_critical_points" => "Critical Points (Refined)",
        "refinement_quality" => "Refinement Success Rate",
        "optimization_quality" => "Optimization Quality",
        "polynomial_timing" => "Polynomial Construction Time (s)",
        "solving_timing" => "Solving Time (s)",
        "refinement_timing" => "Refinement Time (s)",
        "total_timing" => "Total Computation Time (s)",
        "timing" => "Total Time (s)",
    )

    # Create a table with all metrics
    metric_names = String[]
    means = String[]
    mins = String[]
    maxs = String[]
    stds = String[]
    num_exps = Int[]
    best_exps = String[]

    for (metric_name, metric_data) in sort(collect(metrics), by = x -> x[1])
        # Use human-readable label
        readable_name = get(metric_labels, metric_name, metric_name)
        push!(metric_names, readable_name)

        # Format values
        push!(means, format_scientific(get(metric_data, "mean", NaN)))
        push!(mins, format_scientific(get(metric_data, "min", NaN)))
        push!(maxs, format_scientific(get(metric_data, "max", NaN)))
        push!(stds, format_scientific(get(metric_data, "std", NaN)))
        push!(num_exps, get(metric_data, "num_experiments", 0))

        # Extract short readable label from best experiment
        best_exp = get(metric_data, "best_experiment", "N/A")
        best_exp_short =
            best_exp == "N/A" ? "N/A" : extract_experiment_label(best_exp, max_length = 30)
        push!(best_exps, best_exp_short)
    end

    # Display table
    header = ["Metric", "Mean", "Min", "Max", "Std Dev", "N", "Best Experiment"]
    data = hcat(metric_names, means, mins, maxs, stds, num_exps, best_exps)

    styled_table(
        data;
        header = header,
        alignment = [:l, :r, :r, :r, :r, :c, :l],
        header_crayon = crayon"bold cyan",
        border_crayon = crayon"cyan",
    )
end
