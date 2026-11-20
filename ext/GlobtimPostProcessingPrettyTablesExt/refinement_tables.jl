# Table printers moved from src/refinement/api.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function print_comparison_table(
    result::RefinedExperimentResult,
    gradient_validation::Union{GradientValidationResult,Nothing} = nothing;
    raw_gradient_norms::Union{Vector{Float64},Nothing} = nothing,
    n_show::Union{Int,Float64} = 10,
    sort_by::Symbol = :refined_value,
)
    cr_bold = Crayon(bold = true)
    cr_cyan = Crayon(foreground = :cyan)

    n_display = isinf(n_show) ? result.n_raw : min(Int(n_show), result.n_raw)

    # Build indices based on sort order
    indices = collect(1:result.n_raw)
    if sort_by == :raw_value
        sort!(indices, by = i -> result.raw_values[i])
    elseif sort_by == :refined_value
        # Sort converged first by refined value, then non-converged
        sort!(
            indices,
            by = i -> begin
                if result.convergence_status[i]
                    converged_idx = sum(result.convergence_status[1:i])
                    return result.refined_values[converged_idx]
                else
                    return Inf
                end
            end,
        )
    elseif sort_by == :improvement
        sort!(
            indices,
            by = i -> begin
                if result.convergence_status[i]
                    converged_idx = sum(result.convergence_status[1:i])
                    return -result.improvements[converged_idx]  # Negative for descending
                else
                    return 0.0
                end
            end,
            rev = true,
        )
    end
    # :index keeps original order

    indices = indices[1:n_display]

    function format_sci_short(x::Float64)
        if isnan(x) || isinf(x)
            return "N/A"
        end
        @sprintf("%.3e", x)
    end

    # Determine columns: Idx, Raw Val, ||∇f|| raw, Ref Val, Improv, [||∇f|| ref]
    has_raw_grads = raw_gradient_norms !== nothing
    has_ref_grads = gradient_validation !== nothing

    n_cols = 4  # Idx, Raw Val, Ref Val, Improv
    if has_raw_grads
        n_cols += 1
    end
    if has_ref_grads
        n_cols += 1
    end
    data = Matrix{Any}(undef, n_display, n_cols)

    for (row, i) in enumerate(indices)
        col = 1
        data[row, col] = i  # Index
        col += 1
        data[row, col] = format_sci_short(result.raw_values[i])
        col += 1

        # Raw gradient norm (one per raw point, always available if provided)
        if has_raw_grads
            data[row, col] = format_sci_short(raw_gradient_norms[i])
            col += 1
        end

        if result.convergence_status[i]
            converged_idx = sum(result.convergence_status[1:i])
            data[row, col] = format_sci_short(result.refined_values[converged_idx])
            col += 1
            data[row, col] = format_sci_short(result.improvements[converged_idx])
            col += 1
            if has_ref_grads
                data[row, col] = format_sci_short(gradient_validation.norms[converged_idx])
            end
        else
            data[row, col] = "N/A"  # Ref Val
            col += 1
            data[row, col] = "N/A"  # Improv
            col += 1
            if has_ref_grads
                data[row, col] = "N/A"
            end
        end
    end

    # Build header and alignment dynamically
    header_vec = ["Idx", "Raw Val"]
    if has_raw_grads
        push!(header_vec, "||∇f|| raw")
    end
    push!(header_vec, "Ref Val", "Improv")
    if has_ref_grads
        push!(header_vec, "||∇f|| ref")
    end

    alignment = fill(:r, n_cols)

    println()
    println(cr_bold(cr_cyan("Raw vs Refined (showing $n_display of $(result.n_raw))")))
    println()

    styled_table(data; header = header_vec, alignment = alignment)
    println()
end
