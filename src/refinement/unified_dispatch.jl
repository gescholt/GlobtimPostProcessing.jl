"""
    Unified Refinement Dispatch

Single entry point `refine_point` dispatches to the appropriate refinement backend
(Newton trust-region or Optim.jl) based on the `RefinementMethod` type. All methods
return `CriticalPointRefinementResult`.

`refine_points` provides batch refinement with progress reporting.

Created: 2026-03-18
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Single-point dispatch
# ═══════════════════════════════════════════════════════════════════════════════

"""
    refine_point(method, objective, point; bounds=nothing, trace=false)
        -> CriticalPointRefinementResult

Unified entry point — dispatch refinement based on `method` type. All methods
return `CriticalPointRefinementResult` with convergence info, CP classification,
and eigenvalue diagnostics.

# Arguments
- `method::RefinementMethod`: Which refinement method to use (e.g. `NewtonCP()`, `OptimLBFGS()`)
- `objective`: Callable `f(x::Vector{Float64}) -> Float64`
- `point::Vector{Float64}`: Starting point

# Keyword Arguments
- `bounds`: Box constraints as `Vector{Tuple{Float64,Float64}}` or `nothing`
- `trace::Bool`: Record iteration trajectory (default: `false`)
"""
function refine_point end

# ── Newton CP ────────────────────────────────────────────────────────────────

function refine_point(
    m::NewtonCP,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    refine_to_critical_point(
        objective,
        point;
        gradient_method = m.gradient_method,
        tol = m.tol,
        accept_tol = m.accept_tol,
        f_accept_tol = m.f_accept_tol,
        max_iterations = m.max_iterations,
        bounds = bounds,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        trust_radius_fraction = m.trust_radius_fraction,
        trust_expand = m.trust_expand,
        trust_shrink = m.trust_shrink,
        max_f_increase_factor = m.max_f_increase_factor,
        patience = m.patience,
        min_improvement_ratio = m.min_improvement_ratio,
        trace = trace,
        mode = :critical_point,
    )
end

# ── Newton Minimize ──────────────────────────────────────────────────────────

function refine_point(
    m::NewtonMinimize,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    refine_to_critical_point(
        objective,
        point;
        gradient_method = m.gradient_method,
        tol = m.tol,
        accept_tol = m.accept_tol,
        f_accept_tol = m.f_accept_tol,
        max_iterations = m.max_iterations,
        bounds = bounds,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        trust_radius_fraction = m.trust_radius_fraction,
        trust_expand = m.trust_expand,
        trust_shrink = m.trust_shrink,
        max_f_increase_factor = m.max_f_increase_factor,
        patience = m.patience,
        min_improvement_ratio = m.min_improvement_ratio,
        trace = trace,
        mode = :minimize,
    )
end

# ── Optim: NelderMead ────────────────────────────────────────────────────────

function refine_point(
    m::OptimNelderMead,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    r = refine_critical_point(
        objective,
        point;
        method = Optim.NelderMead(),
        bounds = bounds,
        f_abstol = m.f_abstol,
        x_abstol = m.x_abstol,
        max_time = m.max_time,
        max_iterations = m.max_iterations,
        store_trace = trace,
    )
    _wrap_optim_as_cpresult(
        r,
        objective;
        gradient_method = m.gradient_method,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        bounds = bounds,
        initial_point = point,
    )
end

# ── Optim: LBFGS ─────────────────────────────────────────────────────────────

function refine_point(
    m::OptimLBFGS,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    r = refine_critical_point(
        objective,
        point;
        method = :lbfgs,
        bounds = bounds,
        f_abstol = m.f_abstol,
        x_abstol = m.x_abstol,
        max_time = m.max_time,
        max_iterations = m.max_iterations,
        step_size = m.step_size,
        store_trace = trace,
    )
    _wrap_optim_as_cpresult(
        r,
        objective;
        gradient_method = m.gradient_method,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        bounds = bounds,
        initial_point = point,
    )
end

# ── Optim: BFGS ──────────────────────────────────────────────────────────────

function refine_point(
    m::OptimBFGS,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    r = refine_critical_point(
        objective,
        point;
        method = :bfgs,
        bounds = bounds,
        f_abstol = m.f_abstol,
        x_abstol = m.x_abstol,
        max_time = m.max_time,
        max_iterations = m.max_iterations,
        store_trace = trace,
    )
    _wrap_optim_as_cpresult(
        r,
        objective;
        gradient_method = m.gradient_method,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        bounds = bounds,
        initial_point = point,
    )
end

# ── Optim: ConjugateGradient ─────────────────────────────────────────────────

function refine_point(
    m::OptimConjugateGradient,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    r = refine_critical_point(
        objective,
        point;
        method = :conjugategradient,
        bounds = bounds,
        f_abstol = m.f_abstol,
        x_abstol = m.x_abstol,
        max_time = m.max_time,
        max_iterations = m.max_iterations,
        store_trace = trace,
    )
    _wrap_optim_as_cpresult(
        r,
        objective;
        gradient_method = m.gradient_method,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        bounds = bounds,
        initial_point = point,
    )
end

# ── Optim: GradientDescent ───────────────────────────────────────────────────

function refine_point(
    m::OptimGradientDescent,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    r = refine_critical_point(
        objective,
        point;
        method = :gradientdescent,
        bounds = bounds,
        f_abstol = m.f_abstol,
        x_abstol = m.x_abstol,
        max_time = m.max_time,
        max_iterations = m.max_iterations,
        store_trace = trace,
    )
    _wrap_optim_as_cpresult(
        r,
        objective;
        gradient_method = m.gradient_method,
        hessian_tol = m.hessian_tol,
        hessian_relative_tol = m.hessian_relative_tol,
        bounds = bounds,
        initial_point = point,
    )
end

# ── Levenberg–Marquardt on a residual vector (bead rai5.9) ──────────────────

function refine_point(
    m::LevenbergMarquardt,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    project(x) = bounds === nothing ? x : [clamp(x[i], bounds[i][1], bounds[i][2]) for i in eachindex(x)]
    x = project(copy(point))
    t0 = time()
    sumsq(r) = sum(abs2, r)
    r = Float64.(m.residual(x))
    trace_pts = Vector{Float64}[]
    trace_g = Float64[]
    finite_start = all(isfinite, r)
    J = finite_start ? ForwardDiff.jacobian(m.residual, x) : zeros(length(r), length(x))
    f = finite_start ? sumsq(r) : Inf
    g = 2.0 .* (J' * r)                         # ∇‖r‖²
    gnorm(v) = all(isfinite, v) ? maximum(abs, v) : Inf
    initial_gnorm = gnorm(g)
    trace && (push!(trace_pts, copy(x)); push!(trace_g, initial_gnorm))
    lambda = m.lambda0
    iterations = 0
    rejected = 0
    converged = finite_start && initial_gnorm <= m.g_abstol
    stalled = !finite_start
    while !converged && !stalled && iterations < m.max_iterations
        if m.max_time !== nothing && time() - t0 > m.max_time
            break
        end
        A = J' * J
        D = LinearAlgebra.Diagonal(max.(LinearAlgebra.diag(A), eps(Float64)))
        delta = try
            -((A + lambda * D) \ (J' * r))
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            fill(NaN, length(x))
        end
        x_new = all(isfinite, delta) ? project(x .+ delta) : x
        r_new = all(isfinite, delta) ? Float64.(m.residual(x_new)) : r
        f_new = all(isfinite, r_new) ? sumsq(r_new) : Inf
        if all(isfinite, delta) && f_new < f
            step = LinearAlgebra.norm(x_new .- x)
            decrease = (f - f_new) / max(f, floatmin(Float64))
            x = x_new; r = r_new; f = f_new
            J = ForwardDiff.jacobian(m.residual, x)
            g = 2.0 .* (J' * r)
            lambda = max(lambda * m.lambda_down, 1e-15)
            iterations += 1
            trace && (push!(trace_pts, copy(x)); push!(trace_g, gnorm(g)))
            if gnorm(g) <= m.g_abstol
                converged = true
            elseif step <= m.x_abstol || decrease <= m.f_reltol
                stalled = true                  # accepted step too small to matter
            end
        else
            rejected += 1
            lambda *= m.lambda_up
            iterations += 1                     # a rejected step still costs a solve
            lambda > m.lambda_max && (stalled = true)
        end
    end
    # A projected point sits on the box: a zero gradient there is not
    # first-order optimality of the box problem, but the convention shared with
    # the other methods is gradient-based, so `converged` means ‖∇f‖∞ ≤ g_abstol.
    obj_value = try
        Float64(objective(x))
    catch
        NaN
    end
    cp_type = :unknown
    eigenvalues = Float64[]
    eigvecs = Matrix{Float64}(undef, 0, 0)
    cond_num = NaN
    min_eig = NaN
    if finite_start && all(isfinite, r)
        H = ForwardDiff.hessian(y -> sumsq(m.residual(y)), x)
        if all(isfinite, H)
            eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(H))
            eigenvalues = collect(eig.values)
            eigvecs = eig.vectors
            cp_type = _classify_eigenvalues(eigenvalues, m.hessian_tol; relative_tol = m.hessian_relative_tol)
            cond_num, min_eig = _eigenvalue_diagnostics(eigenvalues)
        end
    end
    return CriticalPointRefinementResult(
        point = x,
        objective_value = obj_value,
        cp_type = cp_type,
        eigenvalues = eigenvalues,
        eigenvectors = eigvecs,
        condition_number = cond_num,
        min_eigenvalue = min_eig,
        gradient_norm = gnorm(g),
        converged = converged,
        iterations = iterations,
        initial_gradient_norm = initial_gnorm,
        rejected_steps = rejected,
        trace = trace_pts,
        trace_gradient_norms = trace_g,
        gradient_method = :forwarddiff,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Batch refinement
# ═══════════════════════════════════════════════════════════════════════════════

"""
    refine_points(method, objective, points; bounds, trace, io)
        -> Vector{CriticalPointRefinementResult}

Batch version of `refine_point` with progress reporting. Refines each point
independently and prints progress to `io`.

# Arguments
- `method::RefinementMethod`: Which refinement method to use
- `objective`: Callable `f(x::Vector{Float64}) -> Float64`
- `points::Vector{Vector{Float64}}`: Starting points

# Keyword Arguments
- `bounds`: Box constraints or `nothing`
- `trace::Bool`: Record iteration trajectories (default: `false`)
- `io::IO`: Output stream for progress (default: `stdout`)
"""
function refine_points(
    method::RefinementMethod,
    objective,
    points::Vector{Vector{Float64}};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
    io::IO = stdout,
)
    n_pts = length(points)
    results = Vector{CriticalPointRefinementResult}(undef, n_pts)
    t_total = time()
    cumulative_time = 0.0

    n_conv = 0
    n_accepted = 0
    n_rejected = 0

    mname = method_name(method)

    for (i, pt) in enumerate(points)
        t_start = time()
        results[i] = refine_point(method, objective, pt; bounds = bounds, trace = trace)
        r = results[i]
        elapsed = time() - t_start
        cumulative_time += elapsed

        if r.converged
            n_conv += 1
        elseif r.cp_type != :unknown
            n_accepted += 1
        else
            n_rejected += 1
        end

        # Print accepted/converged CPs immediately
        is_useful = r.cp_type != :unknown || r.converged
        if is_useful
            status = r.converged ? "converged" : "accepted "
            @printf(
                io,
                "    ✓ CP %d/%d: %s %3d iters (%4.1fs)  |∇f| %.2e  f=%.2e  (%s)\n",
                i,
                n_pts,
                status,
                r.iterations,
                elapsed,
                r.gradient_norm,
                r.objective_value,
                r.cp_type
            )
        end

        # Progress line every 20 CPs
        if !is_useful && (i % 20 == 0 || i == n_pts)
            avg_time = cumulative_time / i
            remaining = avg_time * (n_pts - i)
            eta_str =
                remaining >= 60 ? @sprintf("ETA ~%.0fm", remaining / 60) :
                remaining >= 5 ? @sprintf("ETA ~%.0fs", remaining) : ""
            pct = round(Int, 100 * i / n_pts)
            @printf(
                io,
                "    [%s] %d/%d (%d%%) | %d conv, %d acc, %d rej%s\r",
                mname,
                i,
                n_pts,
                pct,
                n_conv,
                n_accepted,
                n_rejected,
                isempty(eta_str) ? "" : " | $eta_str"
            )
            flush(io)
        end
    end

    # Clear progress line and print summary
    print(io, "    " * " "^80 * "\r")
    total_time = time() - t_total
    @printf(
        io,
        "    [%s] Summary: %d converged, %d accepted, %d rejected  (%.1fs)\n",
        mname,
        n_conv,
        n_accepted,
        n_rejected,
        total_time
    )

    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
# RefinementConfig → RefinementMethod conversion
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RefinementMethod(config::RefinementConfig) -> RefinementMethod

Convert a legacy `RefinementConfig` to the modular `RefinementMethod` system.
Inspects `config.method` to determine the appropriate concrete type.
"""
function RefinementMethod(config::RefinementConfig)
    m = config.method
    max_time = config.max_time_per_point
    gm = config.gradient_method
    ht = config.gradient_tolerance

    if m isa Optim.NelderMead
        OptimNelderMead(;
            f_abstol = config.f_abstol,
            x_abstol = config.x_abstol,
            max_time,
            max_iterations = config.max_iterations,
            gradient_method = gm,
            hessian_tol = ht,
        )
    elseif m isa Optim.LBFGS
        OptimLBFGS(;
            f_abstol = config.f_abstol,
            x_abstol = config.x_abstol,
            max_time,
            max_iterations = config.max_iterations,
            gradient_method = gm,
            hessian_tol = ht,
        )
    elseif m isa Optim.BFGS
        OptimBFGS(;
            f_abstol = config.f_abstol,
            x_abstol = config.x_abstol,
            max_time,
            max_iterations = config.max_iterations,
            gradient_method = gm,
            hessian_tol = ht,
        )
    elseif m isa Optim.ConjugateGradient
        OptimConjugateGradient(;
            f_abstol = config.f_abstol,
            x_abstol = config.x_abstol,
            max_time,
            max_iterations = config.max_iterations,
            gradient_method = gm,
            hessian_tol = ht,
        )
    elseif m isa Optim.GradientDescent
        OptimGradientDescent(;
            f_abstol = config.f_abstol,
            x_abstol = config.x_abstol,
            max_time,
            max_iterations = config.max_iterations,
            gradient_method = gm,
            hessian_tol = ht,
        )
    else
        error(
            "Cannot convert RefinementConfig with method type $(typeof(m)) to RefinementMethod. " *
            "Supported: NelderMead, LBFGS, BFGS, ConjugateGradient, GradientDescent",
        )
    end
end
