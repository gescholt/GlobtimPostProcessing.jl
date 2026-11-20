# newton_census.jl — the index-preserving Newton census. It was validated against
# theory on a 4-D Lotka-Volterra parameter-estimation landscape whose critical set is
# known (20 of 20 predicted critical points: 6 minima, 10 index-1, 4 index-2).
#
# WHY A SEPARATE REFINER. The box-constrained minimiser (`Fminbox`) that the sweeps
# used as their census refiner MINIMISES: descent from a saddle lands on a minimum, so
# that pipeline cannot report a saddle even when the polynomial found one, parks on a
# box face whenever the descent direction points outward (a face stop that reports
# λ_min > 0 with ‖∇f‖ of order one passed as a "minimum" for 22 of 23 reported minima
# on a 3-D FitzHugh-Nagumo landscape), and runs one time-capped Optim call per
# candidate, sequentially — a 4 h time-out on a 4-D landscape was ~4,000 such calls. Each candidate here is polished by damped
# Newton on the SQUARE system ∇f(p) = 0, which converges to whatever critical point it
# is near, minimum, saddle or maximum alike, and every surviving point carries the
# evidence to judge it: ‖∇f‖ by exact AD, the full Hessian spectrum, the Morse index,
# and whether it is interior.
#
# The objective must be ForwardDiff-able (ODE misfits are, through the remake+solve
# closure; finite-difference noise sits far above the gradients that matter).

"""
    polish_to_critical(f, p0, lb, ub; iters = 80, gtol = 1e-12, ladder = 6,
                       backtracks = 24, stall_tol = 0.999) -> (p, ‖∇f(p)‖, iterations)

Damped Newton on `∇f(p) = 0`, clamped to the box `[lb, ub]`. Index-PRESERVING: the
step solves `(H + μI) d = -g` and is accepted only if it reduces `‖∇f‖`, so the
iteration is driven by the residual of the square system rather than by the objective
value, which is what lets it stop on a saddle instead of sliding off it. `μ` is raised
through `ladder` rungs (0, 1e-8, ×100, …) until a step decreases the residual; a
singular Hessian or a non-finite step falls back to the gradient direction. The loop
bails when the residual stops shrinking by more than `1 - stall_tol` per iteration (a
seed walking a flat valley is reported as a failure by the caller, never polished for
the full budget: on the 4-D validation landscape that tail was 17 % of the seeds and
most of the wall time).
"""
function polish_to_critical(
    f,
    p0::AbstractVector{<:Real},
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    iters::Int = 80,
    gtol::Real = 1e-12,
    ladder::Int = 6,
    backtracks::Int = 24,
    stall_tol::Real = 0.999,
)
    p = clamp.(Vector{Float64}(p0), lb, ub)
    g = ForwardDiff.gradient(f, p)
    all(isfinite, g) || return p, Inf, 0
    it = 0
    for k in 1:iters
        it = k
        ng = norm(g)
        ng < gtol && break
        H = ForwardDiff.hessian(f, p)
        all(isfinite, H) || break
        stepped = false
        mu = 0.0
        for _ in 1:ladder                  # Levenberg-style damping ladder
            d = try
                -(Symmetric(H) + mu * I) \ g
            catch
                -g
            end
            (all(isfinite, d) && norm(d) > 0) || (d = -g)
            alpha = 1.0
            for _ in 1:backtracks          # backtrack on the RESIDUAL, not on f
                pn = clamp.(p .+ alpha .* d, lb, ub)
                gn = ForwardDiff.gradient(f, pn)
                if all(isfinite, gn) && norm(gn) < ng
                    p, g, stepped = pn, gn, true
                    break
                end
                alpha /= 2
            end
            stepped && break
            mu = (mu == 0.0 ? 1e-8 : mu * 100)
        end
        stepped || break                   # no damping level made progress
        norm(g) > stall_tol * ng && break  # creeping: not in a basin of ∇f = 0
    end
    return p, norm(g), it
end

"""
    critical_point_kind(f, p; htol = 1e-8) -> (kind, index, eigenvalues)

Hessian spectrum of `f` at `p` by exact AD. `kind` is `:minimum`, `:maximum`,
`:saddle` or `:degenerate` (some |λ| ≤ `htol`); `:error` with index -1 when the
Hessian is not finite. `index` is the number of eigenvalues below `-htol`.
"""
function critical_point_kind(f, p::AbstractVector{<:Real}; htol::Real = 1e-8)
    ev = try
        eigvals(Symmetric(ForwardDiff.hessian(f, Vector{Float64}(p))))
    catch
        return :error, -1, Float64[]
    end
    all(isfinite, ev) || return :error, -1, ev
    nneg = count(<(-htol), ev)
    npos = count(>(htol), ev)
    nzero = length(ev) - nneg - npos
    kind = nzero > 0 ? :degenerate : nneg == 0 ? :minimum : npos == 0 ? :maximum : :saddle
    return kind, nneg, ev
end

"""
    dedup_by_radius(pts, rad) -> (kept, members)

Greedy dedup under an absolute per-axis radius vector `rad`. Returns the kept points
and, aligned with them, the indices into `pts` of the candidates each one absorbed
(itself first): the seed's provenance, so a polished point can be traced back to the
leaves whose polynomials proposed it.
"""
function dedup_by_radius(
    pts::AbstractVector{<:AbstractVector{<:Real}},
    rad::AbstractVector{<:Real},
)
    keep = Vector{Vector{Float64}}()
    members = Vector{Vector{Int}}()
    for (k, p) in enumerate(pts)
        j = findfirst(q -> all(i -> abs(p[i] - q[i]) < rad[i], eachindex(p)), keep)
        if j === nothing
            push!(keep, Vector{Float64}(p))
            push!(members, [k])
        else
            push!(members[j], k)
        end
    end
    return keep, members
end

"""
    read_raw_candidates(path, m) -> (points, leaf_ids)

Raw candidates from a CSV with columns `p1..pm` (any other columns are ignored) and,
when the file has a `leaf_id` column (cert_subdiv_sweep `raw_cps.csv`), the leaf each
came from (-1 otherwise).
"""
function read_raw_candidates(path::AbstractString, m::Int)
    pts = Vector{Vector{Float64}}()
    lids = Int[]
    open(path) do io
        header = split(strip(readline(io)), ",")
        idx = [findfirst(==("p$(i)"), header) for i in 1:m]
        any(isnothing, idx) && error("raw file $path lacks p1..p$m columns")
        lidx = findfirst(==("leaf_id"), header)
        for line in eachline(io)
            isempty(strip(line)) && continue
            fs = split(strip(line), ",")
            push!(pts, [parse(Float64, fs[i]) for i in idx])
            push!(lids, lidx === nothing ? -1 : parse(Int, fs[lidx]))
        end
    end
    return pts, lids
end

"""
    newton_census(f, raw, lb, ub; gtol = 1e-8, steptol = 0.0, seedrad = 1e-3,
                  dedup_tol = 1e-6, seedcap = 0, htol = 1e-8, threads = true,
                  progress = false) -> NamedTuple

The census proper: dedup the raw candidates into seeds (per-axis radius
`seedrad × box width`), polish each by [`polish_to_critical`](@ref), classify the ones
that converge by [`critical_point_kind`](@ref), and dedup the polished points (radius
`dedup_tol × width`) into the distinct critical points of `f`.

Acceptance: `‖∇f‖ < gtol` (absolute), or — when `steptol > 0` and the Hessian is
nondegenerate — a Newton step `H \\ ∇f` below `steptol` in box units, the scale-free
test for objectives whose loss scale puts `gtol` below double precision (rates ~30 on
the LV bricks: loss ~1e5, gradient terms ~1e8). A seed whose value is not finite never
converges (a failed ODE solve returns `Inf`, whose AD gradient is exactly 0).

`seedcap > 0` keeps only the `seedcap` lowest-`f` seeds: the result is then a LOWER
BOUND, and the caller must say so. With `threads = true` the seeds are polished under
`Threads.@threads` into preallocated slots, so the output is independent of the thread
count; the final dedup sorts on `(value, coordinates)`, a total order, so the distinct
set is deterministic too.

Returns `(; raw, seeds, members, srec, polished, distinct, nfail, wall, gtol, steptol,
dedup_tol, htol)`: `srec[k]` is the per-seed record `(p, gnorm, iters, ok)`, `polished`
the converged points as `(p, val, gnorm, kind, index, eigs, interior, iters, moved,
accepted_by, newton_step)`, `distinct` their dedup.
"""
function newton_census(
    f,
    raw::AbstractVector{<:AbstractVector{<:Real}},
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    gtol::Real = 1e-8,
    steptol::Real = 0.0,
    seedrad::Real = 1e-3,
    dedup_tol::Real = 1e-6,
    seedcap::Int = 0,
    htol::Real = 1e-8,
    threads::Bool = true,
    progress::Bool = false,
)
    length(lb) == length(ub) || throw(ArgumentError("lb and ub differ in length"))
    all(lb .< ub) || throw(ArgumentError("box must have lb < ub on every axis"))
    gtol > 0 || throw(ArgumentError("gtol must be > 0, got $gtol"))
    steptol >= 0 || throw(ArgumentError("steptol must be >= 0, got $steptol"))
    seedcap >= 0 || throw(ArgumentError("seedcap must be >= 0, got $seedcap"))
    lb = Vector{Float64}(lb)
    ub = Vector{Float64}(ub)
    width = ub .- lb
    raw_v = [Vector{Float64}(p) for p in raw]
    seeds, members = dedup_by_radius(raw_v, seedrad .* width)
    if seedcap > 0 && length(seeds) > seedcap
        ord = sortperm([(v = f(p); isfinite(v) ? v : Inf) for p in seeds])
        seeds = seeds[ord[1:seedcap]]
        members = members[ord[1:seedcap]]
    end

    t0 = time()
    slots = Vector{Union{Nothing,NamedTuple}}(nothing, length(seeds))
    srec = Vector{Union{Nothing,NamedTuple}}(nothing, length(seeds))
    done = Threads.Atomic{Int}(0)
    function polish_seed!(k)
        s = seeds[k]
        p, gn, it = polish_to_critical(f, s, lb, ub; gtol = gtol * 1e-2)
        fv = isfinite(gn) ? f(p) : Inf
        by = !isfinite(fv) ? :none : gn < gtol ? :gtol : :none
        nstep = NaN
        if by === :none && isfinite(fv) && isfinite(gn) && steptol > 0
            H = ForwardDiff.hessian(f, p)
            if all(isfinite, H)
                ev0 = eigvals(Symmetric(H))
                if minimum(abs, ev0) > htol
                    nstep =
                        maximum(abs.(Symmetric(H) \ ForwardDiff.gradient(f, p)) ./ width)
                    nstep < steptol && (by = :step)
                end
            end
        end
        ok = by !== :none
        srec[k] = (p = p, gnorm = gn, iters = it, ok = ok)
        if ok
            kind, idx, ev = critical_point_kind(f, p; htol = htol)
            eps_b = 1e-6 .* width
            is_int = all(i -> lb[i] + eps_b[i] < p[i] < ub[i] - eps_b[i], eachindex(p))
            slots[k] = (
                p = p,
                val = fv,
                gnorm = gn,
                kind = kind,
                index = idx,
                eigs = ev,
                interior = is_int,
                iters = it,
                moved = norm(p .- s),
                accepted_by = string(by),
                newton_step = nstep,
            )
        end
        if progress
            n = Threads.atomic_add!(done, 1) + 1
            n % 50 == 0 &&
                (println("  ... $n/$(length(seeds)) seeds polished"); flush(stdout))
        end
        return nothing
    end
    if threads
        Threads.@threads for k in eachindex(seeds)
            polish_seed!(k)
        end
    else
        for k in eachindex(seeds)
            polish_seed!(k)
        end
    end
    polished = NamedTuple[x for x in slots if x !== nothing]
    nfail = length(seeds) - length(polished)
    wall = time() - t0

    distinct = NamedTuple[]
    for x in sort(polished; by = q -> (q.val, q.p))
        any(
            q -> all(i -> abs(x.p[i] - q.p[i]) < dedup_tol * width[i], eachindex(x.p)),
            distinct,
        ) && continue
        push!(distinct, x)
    end
    return (;
        raw = raw_v,
        seeds,
        members,
        srec,
        polished,
        distinct,
        nfail,
        wall,
        gtol = Float64(gtol),
        steptol = Float64(steptol),
        dedup_tol = Float64(dedup_tol),
        htol = Float64(htol),
    )
end

"""
    poincare_hopf(f, distinct, lb, ub; nsamples = 2000, rng_seed = 1234)
        -> (; sum, n_interior, nrm, nout)

Poincaré–Hopf consistency: on a contractible box whose boundary the gradient crosses
OUTWARD everywhere, Σ (−1)^index over the interior critical points equals 1. A
different value is not automatically an error — the box is an arbitrary parameter
range, so the gradient may genuinely flow through faces — but it is exactly the
signature a MISSING critical point would also produce. `nout / nrm` is the sampled
fraction of boundary points (`nsamples`, uniform over the 2m faces) where the gradient
points outward, i.e. how far the theorem's hypothesis holds.
"""
function poincare_hopf(
    f,
    distinct::AbstractVector,
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    nsamples::Int = 2000,
    rng_seed::Int = 1234,
)
    m = length(lb)
    width = ub .- lb
    ints_ph = [q for q in distinct if q.interior]
    ph = sum(Int[(-1)^q.index for q in ints_ph]; init = 0)
    nrm = 0
    nout = 0
    rng = Random.MersenneTwister(rng_seed)
    for _ in 1:nsamples
        i = rand(rng, 1:m)
        hi = rand(rng, Bool)
        p = [lb[k] + rand(rng) * width[k] for k in 1:m]
        p[i] = hi ? ub[i] : lb[i]
        g = ForwardDiff.gradient(f, p)
        all(isfinite, g) || continue
        nrm += 1
        (hi ? g[i] : -g[i]) > 0 && (nout += 1)   # outward normal ±e_i
    end
    return (; sum = ph, n_interior = length(ints_ph), nrm, nout)
end

"""
    census_headline(c, ph) -> Dict{String,Any}

Interior counts by kind plus the Poincaré–Hopf result: the ledger headline of a census.
"""
function census_headline(c, ph)
    ints = [q for q in c.distinct if q.interior]
    return Dict{String,Any}(
        "n_raw" => length(c.raw),
        "n_seeds" => length(c.seeds),
        "n_converged" => length(c.polished),
        "n_failed" => c.nfail,
        "n_distinct" => length(c.distinct),
        "n_interior_minima" => count(q -> q.kind == :minimum, ints),
        "n_interior_saddles" => count(q -> q.kind == :saddle, ints),
        "n_interior_maxima" => count(q -> q.kind == :maximum, ints),
        "n_interior_degenerate" => count(q -> q.kind == :degenerate, ints),
        "poincare_hopf_sum" => ph.sum,
        "outward_frac" => ph.nrm == 0 ? nothing : ph.nout / ph.nrm,
    )
end

_jsonsafe(x::AbstractFloat) = isfinite(x) ? x : string(x)
_jsonsafe(x::AbstractDict) = Dict(k => _jsonsafe(v) for (k, v) in x)
_jsonsafe(x::AbstractVector) = [_jsonsafe(v) for v in x]
_jsonsafe(x) = x

"""
    write_census(out, seeds_csv, c, raw_lids, lb, ub; entry_name, raw, extra = Dict())

Write the census JSON to `out` and, unless `seeds_csv` is empty, the per-seed file:
one row per polished seed, converged or not, with the leaves that proposed it
(`raw_lids`, aligned with `c.raw`) and the distinct critical point it reached (1-based
index into `points`, 0 = failed). Without it "which leaf found which point" and the
polish cost per leaf cannot be recovered. `extra` keys describe the objective the
census was taken on (ODE tolerance, loss normalisation, …) and are merged into the
JSON verbatim. Non-finite floats are written as strings (JSON has no Inf/NaN).
"""
function write_census(
    out::AbstractString,
    seeds_csv::AbstractString,
    c,
    raw_lids::AbstractVector{<:Integer},
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    entry_name::AbstractString,
    raw::AbstractString,
    extra::AbstractDict = Dict{String,Any}(),
)
    m = length(lb)
    width = ub .- lb
    distinct = c.distinct
    mkpath(dirname(abspath(out)))
    function distinct_id(p)
        for (j, d) in enumerate(distinct)
            all(i -> abs(p[i] - d.p[i]) < c.dedup_tol * width[i], eachindex(p)) && return j
        end
        return 0
    end
    if !isempty(seeds_csv)
        open(seeds_csv, "w") do io
            println(
                io,
                "seed,",
                join(["s$(i)" for i in 1:m], ","),
                ",leaf_ids,n_raw,converged,iters,grad_norm,distinct_id,moved,",
                join(["p$(i)" for i in 1:m], ","),
            )
            for k in eachindex(c.seeds)
                r = c.srec[k]
                r === nothing && continue
                lids = unique(raw_lids[c.members[k]])
                did = r.ok ? distinct_id(r.p) : 0
                println(
                    io,
                    k,
                    ",",
                    join(c.seeds[k], ","),
                    ",",
                    join(lids, "|"),
                    ",",
                    length(c.members[k]),
                    ",",
                    Int(r.ok),
                    ",",
                    r.iters,
                    ",",
                    r.gnorm,
                    ",",
                    did,
                    ",",
                    norm(r.p .- c.seeds[k]),
                    ",",
                    join(r.p, ","),
                )
            end
        end
    end
    doc = Dict{String,Any}(
        "generated" => string(Dates.now()),
        "entry" => entry_name,
        "raw" => raw,
        "gtol" => c.gtol,
        "step_tol" => c.steptol,
        "hess_tol" => c.htol,
        "box" => [[lb[i], ub[i]] for i in 1:m],
        "n_raw" => length(c.raw),
        "n_seeds" => length(c.seeds),
        "n_converged" => length(c.polished),
        "n_failed" => c.nfail,
        "wall_s" => c.wall,
        "seeds_csv" => seeds_csv,
        "points" => [
            Dict(
                "p" => q.p,
                "val" => q.val,
                "grad_norm" => q.gnorm,
                "kind" => string(q.kind),
                "index" => q.index,
                "eigs" => q.eigs,
                "interior" => q.interior,
                "moved" => q.moved,
                "accepted_by" => q.accepted_by,
                "newton_step" => q.newton_step,
            ) for q in distinct
        ],
    )
    for (k, v) in extra
        doc[String(k)] = v
    end
    open(out, "w") do io
        JSON.print(io, _jsonsafe(doc), 2)
    end
    return out
end

# ── Refinement-method wrapper ───────────────────────────────────────────────────

"""
    NewtonIndexPreserving <: RefinementMethod

The census polish ([`polish_to_critical`](@ref)) as a refinement method: LM-damped
Newton on `∇f = 0` with exact AD, backtracking on the residual, so saddles are kept
as saddles. Converged when `‖∇f‖ < gtol`, or when `steptol > 0` and the Newton step in
box units is below it at a nondegenerate Hessian (needs `bounds`). Differs from
[`NewtonCP`](@ref) in having no trust region and no objective-increase safeguard (a
saddle above a plateau must be allowed to climb to), and in defaulting to ForwardDiff.
"""
Base.@kwdef struct NewtonIndexPreserving <: RefinementMethod
    gtol::Float64 = 1e-8
    steptol::Float64 = 0.0
    htol::Float64 = 1e-8
    iters::Int = 80
    ladder::Int = 6
    backtracks::Int = 24
    stall_tol::Float64 = 0.999
end
needs_gradient(::NewtonIndexPreserving) = true
needs_hessian(::NewtonIndexPreserving) = true
method_name(::NewtonIndexPreserving) = :newton_index_preserving

const _KIND_TO_CP_TYPE = Dict(
    :minimum => :min,
    :maximum => :max,
    :saddle => :saddle,
    :degenerate => :degenerate,
    :error => :unknown,
)

function refine_point(
    m::NewtonIndexPreserving,
    objective,
    point::Vector{Float64};
    bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing,
    trace::Bool = false,
)
    n = length(point)
    lb = bounds === nothing ? fill(-Inf, n) : [b[1] for b in bounds]
    ub = bounds === nothing ? fill(Inf, n) : [b[2] for b in bounds]
    g0 = ForwardDiff.gradient(objective, point)
    p, gn, it = polish_to_critical(
        objective,
        point,
        lb,
        ub;
        iters = m.iters,
        gtol = m.gtol * 1e-2,
        ladder = m.ladder,
        backtracks = m.backtracks,
        stall_tol = m.stall_tol,
    )
    fv = isfinite(gn) ? objective(p) : Inf
    converged = isfinite(fv) && gn < m.gtol
    if !converged && isfinite(fv) && isfinite(gn) && m.steptol > 0 && bounds !== nothing
        H = ForwardDiff.hessian(objective, p)
        if all(isfinite, H) && minimum(abs, eigvals(Symmetric(H))) > m.htol
            nstep = maximum(
                abs.(Symmetric(H) \ ForwardDiff.gradient(objective, p)) ./ (ub .- lb),
            )
            converged = nstep < m.steptol
        end
    end
    kind, _, ev =
        isfinite(fv) ? critical_point_kind(objective, p; htol = m.htol) :
        (:error, -1, Float64[])
    return CriticalPointRefinementResult(
        point = p,
        objective_value = fv,
        cp_type = _KIND_TO_CP_TYPE[kind],
        eigenvalues = ev,
        min_eigenvalue = isempty(ev) ? NaN : minimum(ev),
        condition_number = (isempty(ev) || minimum(abs, ev) == 0) ? NaN :
                           maximum(abs, ev) / minimum(abs, ev),
        gradient_norm = gn,
        converged = converged,
        iterations = it,
        initial_gradient_norm = all(isfinite, g0) ? norm(g0) : Inf,
        trace = trace ? [copy(point), copy(p)] : Vector{Float64}[],
        gradient_method = :forwarddiff,
    )
end
