"""
    BenefitMatrix.jl

Assemble the **subdivision × sparsification → parameter-recovery** benefit matrix
across the ODE parameter-estimation (PE) test systems, from EXISTING result data.

This is the *frozen baseline* (Part A of the stabilize→batch effort): it quantifies
how much subdivision and sparsification have actually helped PE *given the data we
already have*, and — just as importantly — enumerates exactly where that data is
missing. Parts B+C (the stabilization gate and the warm-started batch solver) will
later re-run the missing cells cheaply and re-measure against this baseline.

# Honest reporting (no imputation)

Cells the tracked data cannot fill are `missing`. The headline finding the matrix is
designed to surface: the tracked subdivision runs (`per_axis_audit/`) never solved for
critical points, so **`recovery_error` is `missing` for every subdivision-on cell** —
the existing data literally cannot say how much subdivision improved PE recovery. The
one end-to-end subdivision→recovery datapoint (Phase G: fhn3d 2.06e-7, lv4d 4.81e-8)
is an *untracked sandbox smoke* (`experiments/sandbox/smoke_step15_wide_r.jl`,
writeups in gitignored `_generated/phase_G_*.md`) at a different configuration.

# Data sources

  * **subdivision=off** — `<cluster_root>/<config>/results_summary.json`: per-degree
    `recovery_error` / `best_objective` / `relative_l2_error` / `critical_point_solving_time`,
    serialized by the experiment runner. We take the best (min `recovery_error`) across
    successful degrees.
  * **subdivision=on** — `<audit_root>/<system>/<run>/{predcall.jsonl, audit_summary.json}`:
    per-leaf `rel_l2` + `n_leaves` + `wall_s` ONLY. These audits replay split predicates;
    they never solve the gradient system, so `recovery_error`/`best_objective` are `missing`.
  * **sparsify=on** — `<sparsification_root>/<problem>_sparsification/sparsification_results.json`:
    exists only for 2D analytical benchmarks (ackley/levy/deuflhard), never for the ODE
    PE systems → every ODE sparsify cell is `missing`.
"""

const ODE_PE_SYSTEMS = ["lv4d", "fhn3d", "daisy4d", "goodwin4d"]

"""
    BenefitCell

One cell of the benefit matrix: a `(system, subdivision, sparsify)` configuration and the
PE-quality metrics recoverable from existing data. Numeric fields are `missing` when the
tracked data does not carry them. `source` is the file/dir the values came from (or `"—"`),
and `note` records why a cell is `missing` or any comparability caveat.
"""
struct BenefitCell
    system::String
    subdivision::Bool
    sparsify::Bool
    recovery_error::Union{Float64,Missing}
    best_objective::Union{Float64,Missing}
    rel_l2::Union{Float64,Missing}
    n_leaves::Union{Int,Missing}
    solve_time::Union{Float64,Missing}
    source::String
    note::String
end

# ── Single-domain (subdivision=off) ─────────────────────────────────────────────

# Best (min recovery_error) across successful degrees of one results_summary.json.
# Returns (recovery_error, best_objective, rel_l2, solve_time, degree) or `nothing`
# when no degree carries a numeric recovery_error.
function _best_single_domain(summary_path::AbstractString)
    isfile(summary_path) || return nothing
    data = JSON.parsefile(summary_path)
    haskey(data, "results_summary") || return nothing
    rs = data["results_summary"]
    rs isa AbstractDict || return nothing

    best = nothing
    for (deg_key, deg) in rs
        deg isa AbstractDict || continue
        get(deg, "status", "") == "success" || continue
        rec = get(deg, "recovery_error", nothing)
        (rec isa Real) || continue          # JSON null / absent → skip
        if best === nothing || rec < best.recovery_error
            degnum = tryparse(Int, replace(string(deg_key), "degree_" => ""))
            best = (
                recovery_error = Float64(rec),
                best_objective = _num_or_missing(get(deg, "best_objective", nothing)),
                rel_l2 = _num_or_missing(get(deg, "relative_l2_error", nothing)),
                solve_time = _num_or_missing(
                    get(deg, "critical_point_solving_time", nothing),
                ),
                degree = degnum === nothing ? -1 : degnum,
            )
        end
    end
    return best
end

# All <cluster_root>/<config> dirs whose name starts with the system token followed by
# `_` (so "lv4d" matches "lv4d_tight" but not "lv2d"/"lv3d").
function _single_domain_dirs(cluster_root::AbstractString, system::AbstractString)
    isdir(cluster_root) || return String[]
    dirs = String[]
    for entry in readdir(cluster_root)
        full = joinpath(cluster_root, entry)
        isdir(full) || continue
        startswith(entry, system * "_") || continue
        isfile(joinpath(full, "results_summary.json")) || continue
        push!(dirs, full)
    end
    return sort(dirs)
end

# ── Subdivision audits (subdivision=on) ─────────────────────────────────────────

# Most-refined audit run for a system: max n_leaves. Returns
# (rel_l2_min, n_leaves, wall_s, run_dir) or `nothing`.
function _best_audit_run(audit_root::AbstractString, system::AbstractString)
    sysdir = joinpath(audit_root, system)
    isdir(sysdir) || return nothing
    best = nothing
    for run in readdir(sysdir)
        rundir = joinpath(sysdir, run)
        isdir(rundir) || continue
        summary_path = joinpath(rundir, "audit_summary.json")
        isfile(summary_path) || continue
        s = JSON.parsefile(summary_path)
        n_leaves = _int_or_missing(get(s, "n_leaves", nothing))
        n_leaves === missing && continue
        rel_l2 = _audit_min_rel_l2(joinpath(rundir, "predcall.jsonl"))
        wall_s = _num_or_missing(get(s, "wall_s", nothing))
        if best === nothing || n_leaves > best.n_leaves
            best = (rel_l2 = rel_l2, n_leaves = n_leaves, wall_s = wall_s, run_dir = rundir)
        end
    end
    return best
end

# Min per-leaf rel_l2 over a predcall.jsonl (the best approximation quality any leaf
# reached). `missing` if the file is absent or carries no numeric rel_l2.
function _audit_min_rel_l2(predcall_path::AbstractString)
    isfile(predcall_path) || return missing
    best = missing
    for line in eachline(predcall_path)
        isempty(strip(line)) && continue
        row = JSON.parse(line)
        row isa AbstractDict || continue
        v = get(row, "rel_l2", nothing)
        v isa Real || continue
        best = (best === missing) ? Float64(v) : min(best, Float64(v))
    end
    return best
end

# ── Sparsification ──────────────────────────────────────────────────────────────

# Sparsification results exist only for 2D analytical benchmarks, never for ODE PE
# systems. Returns `nothing` whenever the per-system file is absent (the ODE case).
function _sparsification_for(sparsification_root::AbstractString, system::AbstractString)
    path = joinpath(
        sparsification_root,
        "$(system)_sparsification",
        "sparsification_results.json",
    )
    isfile(path) || return nothing
    return path  # present only if someone runs ODE sparsification later
end

# ── Coercion helpers ─────────────────────────────────────────────────────────────

_num_or_missing(x::Real) = Float64(x)
_num_or_missing(::Any) = missing
_int_or_missing(x::Integer) = Int(x)
_int_or_missing(x::Real) = Int(round(x))
_int_or_missing(::Any) = missing

# ── Public API ───────────────────────────────────────────────────────────────────

"""
    assemble_benefit_matrix(; cluster_root, audit_root, sparsification_root,
                              systems=ODE_PE_SYSTEMS)
        -> (cells::Vector{BenefitCell}, gaps::Vector{String})

Scan existing result data and assemble the `{system × subdivision∈{off,on} ×
sparsify∈{off,on}}` benefit matrix. Returns the cells (4 per system) and a list of
explicit data-gap statements. No values are imputed: a metric the tracked data does
not carry is `missing`, and the reason is captured in the cell `note` and/or `gaps`.

# Keyword arguments
- `cluster_root` : dir of single-domain experiment outputs (`<config>/results_summary.json`).
- `audit_root` : dir of per-axis subdivision audits (`<system>/<run>/...`).
- `sparsification_root` : dir of sparsification baseline outputs.
- `systems` : ODE PE systems to scan (default `ODE_PE_SYSTEMS`).
"""
function assemble_benefit_matrix(;
    cluster_root::AbstractString = joinpath("globtim_results", "cluster"),
    audit_root::AbstractString = joinpath("globtim_results", "cluster", "per_axis_audit"),
    sparsification_root::AbstractString = joinpath("globtim_results", "subdivision"),
    systems::AbstractVector{<:AbstractString} = ODE_PE_SYSTEMS,
)
    cells = BenefitCell[]
    gaps = String[]

    for system in systems
        # ---- subdivision=off, sparsify=off : single-domain recovery ----
        sd_dirs = _single_domain_dirs(cluster_root, system)
        if isempty(sd_dirs)
            push!(
                cells,
                BenefitCell(
                    system,
                    false,
                    false,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    "—",
                    "no single-domain results_summary.json",
                ),
            )
            push!(
                gaps,
                "[$system] no single-domain run — subdivision=off recovery is missing",
            )
        else
            # pick the config with the best (min) recovery_error
            best = nothing
            best_dir = ""
            best_deg = -1
            for d in sd_dirs
                r = _best_single_domain(joinpath(d, "results_summary.json"))
                r === nothing && continue
                if best === nothing || r.recovery_error < best.recovery_error
                    best = r
                    best_dir = d
                    best_deg = r.degree
                end
            end
            if best === nothing
                push!(
                    cells,
                    BenefitCell(
                        system,
                        false,
                        false,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        join(basename.(sd_dirs), ","),
                        "single-domain runs present but none recorded a numeric recovery_error",
                    ),
                )
                push!(gaps, "[$system] single-domain runs carry no recovery_error")
            else
                push!(
                    cells,
                    BenefitCell(
                        system,
                        false,
                        false,
                        best.recovery_error,
                        best.best_objective,
                        best.rel_l2,
                        missing,
                        best.solve_time,
                        basename(best_dir),
                        "best of $(length(sd_dirs)) config(s) @ degree $best_deg",
                    ),
                )
            end
        end

        # ---- subdivision=on, sparsify=off : per-axis audit ----
        audit = _best_audit_run(audit_root, system)
        if audit === nothing
            push!(
                cells,
                BenefitCell(
                    system,
                    true,
                    false,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    "—",
                    "no per-axis audit run",
                ),
            )
            push!(gaps, "[$system] no subdivision audit run found")
        else
            push!(
                cells,
                BenefitCell(
                    system,
                    true,
                    false,
                    missing,
                    missing,
                    audit.rel_l2,
                    audit.n_leaves,
                    audit.wall_s,
                    relpath_or(audit.run_dir),
                    "audit replays split predicates only — never solves CPs, so recovery_error is unrecorded",
                ),
            )
            push!(
                gaps,
                "[$system] subdivision-ON recovery_error NEVER RECORDED (audits don't solve) — Part C will fill this",
            )
        end

        # ---- sparsify=on (both subdivision states) ----
        spars = _sparsification_for(sparsification_root, system)
        for subdiv in (false, true)
            if spars === nothing
                push!(
                    cells,
                    BenefitCell(
                        system,
                        subdiv,
                        true,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        "—",
                        "no sparsification run on this ODE PE system",
                    ),
                )
            else
                push!(
                    cells,
                    BenefitCell(
                        system,
                        subdiv,
                        true,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        relpath_or(spars),
                        "sparsification present; PE recovery not measured here",
                    ),
                )
            end
        end
        spars === nothing && push!(
            gaps,
            "[$system] no sparsification run on the ODE system (baselines are 2D analytical only)",
        )
    end

    # Cross-cutting comparability caveat
    push!(
        gaps,
        "[ALL] subdivision-ON audits use tight (~0.02-wide) domains at degrees that " *
        "differ from the single-domain runs → existing on/off pairs are NOT a controlled comparison",
    )

    return cells, gaps
end

# Make a path repo-relative for readable reports when possible; otherwise return as-is.
function relpath_or(p::AbstractString)
    try
        return relpath(p)
    catch
        return String(p)
    end
end
