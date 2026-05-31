"""
    MetricsLoader.jl

Loader for JSONL metrics files written by `Globtim.Metrics.MetricsLogger`.

A metrics file mixes four row kinds — `meta`, `phase`, `leaf`, `cp` — with
different field schemas. `load_metrics(path)` parses the file once and returns
each kind as its own DataFrame (plus the singleton `meta` row as a Dict).
Columns are the union of fields observed across rows of that kind; rows that
omit an optional field get `missing`.
"""

"""
    load_metrics(path) -> NamedTuple

Parse the JSONL metrics file at `path` (written by
`Globtim.Metrics.MetricsLogger`) and return

    (; meta::Dict{String,Any},
       phases::DataFrame,
       leaves::DataFrame,
       cps::DataFrame,
       path::String,
       n_rows::Int)

# Field handling

  * Every non-empty line must be a JSON object with a `kind` field in
    `{"meta", "phase", "leaf", "cp"}`. Lines with any other `kind` are
    dropped (silently — extend later if needed).
  * Each output DataFrame's columns are the *union* of keys observed in rows
    of that kind. Rows missing a given key receive `missing`. This keeps the
    loader forward-compatible with new emitter fields without code changes.
  * For `leaves`, the on-disk `bounds::Vector{Vector{Float64}}` is converted
    to `Vector{Tuple{Float64,Float64}}` for ergonomic indexing.
  * `meta` is returned as a `Dict{String,Any}` rather than a 1-row DataFrame.
    The writer guarantees exactly one meta row per file; multiple meta rows
    throw an `ArgumentError`.

# Errors

  * Missing file → throws `SystemError`.
  * A row without a `kind` field → throws `ArgumentError` (with line number).
  * More than one `meta` row → throws `ArgumentError`.

# Example

```julia
using GlobtimPostProcessing
r = load_metrics("_generated/my_run/metrics.jsonl")
r.meta["run_id"]                              # "run-20260530-204000"
filter(:status => ==("active"), r.leaves)     # only active leaves
maximum(r.phases.wall_s |> skipmissing)        # longest phase
```
"""
function load_metrics(path::AbstractString)
    isfile(path) || throw(SystemError("metrics file not found: $path"))
    meta_rows = Dict{String,Any}[]
    phase_rows = Dict{String,Any}[]
    leaf_rows = Dict{String,Any}[]
    cp_rows = Dict{String,Any}[]
    n_rows = 0
    open(path, "r") do io
        for (lineno, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            row = JSON.parse(line)
            row isa AbstractDict || throw(ArgumentError(
                "metrics row at $path:$lineno is not a JSON object: $line",
            ))
            kind = get(row, "kind", nothing)
            kind === nothing && throw(ArgumentError(
                "metrics row at $path:$lineno has no `kind` field: $line",
            ))
            n_rows += 1
            if kind == "meta"
                push!(meta_rows, row)
            elseif kind == "phase"
                push!(phase_rows, row)
            elseif kind == "leaf"
                push!(leaf_rows, row)
            elseif kind == "cp"
                push!(cp_rows, row)
            end
        end
    end
    length(meta_rows) > 1 && throw(ArgumentError(
        "metrics file $path has $(length(meta_rows)) meta rows (expected 1)",
    ))
    meta = isempty(meta_rows) ? Dict{String,Any}() : meta_rows[1]

    leaves = _metrics_rows_to_dataframe(leaf_rows)
    if "bounds" in names(leaves)
        leaves[!, :bounds] = [_to_tuple_bounds(b) for b in leaves.bounds]
    end

    return (
        meta = meta,
        phases = _metrics_rows_to_dataframe(phase_rows),
        leaves = leaves,
        cps = _metrics_rows_to_dataframe(cp_rows),
        path = String(path),
        n_rows = n_rows,
    )
end

# Build a DataFrame from a Vector{Dict} where rows may have different keys.
# Column order is first-seen order across the input rows.
function _metrics_rows_to_dataframe(rows::Vector{Dict{String,Any}})
    isempty(rows) && return DataFrame()
    seen = Set{String}()
    all_keys = String[]
    for r in rows
        for k in keys(r)
            if !(k in seen)
                push!(seen, k)
                push!(all_keys, k)
            end
        end
    end
    df = DataFrame()
    for k in all_keys
        df[!, Symbol(k)] = Any[get(r, k, missing) for r in rows]
    end
    return df
end

_to_tuple_bounds(::Missing) = missing
function _to_tuple_bounds(b)
    b isa AbstractVector || return b
    return [(Float64(bi[1]), Float64(bi[2])) for bi in b]
end
