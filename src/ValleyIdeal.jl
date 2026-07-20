# ValleyIdeal.jl
# Approximate vanishing ideal of a sampled positive-dimensional set (valley).
#
# The algebraic layer of the non-zero-dimensional minima program (VALLEY-IDEAL
# epic, Route A): given point samples of a valley — a ValleyWalking trace, a
# manifold_resolve sample set, or a cluster of equi-valued degenerate critical
# points — recover the polynomial system that (approximately) defines it.
#
# Method (Breiding–Kališnik–Sturmfels "Learning algebraic varieties from
# samples" flavor): for each total degree D ≤ degree_max, build the evaluation
# matrix M[i, j] = m_j(x̂_i) of all monomials of degree ≤ D at the (affine-
# normalized) samples and take its numerical kernel via SVD. Kernel vectors are
# coefficient vectors of polynomials vanishing on the samples. The sequence of
# kernel dimensions across degrees is the truncated Hilbert function of the
# ideal; minimal generators at degree D are the kernel directions NOT spanned
# by monomial multiples m·g of generators found at lower degree.
#
# Interpolation is dimension-agnostic: it works for k ≥ 2 valleys (where the
# curve-walking machinery cannot trace) as long as the samples cover the set.

using LinearAlgebra: norm, svd, Diagonal

# ─── Configuration and result types ──────────────────────────────────────────

"""
Configuration for approximate vanishing-ideal recovery.

# Fields
- `degree_max`: Largest total degree probed (default: 4)
- `rank_tol`: Relative singular-value threshold — `σ < rank_tol·σ_max` counts as
  kernel (default: 1e-8). The POC showed genuine kernels sit at ~1e-13, so this
  leaves a wide margin on both sides.
- `span_tol`: Relative threshold for the rank of the generator-multiple span
  when separating new generators from multiples of old ones (default: 1e-10)
- `normalize`: Affine-map samples to `[-1, 1]^n` before building the evaluation
  matrix (default: true). Strongly recommended — monomial conditioning degrades
  fast away from the unit box. Generators are reported in normalized
  coordinates; use [`physical_generator`](@ref) to convert back.
"""
@kwdef struct ValleyIdealConfig
    degree_max::Int = 4
    rank_tol::Float64 = 1e-8
    span_tol::Float64 = 1e-10
    normalize::Bool = true
end

"""
One recovered generator of the (approximate) vanishing ideal.

Coefficients are in the graded monomial basis of the **normalized** coordinates
`ẑ = (x − center) ./ halfwidth` when the config had `normalize = true` (the
default); otherwise in physical coordinates. `residual` is
`maxᵢ |g(x̂ᵢ)| / ‖coeffs‖` over the samples — the certificate that the
polynomial really vanishes on the set.
"""
struct ValleyIdealGenerator
    degree::Int
    exponents::Vector{Vector{Int}}
    coeffs::Vector{Float64}
    residual::Float64
end

"""
Result of [`recover_vanishing_ideal`](@ref).

# Fields
- `n_points`, `n_dims`: sample-cloud shape
- `center`, `halfwidth`: the affine normalization `ẑ = (x − center) ./ halfwidth`
  (identity when the config had `normalize = false`)
- `kernel_dims[D]`: numerical kernel dimension of the degree-≤D evaluation
  matrix — the truncated Hilbert function `dim I_{≤D}` of the ideal
- `multiple_dims[D]`: dimension of the span of monomial multiples of generators
  found below degree D — `kernel_dims[D] - multiple_dims[D]` is the number of
  new minimal generators at degree D
- `sigma_gaps[D]`: `σ_min/σ_max` of the evaluation matrix (kernel sharpness)
- `generators`: minimal generators, ordered by degree
"""
struct ValleyIdealResult
    n_points::Int
    n_dims::Int
    center::Vector{Float64}
    halfwidth::Vector{Float64}
    kernel_dims::Vector{Int}
    multiple_dims::Vector{Int}
    sigma_gaps::Vector{Float64}
    generators::Vector{ValleyIdealGenerator}
end

# ─── Monomial machinery ──────────────────────────────────────────────────────

# All exponent vectors of total degree ≤ D in n variables, graded lexicographic.
function _monomial_exponents(n::Int, D::Int)
    out = Vector{Vector{Int}}()
    function rec(prefix::Vector{Int}, remaining::Int, budget::Int)
        if remaining == 1
            for d in 0:budget
                push!(out, vcat(prefix, d))
            end
        else
            for d in 0:budget
                rec(vcat(prefix, d), remaining - 1, budget - d)
            end
        end
    end
    rec(Int[], n, D)
    sort!(out; by = e -> (sum(e), e))
    return out
end

_monomial_string(e::Vector{Int}) =
    sum(e) == 0 ? "1" :
    join(["x$(i)" * (e[i] > 1 ? "^$(e[i])" : "") for i in eachindex(e) if e[i] > 0], "*")

"""
    generator_string(gen::ValleyIdealGenerator; drop = 1e-6) -> String

Human-readable form of a generator, coefficients normalized so the largest is
1, terms below `drop` (relative) omitted.
"""
function generator_string(gen::ValleyIdealGenerator; drop::Float64 = 1e-6)
    c = gen.coeffs ./ gen.coeffs[argmax(abs.(gen.coeffs))]
    terms = String[]
    for (ci, e) in zip(c, gen.exponents)
        abs(ci) < drop && continue
        push!(
            terms,
            string(
                ci >= 0 ? "+" : "-",
                round(abs(ci); sigdigits = 5),
                "·",
                _monomial_string(e),
            ),
        )
    end
    return join(terms, " ")
end

# Evaluation matrix M[i, j] = m_j(points[i]) for the given exponent list.
function _evaluation_matrix(points::Vector{Vector{Float64}}, exps::Vector{Vector{Int}})
    n = length(points[1])
    M = Matrix{Float64}(undef, length(points), length(exps))
    @inbounds for (j, e) in enumerate(exps), (i, p) in enumerate(points)
        v = 1.0
        for d in 1:n
            v *= p[d]^e[d]
        end
        M[i, j] = v
    end
    return M
end

# ─── Core recovery ───────────────────────────────────────────────────────────

"""
    recover_vanishing_ideal(points; config = ValleyIdealConfig()) -> ValleyIdealResult

Recover the approximate vanishing ideal of a sample cloud `points`
(a vector of coordinate vectors — a valley trace, manifold samples, or a
cluster of equi-valued degenerate critical points).

For each degree `D ≤ config.degree_max` the numerical kernel of the monomial
evaluation matrix gives all polynomials of degree ≤ D vanishing on the samples;
minimal generators are separated from monomial multiples of lower-degree
generators by projection. Raises when the sample count is too small for the
requested degree (the kernel would contain interpolation artifacts).
"""
function recover_vanishing_ideal(
    points::AbstractVector{<:AbstractVector{<:Real}};
    config::ValleyIdealConfig = ValleyIdealConfig(),
)
    isempty(points) && error("recover_vanishing_ideal: empty sample cloud")
    pts = [collect(Float64, p) for p in points]
    n = length(pts[1])
    all(p -> length(p) == n, pts) ||
        error("recover_vanishing_ideal: inconsistent point dimensions")

    # Affine normalization ẑ = (x − center) ./ halfwidth onto [-1, 1]^n.
    # Coordinates that are constant on the cloud (e.g. a valley confined to a
    # hyperplane x_i = c) get a floor halfwidth — their normalized value is 0.
    if config.normalize
        lo = [minimum(p[d] for p in pts) for d in 1:n]
        hi = [maximum(p[d] for p in pts) for d in 1:n]
        center = (lo .+ hi) ./ 2
        halfwidth = max.((hi .- lo) ./ 2, 1e-8)
        pts = [(p .- center) ./ halfwidth for p in pts]
    else
        center = zeros(n)
        halfwidth = ones(n)
    end

    all_exps = _monomial_exponents(n, config.degree_max)
    n_monomials = length(all_exps)
    length(pts) > n_monomials || error(
        "recover_vanishing_ideal: $(length(pts)) samples for $n_monomials monomials " *
        "(degree ≤ $(config.degree_max), $n vars) — the kernel would contain " *
        "interpolation artifacts. Supply > $n_monomials samples or lower degree_max.",
    )
    exp_index = Dict(e => j for (j, e) in enumerate(all_exps))

    kernel_dims = Int[]
    multiple_dims = Int[]
    sigma_gaps = Float64[]
    generators = ValleyIdealGenerator[]

    for D in 1:config.degree_max
        exps = filter(e -> sum(e) <= D, all_exps)
        nb = length(exps)
        M = _evaluation_matrix(pts, exps)

        # Column scaling for conditioning across degrees. An exactly-zero
        # column (monomial involving a degenerate coordinate ẑᵢ ≡ 0) gets
        # scale 1: any positive floor would AMPLIFY kernel-vector components
        # on those coordinates during un-scaling and drown the genuine
        # generators (the circle-in-a-hyperplane failure mode).
        colnorm = [
            let s = norm(view(M, :, j))
                s == 0.0 ? 1.0 : s
            end for j in 1:nb
        ]
        sv = svd(M ./ colnorm')
        push!(sigma_gaps, sv.S[end] / sv.S[1])
        k = count(<(config.rank_tol * sv.S[1]), sv.S)
        push!(kernel_dims, k)

        # Kernel basis as coefficient vectors in the (unscaled) monomial basis.
        V = Matrix{Float64}(undef, nb, k)
        for i in 1:k
            v = sv.V[:, end-i+1] ./ colnorm
            V[:, i] = v ./ norm(v)
        end

        # Span of monomial multiples m·g of already-found generators, deg ≤ D.
        prods = Vector{Vector{Float64}}()
        for g in generators
            for m in _monomial_exponents(n, D - g.degree)
                v = zeros(nb)
                for (c, e) in zip(g.coeffs, g.exponents)
                    v[exp_index[e.+m]] += c
                end
                push!(prods, v ./ norm(v))
            end
        end

        r = 0
        if k > 0 && !isempty(prods)
            P = reduce(hcat, prods)
            psv = svd(P)
            r = count(>(config.span_tol * psv.S[1]), psv.S)
            Q = psv.U[:, 1:r]
            V = V .- Q * (Q' * V)   # residuals still lie in the kernel subspace
        end
        push!(multiple_dims, r)

        # New minimal generators: the k − r strongest residual directions.
        n_new = k - r
        if n_new > 0
            rsv = svd(V)
            for i in 1:n_new
                u = rsv.U[:, i]
                res = maximum(abs.(M * u)) / norm(u)
                push!(generators, ValleyIdealGenerator(D, exps, u, res))
            end
        end
    end

    return ValleyIdealResult(
        length(pts),
        n,
        center,
        halfwidth,
        kernel_dims,
        multiple_dims,
        sigma_gaps,
        generators,
    )
end

"""
    recover_vanishing_ideal(trace::ValleyTraceResult; config) -> ValleyIdealResult

Convenience overload: recover the vanishing ideal of a valley traced by
[`trace_valley`](@ref), using both walk directions as the sample cloud.
"""
function recover_vanishing_ideal(
    trace::ValleyTraceResult;
    config::ValleyIdealConfig = ValleyIdealConfig(),
)
    pts = vcat(reverse(trace.path_negative), trace.path_positive)
    return recover_vanishing_ideal(pts; config = config)
end

# ─── Back-conversion to physical coordinates ─────────────────────────────────

"""
    physical_generator(gen, result; drop = 1e-10, clean = 1e-9) -> (exponents, coeffs)

Convert a generator from normalized coordinates `ẑ = (x − c) ./ h` back to
physical coordinates by expanding `g((x − c) ./ h)` into monomials in `x`.
Coefficients are normalized so the largest magnitude is 1; terms below `drop`
(relative) are removed. Returns graded-sorted exponent vectors and coefficients.

Normalized coefficients below `clean` (relative) are zeroed BEFORE expansion:
the conversion multiplies the coefficient of a degree-k monomial in `ẑᵢ` by
`hᵢ^{-k}`, so on a degenerate axis (constant coordinate, halfwidth floored near
1e-8) numerical dust at 1e-16 in the normalized basis would be amplified to
O(1) in physical coordinates and drown the true terms.
"""
function physical_generator(
    gen::ValleyIdealGenerator,
    result::ValleyIdealResult;
    drop::Float64 = 1e-10,
    clean::Float64 = 1e-9,
)
    n = result.n_dims
    c = result.center
    h = result.halfwidth
    cleaned = copy(gen.coeffs)
    cleaned[abs.(cleaned).<clean*maximum(abs, cleaned)] .= 0.0
    acc = Dict{Vector{Int},Float64}()
    for (coef, e) in zip(cleaned, gen.exponents)
        # Expand ∏ᵢ ((xᵢ − cᵢ)/hᵢ)^{eᵢ} term by term via per-variable binomials.
        terms = Dict(zeros(Int, n) => coef)
        for i in 1:n
            k = e[i]
            k == 0 && continue
            uni = [binomial(k, j) * (-c[i])^(k - j) / h[i]^k for j in 0:k]
            nxt = Dict{Vector{Int},Float64}()
            for (te, tc) in terms, j in 0:k
                ne = copy(te)
                ne[i] += j
                nxt[ne] = get(nxt, ne, 0.0) + tc * uni[j+1]
            end
            terms = nxt
        end
        for (te, tc) in terms
            acc[te] = get(acc, te, 0.0) + tc
        end
    end
    cmax = maximum(abs, values(acc))
    kept = [(e, v / cmax) for (e, v) in acc if abs(v / cmax) >= drop]
    sort!(kept; by = t -> (sum(t[1]), t[1]))
    return [t[1] for t in kept], [t[2] for t in kept]
end

# ─── Display ─────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", r::ValleyIdealResult)
    println(io, "ValleyIdealResult: $(r.n_points) samples in ℝ^$(r.n_dims)")
    println(io, "  degree  kernel  multiples  new  σ_min/σ_max")
    for D in eachindex(r.kernel_dims)
        new = r.kernel_dims[D] - r.multiple_dims[D]
        println(
            io,
            "    ≤$D      $(r.kernel_dims[D])        $(r.multiple_dims[D])       $new   $(round(r.sigma_gaps[D]; sigdigits=3))",
        )
    end
    if isempty(r.generators)
        print(io, "  no vanishing polynomials up to the probed degree")
    else
        println(io, "  minimal generators (normalized coordinates):")
        for g in r.generators
            println(
                io,
                "    deg $(g.degree)  residual $(round(g.residual; sigdigits=3)):  0 ≈ $(generator_string(g))",
            )
        end
    end
end
