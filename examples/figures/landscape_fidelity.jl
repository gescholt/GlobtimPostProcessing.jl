# landscape_fidelity.jl
# Regenerate the landscape-fidelity illustration for GlobtimPostProcessing docs.
#
# Run with:
#   julia --project=profiles/viz pkg/globtimpostprocessing/examples/figures/landscape_fidelity.jl
#
# Output: pkg/globtimpostprocessing/docs/src/assets/landscape_fidelity.png
#
# Loads the deuflhard_2d_ext refinement_comparison CSV (deg 10) and overlays:
#   • objective contour
#   • raw polynomial CPs (★)
#   • refined CPs coloured by Morse type (●, ▲, ▼)
#   • arrows from raw → refined for points that moved
#   • Hessian basin ellipse around p_true
#
# Requires GlobtimPostProcessing + Globtim (Deuflhard) + CSV + DataFrames + CairoMakie.

using Globtim: Deuflhard
using GlobtimPostProcessing: estimate_basin_radius
using CSV, DataFrames
using ForwardDiff
using LinearAlgebra
using CairoMakie

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", "..", ".."))
const CSV_PATH  = joinpath(REPO_ROOT,
    "globtim_results", "cluster", "deuflhard_2d", "refinement_comparison_deg_10.csv")
const ASSET_DIR = abspath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets"))
isdir(ASSET_DIR) || mkpath(ASSET_DIR)

isfile(CSV_PATH) || error("Missing refinement CSV: $CSV_PATH")

println("Building landscape-fidelity figure (Deuflhard 2D, deg 10) …")
df = CSV.read(CSV_PATH, DataFrame)
println("  $(nrow(df)) (raw, refined) pairs")

# Deuflhard global minima (paper config): two symmetric points on x+y=0 ∩ x²+y²=log(3)
r  = sqrt(log(3) / 2)
p_true     = [+r, -r]
p_true_sym = [-r, +r]

# ─── Contour grid ────────────────────────────────────────────────────────────
xs = range(-1.2, 1.2; length = 220)
ys = range(-1.2, 1.2; length = 220)
Z  = [Deuflhard([x, y]) for x in xs, y in ys]
# Log-spaced levels so the four wells are visible against the steep walls
zmin, zmax = minimum(Z), maximum(Z)
levels = exp.(range(log(max(zmin, 1e-6)), log(zmax); length = 22))

# ─── Hessian basin ellipse at p_true ─────────────────────────────────────────
H_at_p = ForwardDiff.hessian(Deuflhard, p_true)
H_sym  = Symmetric(H_at_p)
λ, V   = eigen(H_sym)
threshold_factor = 0.1
f_at_p = Deuflhard(p_true)
Δf_threshold = abs(f_at_p) < 1e-10 ? threshold_factor : threshold_factor * abs(f_at_p)
# Semi-axes from quadratic level set ½ λᵢ rᵢ² = Δf  ⇒  rᵢ = √(2 Δf / λᵢ)
semi_axes = sqrt.(2 * Δf_threshold ./ λ)
θ = range(0, 2π; length = 200)
ellipse_x = [p_true[1] + semi_axes[1] * cos(t) * V[1, 1] + semi_axes[2] * sin(t) * V[1, 2] for t in θ]
ellipse_y = [p_true[2] + semi_axes[1] * cos(t) * V[2, 1] + semi_axes[2] * sin(t) * V[2, 2] for t in θ]
r_basin   = estimate_basin_radius(p_true, Deuflhard, Matrix(H_at_p); threshold_factor)
println("  basin radius at p_true = $(round(r_basin, sigdigits=3)); semi-axes = $(round.(semi_axes, sigdigits=3))")

# ─── Split refined CPs by Morse type ─────────────────────────────────────────
is_type(t)    = row -> coalesce(row.critical_point_type == t, false)
mins      = filter(is_type("minimum"),    df)
saddles   = filter(is_type("saddle"),     df)
maxes     = filter(is_type("maximum"),    df)
others    = filter(row -> coalesce(row.critical_point_type ∉ ("minimum","saddle","maximum"), true), df)
println("  refined minima=$(nrow(mins))  saddles=$(nrow(saddles))  maxima=$(nrow(maxes))  other=$(nrow(others))")

# ─── Identify pairs that actually moved during refinement ────────────────────
function moved_pairs(rows; min_disp = 0.04)
    raws = Tuple{Float64,Float64}[]
    refs = Tuple{Float64,Float64}[]
    for row in eachrow(rows)
        dx = row.refined_dim1 - row.raw_dim1
        dy = row.refined_dim2 - row.raw_dim2
        hypot(dx, dy) > min_disp || continue
        push!(raws, (row.raw_dim1, row.raw_dim2))
        push!(refs, (row.refined_dim1, row.refined_dim2))
    end
    return raws, refs
end
raw_moved, ref_moved = moved_pairs(df)
println("  $(length(raw_moved)) raw→refined pairs moved by > 0.04")

# ─── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(size = (1300, 640), fontsize = 14)

Label(fig[0, 1:4],
    "Landscape fidelity — Deuflhard 2D (degree 10 polynomial approximation)";
    fontsize = 17, font = :bold, halign = :center)

# Panel A: full landscape with all CPs
axA = Axis(fig[1, 1];
    title  = "(A) Full landscape — polynomial CPs and their refined counterparts",
    xlabel = "x₁", ylabel = "x₂",
    aspect = DataAspect(),
    limits = ((-1.2, 1.2), (-1.2, 1.2)),
)
cf = contourf!(axA, xs, ys, Z;
    levels = levels, colormap = (:viridis, 0.85))
contour!(axA, xs, ys, Z;
    levels = levels, color = (:white, 0.18), linewidth = 0.4)

# Raw polynomial CPs — gray stars (semi-transparent)
scatter!(axA, df.raw_dim1, df.raw_dim2;
    marker = :star5, markersize = 11,
    color = (:gray80, 0.55), strokecolor = :gray30, strokewidth = 0.8,
    label = "Polynomial CP (raw)")
# Refined CPs by Morse type
scatter!(axA, mins.refined_dim1, mins.refined_dim2;
    marker = :circle, markersize = 11,
    color = :steelblue, strokecolor = :white, strokewidth = 0.8,
    label = "Refined minimum")
scatter!(axA, saddles.refined_dim1, saddles.refined_dim2;
    marker = :utriangle, markersize = 11,
    color = :mediumpurple, strokecolor = :white, strokewidth = 0.8,
    label = "Refined saddle")
scatter!(axA, maxes.refined_dim1, maxes.refined_dim2;
    marker = :dtriangle, markersize = 11,
    color = :crimson, strokecolor = :white, strokewidth = 0.8,
    label = "Refined maximum")
# (raw → refined arrows are deferred to Panel B's zoom — too busy at full scale)
# Ground-truth minima
scatter!(axA, [p_true[1], p_true_sym[1]], [p_true[2], p_true_sym[2]];
    marker = :star8, markersize = 24,
    color = :gold, strokecolor = :black, strokewidth = 1.4,
    label = "p* (ground truth)")
axislegend(axA; position = :lt, framevisible = true,
    backgroundcolor = (:white, 0.85), labelsize = 10, patchsize = (16, 16))
Colorbar(fig[1, 2], cf; label = "f(x)", width = 12)

# Panel B: zoom around p_true, with Hessian basin ellipse and refinement arrows
zoom_r  = 0.35
axB = Axis(fig[1, 3];
    title  = "(B) Basin around p* — Hessian quadratic ellipse, refinement paths",
    xlabel = "x₁", ylabel = "x₂",
    aspect = DataAspect(),
    limits = ((p_true[1] - zoom_r, p_true[1] + zoom_r), (p_true[2] - zoom_r, p_true[2] + zoom_r)),
)
contourf!(axB, xs, ys, Z;
    levels = levels, colormap = (:viridis, 0.9))
contour!(axB, xs, ys, Z;
    levels = levels, color = (:white, 0.25), linewidth = 0.5)
# Basin ellipse (filled translucent + boundary)
poly!(axB, Point2f.(ellipse_x, ellipse_y);
    color = (:gold, 0.18), strokecolor = (:gold, 0.85), strokewidth = 2.0,
    label = "Hessian basin (Δf = 10% of |f*|)")
# Raw CPs falling inside the zoom box
in_zoom = (df.raw_dim1 .> p_true[1] - zoom_r) .& (df.raw_dim1 .< p_true[1] + zoom_r) .&
          (df.raw_dim2 .> p_true[2] - zoom_r) .& (df.raw_dim2 .< p_true[2] + zoom_r)
zdf = df[in_zoom, :]
scatter!(axB, zdf.raw_dim1, zdf.raw_dim2;
    marker = :star5, markersize = 14,
    color = (:gray80, 0.6), strokecolor = :gray30, strokewidth = 1.0,
    label = "Polynomial CP (raw)")
# Same-zoom refined CPs by type
zmins    = filter(row -> coalesce(row.critical_point_type == "minimum", false), zdf)
zsaddles = filter(row -> coalesce(row.critical_point_type == "saddle",  false), zdf)
scatter!(axB, zmins.refined_dim1, zmins.refined_dim2;
    marker = :circle, markersize = 14,
    color = :steelblue, strokecolor = :white, strokewidth = 1.0,
    label = "Refined minimum")
scatter!(axB, zsaddles.refined_dim1, zsaddles.refined_dim2;
    marker = :utriangle, markersize = 14,
    color = :mediumpurple, strokecolor = :white, strokewidth = 1.0,
    label = "Refined saddle")
# Refinement arrows within zoom
for row in eachrow(zdf)
    dx = row.refined_dim1 - row.raw_dim1
    dy = row.refined_dim2 - row.raw_dim2
    hypot(dx, dy) > 0.005 || continue
    arrows!(axB, [row.raw_dim1], [row.raw_dim2], [dx], [dy];
        color = (:black, 0.7), arrowsize = 10, linewidth = 1.2)
end
# p_true marker on top
scatter!(axB, [p_true[1]], [p_true[2]];
    marker = :star8, markersize = 30,
    color = :gold, strokecolor = :black, strokewidth = 1.6,
    label = "p* (ground truth)")
axislegend(axB; position = :lt, framevisible = true,
    backgroundcolor = (:white, 0.9), labelsize = 10, patchsize = (16, 16))
Colorbar(fig[1, 4], cf; label = "f(x)", width = 12)

# Footer
Label(fig[2, 1:4],
    "Fidelity check (rigorous): a raw polynomial CP x* lies in the basin of x_min iff ‖x* − x_min‖ < r, " *
    "with r = √(2 Δf / λ_min(H(x_min))). The gold ellipse in (B) is the basin level set " *
    "{x : ½(x − p*)ᵀ H (x − p*) ≤ Δf}; raw polynomial CPs inside it map to p* under refinement.";
    fontsize = 11, color = :gray30, halign = :center)

rowgap!(fig.layout, 1, 4)
rowgap!(fig.layout, 2, 8)

out_path = joinpath(ASSET_DIR, "landscape_fidelity.png")
save(out_path, fig; px_per_unit = 2)
println("→ wrote $(out_path)")
