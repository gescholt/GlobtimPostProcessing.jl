# test_saddle_links.jl — newton_critical_point / descend_to_minimum / morse_data /
# saddle_links on analytic landscapes (bead globopt_merged-o6q5.4).
#
# Pure ForwardDiff + LinearAlgebra; no Globtim dependency.

using Test
using GlobtimPostProcessing
using LinearAlgebra
using Random
using ForwardDiff

@testset "saddle_links: double well" begin
    # f = (x² − 1)² + y²: minima at (±1, 0), index-1 saddle at the origin.
    f(p) = (p[1]^2 - 1)^2 + p[2]^2
    lo, hi = [-2.0, -2.0], [2.0, 2.0]

    # Newton on ∇f = 0 from a point near the origin reaches the SADDLE, not a minimum
    x, ok, gn = newton_critical_point(f, [0.05, 0.1]; lo = lo, hi = hi)
    @test ok
    @test norm(x) < 1e-8
    @test gn < 1e-8
    idx, eigs, vecs, asym = morse_data(f, x)
    @test idx == 1
    @test asym < 1e-12
    @test eigs[1] < 0 < eigs[2]

    # from near a minimum it reaches that minimum
    xm, okm, _ = newton_critical_point(f, [0.8, 0.3]; lo = lo, hi = hi)
    @test okm
    @test isapprox(xm, [1.0, 0.0]; atol = 1e-8)
    @test morse_data(f, xm)[1] == 0

    # descent from either side of the saddle lands on the respective minimum
    @test isapprox(descend_to_minimum(f, [0.1, 0.0]; lo = lo, hi = hi), [1.0, 0.0]; atol = 1e-4)
    @test isapprox(descend_to_minimum(f, [-0.1, 0.0]; lo = lo, hi = hi), [-1.0, 0.0]; atol = 1e-4)

    minima = [(theta = [-1.0, 0.0],), (theta = [1.0, 0.0],)]
    saddles = [(theta = x, eigenvalues = eigs, eigenvectors = vecs)]
    links = saddle_links(f, saddles, minima; lo = lo, hi = hi)
    @test length(links) == 1
    L = links[1]
    @test L.saddle == 1
    @test L.distinct
    @test Set((L.min_i, L.min_j)) == Set((1, 2))
    @test length(L.endpoints) == 2

    # a point with no negative eigenvalue is skipped
    @test isempty(saddle_links(f, [(theta = xm, eigenvalues = morse_data(f, xm)[2], eigenvectors = morse_data(f, xm)[3])], minima; lo = lo, hi = hi))
end

@testset "lm_critical_point: damped, reaches the saddle where Newton overshoots" begin
    # log-compressed double well: f = log1p(100 ((x²−1)² + y²)); same critical points
    # (origin saddle, (±1,0) minima), but far from them the quadratic model is poor.
    f(p) = log1p(100 * ((p[1]^2 - 1)^2 + p[2]^2))
    lo, hi = [-2.0, -2.0], [2.0, 2.0]
    for start in ([0.3, 0.4], [-0.25, 0.6], [0.1, -0.9])
        x, ok, gn = lm_critical_point(f, start; lo = lo, hi = hi, iters = 200, gtol = 1e-9, step_max = 0.5)
        @test ok
        @test gn < 1e-8
        idx = morse_data(f, x)[1]
        @test idx in (0, 1)
        if idx == 1
            @test norm(x) < 1e-7
        else
            @test abs(abs(x[1]) - 1) < 1e-7 && abs(x[2]) < 1e-7
        end
    end
    # a NaN-returning gradient (plateau) rejects the step rather than converging on it
    g_plateau(x) = x[1] > 1.5 ? fill(NaN, 2) : ForwardDiff.gradient(f, x)
    x, ok, _ = lm_critical_point(f, [0.2, 0.1]; lo = lo, hi = hi, grad = g_plateau, step_max = 0.5)
    @test ok
    @test x[1] <= 1.5
    # step cap is honoured on the first step
    x1, _, _ = lm_critical_point(f, [0.3, 0.4]; lo = lo, hi = hi, iters = 1, step_max = 0.01)
    @test norm(x1 .- [0.3, 0.4]) <= 0.01 + 1e-12
end

@testset "minmode_saddle: climbs out of a basin to the index-1 saddle" begin
    f(p) = log1p(100 * ((p[1]^2 - 1)^2 + p[2]^2))
    lo, hi = [-2.0, -2.0], [2.0, 2.0]
    # starts inside the right basin: a gradient-norm minimiser would return the minimum
    # (or stall); min-mode following climbs to the saddle at the origin
    for start in ([0.6, 0.1], [0.8, -0.3], [-0.5, 0.2])
        x, ok, gn, nN = minmode_saddle(f, start; lo = lo, hi = hi, iters = 500, gtol = 1e-9, step_max = 0.1)
        @test ok
        @test gn < 1e-8
        @test norm(x) < 1e-7
        @test morse_data(f, x)[1] == 1
        @test nN >= 1          # the Newton endgame engaged
    end
    # 3-D rotated well with a 2-D saddle set: still index 1 at the rotated origin
    rng = MersenneTwister(5)
    Q, _ = qr(randn(rng, 3, 3)); Q = Matrix(Q)
    c = [0.1, -0.2, 0.05]
    f3(p) = (u = Q' * (p .- c); (u[1]^2 - 1)^2 + 3u[2]^2 + 0.7u[3]^2)
    x3, ok3, _, _ = minmode_saddle(f3, c .+ Q * [0.7, 0.2, -0.1]; lo = fill(-3.0, 3), hi = fill(3.0, 3), iters = 500, gtol = 1e-9, step_max = 0.1)
    @test ok3
    @test isapprox(x3, c; atol = 1e-6)
    # plateau-aware gradient: a NaN region is never entered
    g_plateau(x) = x[1] < -0.5 ? fill(NaN, 2) : ForwardDiff.gradient(f, x)
    xp, okp, _, _ = minmode_saddle(f, [0.6, 0.1]; lo = lo, hi = hi, grad = g_plateau, step_max = 0.1)
    @test okp
    @test xp[1] >= -0.5
end

@testset "saddle_links: flat trough and same-minimum" begin
    # f = y²: no saddle anywhere (a trough along x); morse index of (0.3, 0) has a zero eigenvalue
    g(p) = p[2]^2
    lo, hi = [-1.0, -1.0], [1.0, 1.0]
    idx, eigs, vecs, _ = morse_data(g, [0.3, 0.0])
    @test idx == 0
    @test abs(eigs[1]) < 1e-12
    @test isempty(saddle_links(g, [(theta = [0.3, 0.0], eigenvalues = eigs, eigenvectors = vecs)], [(theta = [0.0, 0.0],)]; lo = lo, hi = hi))

    # a "saddle" whose two descents land on the SAME supplied minimum is not distinct:
    # f = (x² − 1)² + y² with only the right minimum supplied → left descent matches nothing
    f(p) = (p[1]^2 - 1)^2 + p[2]^2
    lo2, hi2 = [-2.0, -2.0], [2.0, 2.0]
    s = [(theta = [0.0, 0.0], eigenvalues = morse_data(f, [0.0, 0.0])[2], eigenvectors = morse_data(f, [0.0, 0.0])[3])]
    links = saddle_links(f, s, [(theta = [1.0, 0.0],)]; lo = lo2, hi = hi2)
    @test length(links) == 1
    @test !links[1].distinct
    @test 0 in (links[1].min_i, links[1].min_j)
end

@testset "saddle_links: rotated 3-D double well, injectable descend" begin
    rng = MersenneTwister(7)
    Q, _ = qr(randn(rng, 3, 3)); Q = Matrix(Q)
    c = [0.2, -0.1, 0.3]
    # double well along the first rotated axis, bowls along the others
    f(p) = (u = Q' * (p .- c); (u[1]^2 - 1)^2 + 2u[2]^2 + 0.5u[3]^2)
    lo, hi = fill(-3.0, 3), fill(3.0, 3)
    m1 = c .+ Q * [-1.0, 0.0, 0.0]; m2 = c .+ Q * [1.0, 0.0, 0.0]
    s, ok, _ = newton_critical_point(f, c .+ Q * [0.1, 0.05, -0.05]; lo = lo, hi = hi)
    @test ok
    @test isapprox(s, c; atol = 1e-7)
    idx, eigs, vecs, _ = morse_data(f, s)
    @test idx == 1
    # the negative eigenvector is the rotated first axis, not an axis of R^3
    @test abs(abs(dot(vecs[:, 1], Q[:, 1])) - 1) < 1e-8
    links = saddle_links(f, [(theta = s, eigenvalues = eigs, eigenvectors = vecs)], [(theta = m1,), (theta = m2,)]; lo = lo, hi = hi)
    @test links[1].distinct

    # injectable descend: a stub that always returns m1 makes both ends identical → not distinct
    stub = saddle_links(f, [(theta = s, eigenvalues = eigs, eigenvectors = vecs)], [(theta = m1,), (theta = m2,)];
        lo = lo, hi = hi, descend = x -> m1)
    @test !stub[1].distinct
    @test stub[1].min_i == stub[1].min_j == 1
end

@testset "whitened wrapper preserves the Morse index" begin
    rng = MersenneTwister(11)
    n = 4
    # indefinite quadratic with index 2
    D = Diagonal([3.0, 1.0, -2.0, -0.5])
    R, _ = qr(randn(rng, n, n)); R = Matrix(R)
    H = R * D * R'
    f(p) = 0.5 * dot(p, H * p)
    A = randn(rng, n, n) + 3I       # invertible, not orthonormal
    c = randn(rng, n)
    g(u) = f(c .+ A * u)
    u0 = -A \ c                      # maps to p = 0, the critical point of f
    @test morse_data(f, zeros(n))[1] == 2
    @test morse_data(g, u0)[1] == 2
    # Newton in the whitened frame reaches the same point
    u, ok, _ = newton_critical_point(g, u0 .+ 0.1 * randn(rng, n); lo = fill(-10.0, n), hi = fill(10.0, n))
    @test ok
    @test norm(c .+ A * u) < 1e-7
end
