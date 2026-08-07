using Test
using GlobtimPostProcessing
using DataFrames
using LinearAlgebra

@testset "match_critical_point_sets" begin
    C1 = [[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]
    C2 = [[0.01, 0.0], [1.0, 0.98]]
    pairs, u1, u2 = match_critical_point_sets(C1, C2; tol = 0.1)
    @test length(pairs) == 2
    @test (1, 1) in pairs && (2, 2) in pairs
    @test length(u1) == 1 && u1[1] == [5.0, 5.0]   # fit-specific, no partner
    @test isempty(u2)

    # mutual exclusion: one C2 point cannot serve two C1 points
    pairs2, _, _ = match_critical_point_sets([[0.0, 0.0], [0.02, 0.0]], [[0.01, 0.0]]; tol = 0.1)
    @test length(pairs2) == 1                       # only the nearer of the two is matched
end

@testset "validate_consensus: stiff/sloppy tube on an anisotropic bowl" begin
    # stiff in x (λ≈20), sloppy in y (λ≈0.02): ratio 1000. Single minimum at the origin.
    f = x -> 10.0 * x[1]^2 + 0.01 * x[2]^2
    # two fits' copies of the CP: agree in stiff x (Δx=0.002), disagree along sloppy y (Δy=0.40)
    a = [0.001, 0.20]
    b = [-0.001, -0.20]
    res = validate_consensus([a], [b], f;
                             config = ConsensusValidationConfig(match_tol = 1.0, sloppy_ratio = 1e-2))
    @test res.n_matched == 1
    cp = res.consensus[1]

    @test cp.location ≈ [0.0, 0.0] atol = 1e-12      # midpoint = the true min
    @test cp.cp_type == :minimum
    @test cp.n_stiff == 1                            # one identifiable direction
    @test abs(cp.identifiable_dirs[1, 1]) > 0.99     # stiff dir aligned with the x-axis
    @test abs(cp.sloppy_dirs[2, 1]) > 0.99           # sloppy dir aligned with the y-axis

    @test cp.stiff_spread < 0.01                     # fits agree on the identifiable combination
    @test cp.tube_extent > 0.35                      # and disagree along the sloppy axis (the tube)
    @test cp.tube_extent > 20 * cp.stiff_spread      # genuinely anisotropic tube
    @test cp.confidence > 0.99                       # stiff agreement ≪ match_tol=1 ⇒ high confidence
    @test cp.newton_decrement < 1e-6                 # gradient vanishes at the min ⇒ stationary
end

@testset "validate_consensus: unmatched points are reported, well-conditioned CP has no tube" begin
    f = x -> x[1]^2 + x[2]^2                          # isotropic: no sloppy direction
    res = validate_consensus([[0.0, 0.0], [0.6, 0.6]], [[0.001, -0.001]], f;
                             config = ConsensusValidationConfig(match_tol = 0.1, sloppy_ratio = 1e-2))
    @test res.n_matched == 1
    @test length(res.unmatched1) == 1 && res.unmatched1[1] == [0.6, 0.6]
    cp = res.consensus[1]
    @test cp.n_stiff == 2                             # both directions stiff (isotropic)
    @test cp.tube_extent == 0.0                       # no sloppy subspace ⇒ no tube
    @test cp.stiff_spread > 0.0                       # the small disagreement lives entirely in stiff
end

@testset "validate_consensus: DataFrame overload" begin
    f = x -> 10.0 * x[1]^2 + 0.01 * x[2]^2
    df1 = DataFrame(x1 = [0.001], x2 = [0.20])
    df2 = DataFrame(x1 = [-0.001], x2 = [-0.20])
    res = validate_consensus(df1, df2, f;
                             config = ConsensusValidationConfig(match_tol = 1.0, sloppy_ratio = 1e-2))
    @test res.n_matched == 1
    @test res.consensus[1].n_stiff == 1
    @test res.consensus[1].tube_extent > 0.35
end
