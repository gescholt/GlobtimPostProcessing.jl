# N6: the index-preserving Newton census promoted from experiments/sandbox/lv_critical_census.jl.
# Guards: saddles stay saddles (the Fminbox refiner slid them onto minima), the output
# does not depend on the thread count, failures are counted and never dropped, the
# scale-free step acceptance works where the absolute gradient test cannot, and the
# Poincaré–Hopf bookkeeping and the files are what the sweep reads.

using Test
using GlobtimPostProcessing
using LinearAlgebra
using JSON

@testset "Newton census (N6)" begin
    # double well: minima at (±1, 0), index-1 saddle at the origin
    dw(x) = (x[1]^2 - 1)^2 + x[2]^2
    lb = [-2.0, -2.0]
    ub = [2.0, 2.0]

    @testset "polish_to_critical preserves the index" begin
        p, gn, it = polish_to_critical(dw, [0.05, 0.02], lb, ub)
        @test gn < 1e-10
        @test norm(p) < 1e-6            # stayed on the saddle, did not slide to a minimum
        @test it >= 1
        kind, idx, ev = critical_point_kind(dw, p)
        @test kind == :saddle && idx == 1 && length(ev) == 2
        p2, gn2, _ = polish_to_critical(dw, [0.7, 0.3], lb, ub)
        @test gn2 < 1e-10
        @test p2 ≈ [1.0, 0.0] atol = 1e-6
        @test critical_point_kind(dw, p2)[1] == :minimum
        # a seed with a non-finite gradient is returned unpolished with Inf residual
        # (dw / 0 has value Inf and NaN partials; a constant Inf would have ZERO partials,
        # which is why newton_census also checks the value — see "failures are counted")
        pinf, gninf, itinf = polish_to_critical(x -> dw(x) / (x[1] > 0 ? 0.0 : 1.0), [0.5, 0.0], lb, ub)
        @test gninf == Inf && itinf == 0
    end

    @testset "dedup_by_radius keeps provenance" begin
        pts = [[0.0, 0.0], [1e-4, 0.0], [1.0, 0.0], [0.0, 1e-4], [1.0, 1e-9]]
        keep, members = dedup_by_radius(pts, [1e-3, 1e-3])
        @test length(keep) == 2
        @test members[1] == [1, 2, 4] && members[2] == [3, 5]
    end

    raw = [[0.9, 0.1], [1.05, -0.05], [-0.95, 0.02], [0.02, -0.03], [0.9000001, 0.1],
           [-1.02, 0.01], [0.0, 0.04]]
    @testset "census: minima, saddle, duplicates, thread invariance" begin
        c = newton_census(dw, raw, lb, ub; threads = false)
        @test length(c.seeds) == 6                 # the 1e-7 duplicate is pre-deduped
        @test c.nfail == 0
        @test length(c.distinct) == 3
        kinds = sort([string(q.kind) for q in c.distinct])
        @test kinds == ["minimum", "minimum", "saddle"]
        @test all(q.interior for q in c.distinct)
        @test all(q.gnorm < 1e-8 for q in c.distinct)
        @test all(q.accepted_by == "gtol" for q in c.distinct)
        ct = newton_census(dw, raw, lb, ub; threads = true)
        @test length(ct.distinct) == 3
        for (a, b) in zip(c.distinct, ct.distinct)
            @test a.p ≈ b.p atol = 1e-12
            @test a.kind == b.kind
        end
        ph = poincare_hopf(dw, c.distinct, lb, ub)
        @test ph.sum == 1 && ph.n_interior == 3
        @test ph.nout == ph.nrm > 0                # gradient points outward on every face
        hl = census_headline(c, ph)
        @test hl["n_interior_minima"] == 2 && hl["n_interior_saddles"] == 1
        @test hl["n_interior_maxima"] == 0 && hl["n_distinct"] == 3
        @test hl["poincare_hopf_sum"] == 1 && hl["outward_frac"] == 1.0
        @test hl["n_raw"] == 7 && hl["n_seeds"] == 6 && hl["n_converged"] == 6
    end

    @testset "failures are counted, not dropped" begin
        # Inf outside x1 < 0.5 and a seed in that region: it cannot converge
        f = x -> x[1] > 0.5 ? Inf : dw(x)
        c = newton_census(f, [[-0.9, 0.1], [0.9, 0.0]], lb, ub; threads = false)
        @test c.nfail == 1
        @test length(c.distinct) == 1
        @test c.distinct[1].p ≈ [-1.0, 0.0] atol = 1e-6
        @test c.srec[2] !== nothing && !c.srec[2].ok
        # the seed cap keeps the lowest-f seeds
        cc = newton_census(dw, raw, lb, ub; seedcap = 2, threads = false)
        @test length(cc.seeds) == 2
    end

    @testset "step-tolerance acceptance" begin
        # minimum at (√2, 0), where the gradient is not exactly representable as zero, so
        # gtol = 1e-30 is unreachable; the Newton step in box units is tiny at a
        # nondegenerate critical point, so the step test accepts what gtol rejects
        dw2(x) = (x[1]^2 - 2)^2 + x[2]^2
        c0 = newton_census(dw2, [[1.3, 0.1]], lb, ub; gtol = 1e-30, threads = false)
        @test c0.nfail == 1
        c1 = newton_census(dw2, [[1.3, 0.1]], lb, ub; gtol = 1e-30, steptol = 1e-9, threads = false)
        @test c1.nfail == 0 && c1.distinct[1].accepted_by == "step"
        @test isfinite(c1.distinct[1].newton_step) && c1.distinct[1].newton_step < 1e-9
    end

    @testset "write_census and read_raw_candidates round-trip" begin
        c = newton_census(dw, raw, lb, ub; threads = false)
        ph = poincare_hopf(dw, c.distinct, lb, ub)
        dir = mktempdir()
        rawcsv = joinpath(dir, "raw_cps.csv")
        open(rawcsv, "w") do io
            println(io, "p1,p2,objective,leaf_id,in_own_leaf")
            for (k, p) in enumerate(raw)
                println(io, p[1], ",", p[2], ",", dw(p), ",", k % 3, ",1")
            end
        end
        pts, lids = read_raw_candidates(rawcsv, 2)
        @test pts == raw && lids == [k % 3 for k in 1:7]
        out = joinpath(dir, "census.json")
        seeds = joinpath(dir, "seeds.csv")
        write_census(out, seeds, c, lids, lb, ub; entry_name = "dw", raw = rawcsv,
                     extra = Dict("ode_tol" => 1e-8))
        d = JSON.parsefile(out)
        @test d["entry"] == "dw" && length(d["points"]) == 3
        @test d["ode_tol"] == 1e-8 && d["hess_tol"] == 1e-8
        @test Set(p["kind"] for p in d["points"]) == Set(["minimum", "saddle"])
        lines = readlines(seeds)
        @test length(lines) == 1 + length(c.seeds)
        @test startswith(lines[1], "seed,s1,s2,leaf_ids,n_raw,converged,iters,grad_norm,distinct_id,moved,p1,p2")
    end

    @testset "NewtonIndexPreserving refinement method" begin
        m = NewtonIndexPreserving()
        @test method_name(m) == :newton_index_preserving
        @test needs_gradient(m) && needs_hessian(m)
        r = refine_point(m, dw, [0.03, -0.02]; bounds = [(-2.0, 2.0), (-2.0, 2.0)])
        @test r.converged && r.cp_type == :saddle && norm(r.point) < 1e-6
        @test r.gradient_method == :forwarddiff && length(r.eigenvalues) == 2
        r2 = refine_point(m, dw, [0.8, 0.2])            # no bounds: unconstrained
        @test r2.converged && r2.cp_type == :min
        @test r2.point ≈ [1.0, 0.0] atol = 1e-6
        @test r2.objective_value ≈ 0.0 atol = 1e-12
        r3 = refine_point(NewtonIndexPreserving(gtol = 1e-30, steptol = 1e-9), dw, [0.8, 0.2];
                          bounds = [(-2.0, 2.0), (-2.0, 2.0)])
        @test r3.converged
        @test refinement_method(:newton_index_preserving; gtol = 1e-6) isa NewtonIndexPreserving
    end
end
