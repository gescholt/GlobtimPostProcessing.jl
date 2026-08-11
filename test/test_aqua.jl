using Aqua
using GlobtimPostProcessing

@testset "Aqua.jl Quality Assurance" begin
    # Full Aqua suite, no exclusions.
    #
    # Two previous exclusions were dropped as stale (verified on Julia 1.12.6 /
    # Aqua 0.8.14):
    #   - `stale_deps = (; ignore = [:Tidier])` — Tidier is not in [deps] at all,
    #     so the ignore entry did nothing. stale_deps passes clean without it.
    #   - `persistent_tasks = false` ("Julia version compatibility") — passes on
    #     1.12 in ~12s.
    #
    # undocumented_names is off by Aqua's own default and stays off for now:
    # public names still lack docstrings. Tracked separately.
    Aqua.test_all(GlobtimPostProcessing)
end
