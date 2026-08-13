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
    # persistent_tasks tmax: Aqua defaults to 10s for the spawned subprocess to
    # load the package and exit. A cold load under CI contention can exceed that
    # and be misread as a lingering task — observed here as a real flake on
    # 2026-08-13 (failed with a formatter running alongside, passed on an idle
    # machine, same commit). A genuine persistent task never exits, so a longer
    # budget costs nothing on the happy path and removes the false positive.
    Aqua.test_all(GlobtimPostProcessing; persistent_tasks = (; tmax = 90))
end
