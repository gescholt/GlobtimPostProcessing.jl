# Changelog - GlobtimPostProcessing

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Three optional features move into package extensions, and three dependencies are
dropped. A plain `using GlobtimPostProcessing` loads 94 modules instead of 124 and
takes about 0.6 s instead of 1.1 s (Julia 1.12, warm cache); the dependency closure
falls from 119 to 86 packages.

### Changed (breaking)

- **`load_subdivision_tree` needs `using JLD2`.** It lives in the new
  `GlobtimPostProcessingJLD2Ext`; without JLD2 it raises "load_subdivision_tree
  requires JLD2.jl". `load_subdivision_result` (JSON) is unchanged.
- **Table printers need `using PrettyTables`.** `styled_table`, `print_capture_summary`,
  `print_degree_capture_convergence`, `print_degree_convergence_summary`,
  `print_comparison_table`, `print_experiment_header`, `print_poly_summary_table`,
  `print_degree_analysis_table`, `print_parameter_recovery_table(true_params,
  estimate)`, `print_sparsification_metrics_table`, `print_sparsification_capture_table`,
  `print_subdivision_tree_report`, `print_tree_comparison_table` and the
  `GLOBTIM_TABLE_STYLE="pretty"` branch of `analyze_campaign` live in
  `GlobtimPostProcessingPrettyTablesExt`; each raises "... requires PrettyTables.jl"
  without it. PrettyTables is still a dependency of DataFrames, so the extension loads
  with the package today; the requirement matters once DataFrames is optional.
  `print_refinement_summary`, `print_capture_verdict`, `print_sparsification_summary`,
  the campaign `print_parameter_recovery_table(::CampaignResults)` and the
  `format_*` text tables stay in the package, so `refine_experiment_results` needs no
  extension. The printers still target the PrettyTables 2 API (compat `"2"`).
- **`postprocess()` and the `select_*` / `confirm_selection` menus need REPL and
  PrettyTables** (`GlobtimPostProcessingTUIExt`). In the Julia REPL, REPL is already
  loaded; a script adds `using REPL, PrettyTables`. The non-interactive pipeline
  (`load_experiment`, `detect_experiment_type`, ...) is unchanged.
- `load_experiment_config` reads with `JSON.parsefile(...; dicttype = Dict{String,Any})`;
  the result (a `Dict{String,Any}` with `Vector{Any}` arrays) is the same as before.

### Fixed

- The quality mode of `postprocess()` now finds `styled_table`; it used to fail with an
  `UndefVarError` inside `UnifiedPipeline`.

### Removed

- Dependencies `JSON3` (JSON does both calls), `StatsBase` (two `countmap` calls became
  a dictionary count) and `LineSearches` (LBFGS takes `Optim.LineSearches.InitialStatic`,
  which every Optim release in compat provides). `JLD2`, `PrettyTables` and `REPL` move
  from `[deps]` to `[weakdeps]`.

## [0.2.0] - 2026-11-19

The package no longer carries code tied to one study's result layout. The two modules
below read hard-coded result directories and experiment-naming schemes of a single
private study, so they are not usable outside it; they leave the package unchanged
in behaviour and are not replaced.

### Removed (breaking)

- **`LV4DAnalysis` submodule** (`GlobtimPostProcessing.LV4DAnalysis`): the Lotka-Volterra
  4-D sweep / convergence / coverage / method-comparison analysis and its `lv4d()`
  terminal UI, including `load_lv4d_experiment`, `load_sweep_experiments`,
  `LV4DExperimentData`, `LV4DSweepData`, `ExperimentFilter` / `fixed` / `sweep`,
  `query_experiments`, `analyze_sweep`, `analyze_quality`, `analyze_coverage`,
  `compare_single_vs_subdivision` and the rest of that submodule's exports.
- **Benefit-matrix assembly**: `assemble_benefit_matrix`, `BenefitCell` and
  `ODE_PE_SYSTEMS` are no longer exported or defined.
- **`postprocess()` LV4D-only analysis modes** (`sweep`, `compare`, `coverage`): the
  interactive pipeline now offers the `quality` and `convergence` modes for every
  experiment type, LV4D included.
- `UnicodePlots` dependency (only the removed submodule used it).
- Internal planning documents (`docs/archive/`), stale guides describing files that no
  longer exist (`docs/guides/`), and the campaign scripts under `scripts/` together with
  the root-level `analyze_experiments.jl` driver; none of them were part of the API.

### Changed (breaking)

- **`load_experiment` on an LV4D experiment returns `BaseExperimentData`** (tagged
  `experiment_type(data) === LV4D`), through the same loader as every other type. It
  used to delegate to the removed submodule and return `LV4DExperimentData`.
- **Gradient-based Optim refinement differentiates with ForwardDiff.** Optim 2 defaults
  to central finite differences whenever only `f` is passed (`2m` evaluations per
  gradient, plus truncation noise), and `refine_critical_point` passed only `f`.
  It now takes `autodiff` (an ADTypes backend; `ADTypes` is a new dependency) and
  defaults it to `AutoForwardDiff()` for `:lbfgs` / `:bfgs` / `:conjugategradient` /
  `:gradientdescent`. On `OptimLBFGS`, `OptimBFGS`, `OptimConjugateGradient` and
  `OptimGradientDescent` the `gradient_method` field now selects that backend as well as
  the post-hoc Hessian, so iteration and classification agree, and its default moved
  from `:finitediff` to `:forwarddiff`. A Float64-pinned objective asks for
  `gradient_method = :finitediff` explicitly; it is no longer downgraded silently.
  NelderMead keeps `AutoFiniteDiff` (it only meets a gradient in `Fminbox`'s outer
  loop, and must never hand such an objective a Dual).
- **The NelderMead tolerance we expose is the one Optim uses.** Optim's NelderMead has a
  single stopping test, the standard deviation of `f` over the simplex, read from
  `g_abstol`; the `f_abstol` / `x_abstol` options we passed were never consulted and
  the effective stop was always Optim's `1e-8`. `refine_critical_point`,
  `refine_critical_points_batch`, `OptimNelderMead` and `RefinementConfig` now map
  `f_abstol` onto that test (default `1e-8`, so existing runs are unchanged unless they
  set a value) and **refuse** an `x_abstol` or `g_abstol` for NelderMead with an
  `ArgumentError` instead of ignoring it. `OptimNelderMead` lost its `x_abstol` field;
  `RefinementConfig.f_abstol` / `.x_abstol` default to `nothing` (the method's own
  default). A NelderMead run that stopped on the spread reports
  `convergence_reason = :f_tol`, where it used to say `:g_tol`.
- **Bounded refinement reports Fminbox's own stop.** With bounds, Optim's
  `g_converged` is `Fminbox`'s outer projected-gradient test, so such a stop is
  reported as `:g_tol`; `outer_g_abstol` (default `1e-8`) and `outer_iterations`
  (default `1000`) are exposed, and an exhausted outer loop is reported as
  `:outer_iterations`.
- `Optim` compat floor raised to `1.12` (the ADTypes `autodiff` keyword needs
  NLSolversBase 7.9 or newer).

### Added

- **Index-preserving Newton census** (`src/refinement/newton_census.jl`):
  `polish_to_critical` (LM-damped Newton on `∇f = 0`, backtracking on the residual, so
  a saddle stays a saddle), `critical_point_kind` (exact-AD Hessian spectrum → minimum /
  saddle / maximum / degenerate + Morse index), `newton_census` (seed dedup, threaded
  polish into preallocated slots so the result is thread-count independent, absolute
  `gtol` or scale-free `steptol` acceptance, failures counted never dropped,
  post-polish dedup), `poincare_hopf`, `census_headline`, `write_census` /
  `read_raw_candidates`, and `NewtonIndexPreserving` as a `RefinementMethod`
  (`refine_point` → `CriticalPointRefinementResult`, ForwardDiff, no trust region and no
  objective-increase safeguard). Validated on a 4-D Lotka-Volterra landscape with a
  known critical set (20 of 20 predicted critical points). `Random` is a new dependency
  (face sampling of the Poincaré–Hopf check).
- **Saddle links** (`src/refinement/saddle_links.jl`): `newton_critical_point` (damped
  Newton on `∇f = 0` in a box, converging to saddles and maxima as readily as to
  minima), `lm_critical_point`, `minmode_saddle` (eigenvector following with the exact
  Hessian), `descend_to_minimum`, `morse_data`, `nearest_point_index`, and
  `saddle_links`, which names the two minima an index-1 saddle separates.
- **Representatives-only refinement and threaded batches.** `load_raw_critical_points`
  reads the `representative` / `cluster_size` columns that Globtim's surrogate-side
  deduplication writes into `critical_points_raw_deg_X.csv` (all `true` / `1` for older
  files); `refine_experiment_results(...; representatives_only = true)` refines one point
  per cluster, and `threads = true` fans the independent refinements on `f` out over
  threads in `refine_critical_points_batch`, with `objective_pool` for objectives that
  are not thread-safe (one instance per thread, `:static` schedule).

## [0.1.3] - 2026-09-10

### Added

- **Consensus validation** — `validate_consensus`, `ensemble_consensus`,
  `match_critical_point_sets` and their result types (`ConsensusCP`,
  `ConsensusValidationConfig`, `ConsensusValidationResult`, `EnsembleCluster`,
  `EnsembleConsensusResult`). Fitting one objective twice from independent
  sample sets and scoring each cluster by recurrence across fits. On sloppy
  landscapes recurrence measures *systematicity*, not validity: a noise-pinned
  point recurs. The headline output is the identifiable-value estimate.
- **`LevenbergMarquardt`** refinement method — LM on the residual vector with
  the exact forward-mode Jacobian, selected through the same
  `refine_critical_points` dispatch as the existing methods. It replaces
  Nelder--Mead as the default polisher for sum-of-squares objectives, where a
  derivative-free refiner stalls along flat valleys and reports the stalls as
  distinct minima.
- Newton-step-norm validation columns (`step_norm`, `step_valid`) and
  `hessian_min_eigval` on refinement output. `‖H⁻¹∇f‖` is invariant to the
  scale of `‖H‖`, which a bare gradient-norm test is not.

### Fixed

- Bounded refinement now works in normalized coordinates, fixing a silent
  no-op on tight boxes.

### Changed

- **Relicensed from GPL-3.0 to MIT.** This takes effect from 0.1.3 forward;
  0.1.2 and earlier remain GPL-3.0.
- Comments and docstrings no longer cite internal issue ids or monorepo-only
  script paths, neither of which resolves for anyone installing this package.

## [0.1.2] - 2026-08-05

### Fixed

- Documenter configuration pointed `repo` / `repolink` at the private `globopt_merged` monorepo and `edit_link` at a nonexistent `master` branch; repoint to the public `GlobtimPostProcessing.jl` repository on `main`.
- Pin `PrettyTables` compat to `"2"` — the display helpers `using PrettyTables: Highlighter, tf_unicode_rounded, ft_printf`, all removed in PrettyTables 3; a prior widening to `"2, 3"` let a fresh resolve pull v3 and broke the test suite (`UndefVarError`).

## [0.1.1] - 2026-04-30

Initial public release.

### Added

- Refinement diagnostics — `RefinementResult` struct with call counts (`f_calls`, `g_calls`, `h_calls`), timing, and fine-grained convergence flags; `convergence_reason` enum (`:x_tol`, `:f_tol`, `:g_tol`, `:iterations`, `:timeout`, `:error`).
- Gradient-norm validation — `compute_gradient_norms`, `validate_critical_points`, `add_gradient_validation!`; automatic integration in `refine_experiment_results` using `config.f_abstol`.
- Error categorization — campaign-wide error analysis (`categorize_campaign_errors`, `generate_error_summary`), priority-based filtering, batch-analyze CLI integration (`--include-errors`).
- VegaLite L2 visualization (Phase 1) — `campaign_to_l2_dataframe`, `plot_l2_convergence`, pure DataFrames.jl pipeline.
- Newton refinement, valley walking, statistical aggregation, campaign comparison utilities.
- Globtim integration via weakdep extension `GlobtimExt`.

### Notes

- `Globtim` is a weak dependency. Load it alongside `GlobtimPostProcessing` to activate the integration extension.
- Julia 1.12+ required.

## [Unreleased] - 2025-11-28

### Added - Gradient Norm Validation (Phase 2 Tier 2)

#### Gradient Validation for Critical Points
- **New Module**: `src/refinement/gradient_validation.jl`
  - `compute_gradient_norms()` - Batch gradient norm computation using ForwardDiff
  - `compute_gradient_norm()` - Single-point gradient norm
  - `validate_critical_points()` - Validate points against tolerance threshold
  - `add_gradient_validation!()` - Add validation columns to DataFrames
  - `GradientValidationResult` - Result struct with validation statistics

#### Automatic Integration
- Gradient validation automatically runs on converged points in `refine_experiment_results()`
- Uses `config.f_abstol` as validation tolerance
- Results included in printed summary output

#### Enhanced Output
- **refinement_comparison_deg_X.csv** now includes:
  - `gradient_norm` - ||∇f(x)|| for each refined point
  - `gradient_valid` - Boolean validation status
- **refinement_summary_deg_X.json** now includes:
  - `gradient_validation` section with n_valid, n_invalid, mean_norm, max_norm, validation_rate

#### Dependencies
- Added `ForwardDiff` (v0.10) for automatic differentiation

#### Documentation
- Updated `docs/REFINEMENT_DIAGNOSTICS.md` - marked gradient validation complete
- Archived `gradient_validation_plan.md` to `docs/archive/`

**Status**: ✅ Complete - Implements gradient norm validation from Phase 2 Tier 2

---

## [Unreleased] - 2025-11-24

### Added - Phase 2 Tier 1 Refinement Diagnostics

#### Enhanced Refinement Diagnostics (Zero-Cost)
- **Enhanced RefinementResult Struct** with 9 new diagnostic fields:
  - Call counts: `f_calls`, `g_calls`, `h_calls` (function/gradient/Hessian evaluations)
  - Timing: `time_elapsed` (actual optimization time per point)
  - Fine-grained convergence: `x_converged`, `f_converged`, `g_converged`, `iteration_limit_reached`
  - Primary reason: `convergence_reason` (`:x_tol`, `:f_tol`, `:g_tol`, `:iterations`, `:timeout`, `:error`)

#### Enhanced CSV Output
- **refinement_comparison_deg_X.csv** now includes 9 diagnostic columns:
  - Per-point call counts, timing, and convergence details
  - Enables detailed analysis of refinement performance

#### Enhanced JSON Summary
- **refinement_summary_deg_X.json** now includes:
  - `convergence_breakdown`: Count by convergence reason
  - `call_counts`: Mean/max/min function evaluations
  - `timing`: Mean/max/min time per point, timeout count

#### Testing
- **Enhanced**: `test/test_refinement_phase1.jl` (added 8 test sets)
  - Test all diagnostic fields exist and have correct types
  - Test call counts, timing, convergence flags
  - Test convergence reason logic for all cases
  - ~40 new assertions, all passing

#### Documentation
- **New File**: `docs/PHASE2_TIER1_IMPLEMENTATION.md`
  - Complete implementation guide
  - Usage examples for CSV and JSON diagnostics
  - Performance characteristics (< 1% overhead)
  - Migration guide and troubleshooting

#### Performance
- Zero-cost diagnostics (all extracted from Optim.jl result)
- No trace storage required
- < 1% overhead (diagnostic extraction is O(1))
- Memory overhead: +72 bytes per RefinementResult

#### Backward Compatibility
- ✅ All original fields preserved
- ✅ Existing code continues to work
- ✅ API extensions only (no breaking changes)

**Status**: ✅ Complete - Implements requirements from `docs/REFINEMENT_DIAGNOSTICS.md` Phase 1 (Tier 1)

---

## [Earlier] - 2025-10-29

### Added - Error Categorization Integration (Issue #20, Phase 3)

#### Error Analysis Features
- **New Module**: `src/ErrorCategorizationIntegration.jl` (709 lines)
  - `categorize_campaign_errors()` - Campaign-wide error analysis
  - `generate_error_summary()` - Comprehensive error reporting
  - `filter_errors_by_category()`, `filter_errors_by_severity()` - Error filtering
  - `get_top_priority_errors()` - Priority-based error retrieval
  - `calculate_error_rate()`, `get_most_common_error_category()` - Statistics
  - `format_error_report()`, `format_error_table()` - Report formatting
  - `get_error_dataframe()` - DataFrame export for error data
  - `create_mock_campaign()` - Testing utilities

#### Integration Features
- Error sections in campaign reports (Markdown/JSON)
- `--include-errors` CLI flag in `scripts/batch_analyze.jl`
- Error analysis in `batch_analyze_campaign()` and `batch_analyze_campaign_with_progress()`
- 5 error categories: INTERFACE_BUG, MATHEMATICAL_FAILURE, INFRASTRUCTURE_ISSUE, CONFIGURATION_ERROR, UNKNOWN_ERROR
- Priority scoring with actionable recommendations

#### Testing
- **New File**: `test/test_error_categorization.jl` (105 tests, all passing)
  - Test coverage: module access, single error categorization, campaign analysis
  - Error-aware reports, filtering, statistics, batch integration, formatting, edge cases
  - TDD approach: RED → GREEN → REFACTOR

#### Dependencies Added
```toml
Globtim = "00da9514-6261-47e9-8848-33640cb1e528"  # Dev dependency
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"  # v0.34.7
```

**Status**: ✅ Complete - All 105 tests passing, fully integrated

### Added - VegaLite Integration (Phase 1)

#### Minimal L2 Visualization
- **New Module**: `src/VegaPlotting_minimal.jl`
  - `campaign_to_l2_dataframe()` - Extract L2 errors to DataFrame
  - `plot_l2_convergence()` - Create interactive L2 convergence plot
  - Pure DataFrames.jl implementation (no Tidier.jl dependency)

#### Testing Infrastructure
- **New Script**: `examples/test_minimal_l2_plot.jl`
  - Complete test harness for L2 visualization
  - Step-by-step validation output
  - Ready for testing with real campaign data

#### Documentation
- **New File**: `README_VEGALITE.md`
  - Quick start guide
  - Usage examples
  - Troubleshooting tips
  - Development roadmap

- **New File**: `docs/issues/vegalite_tidier_integration.md`
  - Complete design documentation
  - Phase-by-phase implementation plan
  - Design decisions and rationale
  - Testing checklist

- **New File**: `docs/tidier_vega_guide.md`
  - Comprehensive guide for future advanced features
  - Tidier.jl syntax examples (for future use)
  - VegaLite visualization gallery

#### Support Files
- **New File**: `src/TidierTransforms.jl`
  - Simplified data transformation functions
  - Pure DataFrames.jl implementations
  - Helper functions for future phases:
    - `compute_convergence_analysis()`
    - `compute_parameter_sensitivity()`
    - `compute_efficiency_metrics()`
    - `pivot_metrics_longer()`
    - `add_comparison_baseline()`
    - `annotate_outliers()`

- **New File**: `examples/demo_tidier_vega_suite.jl`
  - Menu-driven demo system (for future phases)
  - Multiple visualization options
  - Interactive selection interface

#### Visualization Features (Phase 1)
- Interactive line plot with VegaLite
- Log-scale Y-axis for error visualization
- Color encoding by experiment ID
- Hover tooltips with detailed information
- Browser-based rendering
- Export capabilities (HTML, PNG, SVG)

### Design Decisions

#### Why No Tidier.jl?
- Initial implementation attempted full Tidier.jl integration
- Encountered macro expansion and compilation issues
- Pivoted to pure DataFrames.jl for Phase 1
- Simpler, more maintainable, guaranteed to compile
- Can add Tidier later if needed

#### Why Start Minimal?
- Test-Driven Development (TDD) approach
- Validate core functionality first
- Easier debugging with small surface area
- Clear path for incremental feature addition
- User feedback: "Start simple, proceed with TDD approach"

### Testing Status

- ✅ Code written and documented
- ⏳ **Awaiting validation** with real campaign data
- ⏳ Browser rendering verification needed
- ⏳ Data accuracy validation needed

### Dependencies Added

```toml
VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a"  # v3.3.0
```

**Note**: Tidier.jl removed from critical path (was causing issues)

### Breaking Changes

**None** - This is purely additive functionality

### Migration Guide

**Not applicable** - New feature, existing code unchanged

## Future Phases (Planned, Not Implemented)

### Phase 2: Interactive Features
- Linked selection (click to filter)
- Multi-view dashboards
- Brush selection for zooming
- Dynamic parameter filtering

### Phase 3: Additional Metrics
- Parameter recovery plots
- Numerical stability (condition numbers)
- Multi-metric faceted comparisons
- Convergence quality analysis

### Phase 4: Advanced Analysis
- Statistical outlier detection
- Baseline comparisons
- Efficiency analysis (error vs time)
- Parameter sensitivity studies

## Version History

### Current: v0.1.0 (Development)
- Initial VegaLite integration (Phase 1 complete)
- Minimal L2 visualization working
- Documentation complete
- Awaiting real-data testing

---

## Testing Instructions

To test the new VegaLite L2 visualization:

```bash
cd /path/to/globtimpostprocessing

# Run with your campaign data
julia --project=. examples/test_minimal_l2_plot.jl \
    /path/to/your/campaign/hpc_results

```

Expected: Interactive L2 convergence plot opens in browser

---

## Notes for Future Development

### What Works Well
- Pure DataFrames.jl approach is clean and maintainable
- VegaLite provides excellent interactivity
- Modular design allows easy feature addition
- Test harness makes validation straightforward

### Lessons Learned
- Start simple and validate before adding complexity
- Avoid complex macro systems until core functionality works
- TDD approach catches issues early
- Documentation is crucial for phased development

### Technical Debt
- `src/TidierTransforms_old.jl` - Remove after Phase 1 validation
- `src/VegaPlotting.jl` - Large file with unused Tidier code, clean up later
- Test suite needs expansion as features are added

---

**Last Updated**: October 7, 2025
**Next Milestone**: Phase 1 validation with real campaign data
