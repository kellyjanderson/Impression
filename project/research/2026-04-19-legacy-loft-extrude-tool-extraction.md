# 2026-04-19 Legacy Loft And Extrude Tool Extraction

## Purpose

Identify useful functions embedded in legacy mesh-era loft and extrude code that
should be extracted into durable tooling rather than discarded with the old
modeling paths.

This is not a list of everything in `loft.py` or `extrude.py`.

It is a list of functions that look valuable as:

- analysis tooling
- repair/support tooling
- surfaced-adjacent utilities
- correspondence diagnostics
- cap/profile preparation utilities

## Source Files Reviewed

- `src/impression/modeling/extrude.py`
- `src/impression/modeling/loft.py`

## Extraction Priorities

### Tier 1: High-Value Tooling To Preserve

These are the most valuable extractions because they support analysis,
verification, or surfaced-adjacent workflows directly.

#### Path / Frame / Station Tooling

These are useful beyond legacy loft execution and should likely become explicit
geometry or analysis helpers.

- `loft.py:_resolve_positions`
  - normalizes caller path input into a 3D sample set
  - useful for any stationed tool pipeline

- `loft.py:_resample_path`
  - deterministic path resampling
  - useful for sectioning, reconstruction, and progression-based analysis

- `loft.py:_normalized_path_parameters`
  - normalized progression values from 3D path samples
  - useful for any “sample a model back along progression” tooling

- `loft.py:_compute_frames`
  - parallel-transport frame generation along a path
  - very valuable as general tooling

- `loft.py:_normalized_vector`
  - likely too small to extract on its own, but useful if the frame helpers are
    extracted together

- `loft.py:_orthonormalize_frame`
  - useful to normalize reconstructed frames or validate station input

- `loft.py:_build_stations`
  - converts sampled positions into station frames
  - useful for loft prep and analysis tools

- `loft.py:_build_profile_section_stations`
  - useful if we want an explicit “build station records from sections+path”
    helper outside the legacy executor

#### Correspondence / Twist / Ambiguity Tooling

These are the most important candidates for future loft diagnostics and
reference tests.

- `loft.py:_align_loops_for_loft`
  - aligns loop ordering across profiles
  - directly relevant to correspondence testing

- `loft.py:_resolve_transition_loop_start`
  - start-vertex alignment for transition loops
  - high-value for twist detection and recovery

- `loft.py:_loop_sort_key`
  - deterministic loop ordering helper
  - useful for stable comparison and reference generation

- `loft.py:_enumerate_region_ambiguity_candidates`
  - explicit candidate generation for region ambiguity
  - useful for diagnostics and inspection tooling

- `loft.py:_enumerate_hole_ambiguity_candidates`
  - hole-level ambiguity candidate generation
  - useful for correspondence-debug tooling

- `loft.py:_minimum_cost_subset_assignment`
  - core candidate assignment logic
  - useful as standalone correspondence solver tooling

- `loft.py:_minimum_cost_region_assignment`
  - useful as deterministic region correspondence tool

- `loft.py:_minimum_cost_hole_assignment`
  - useful as deterministic hole correspondence tool

- `loft.py:_score_subset_assignment`
  - valuable for “why did this correspondence win?” reporting

- `loft.py:_local_assignment_fairness`
  - useful for fairness diagnostics in correspondence tools

- `loft.py:_assignment_vectors_world`
  - useful for geometric analysis of chosen correspondence

- `loft.py:_transition_curr_vectors`
  - useful for ambiguity and fairness diagnostics

- `loft.py:_count_segment_crossings_2d`
  - useful for branch-crossing analysis

- `loft.py:_segments_intersect_2d`
  - low-level helper, worth extracting only if bundled with crossing analysis

- `loft.py:_normalized_loop_area_delta`
  - useful comparison signal for similarity checks

- `loft.py:_region_pair_centerline_world`
  - useful for analysis and debugging of branch correspondence

- `loft.py:_is_stable_loop_transition`
  - strong candidate for standalone transition classifier

- `loft.py:_is_split_merge_ambiguous`
  - strong candidate for standalone ambiguity classifier

- `loft.py:_classify_region_transition_ambiguity`
  - useful surfaced-adjacent diagnostic tool

- `loft.py:_has_containment_ambiguity`
  - useful diagnostic subtool

- `loft.py:_has_assignment_symmetry_or_permutation`
  - useful diagnostic subtool

- `loft.py:_is_symmetry_layout`
  - useful diagnostic subtool

#### Section / Topology Reconstruction Helpers

These are useful for section-based analysis, including the future plane-section
comparison workflow.

- `loft.py:_section_to_region_loops`
  - strong candidate for extracted section-normalization tool

- `loft.py:_section_from_region_loops`
  - useful inverse helper for reconstructed topology

- `loft.py:_canonicalize_section_for_loft`
  - strong candidate for explicit section canonicalization utility

- `loft.py:_loops_resampled_anchored`
  - useful for stable comparison fixtures and reconstructed section matching

#### Endcap / Cap-Shaping Tooling

These look useful as reusable standalone cap-profile generation tools even if
legacy mesh endcap execution is deleted.

- `loft.py:_resolve_endcap_amounts`
  - useful endcap parameter normalization

- `loft.py:_apply_caps`
  - likely too tied to legacy flow to keep as-is, but worth mining

- `loft.py:_cap_profile_series`
  - useful cap-profile schedule generator

- `loft.py:_cap_ease`
  - useful cap-shape profile utility

- `loft.py:_scale_loop`
  - useful loop-scaling helper

- `loft.py:_loop_half_dims`
  - useful for anisotropic cap logic and comparison utilities

- `loft.py:_build_endcap_sections`
  - high-value candidate for surfaced cap tooling

- `loft.py:_endcap_schedule`
  - useful cap scheduling helper

### Tier 2: Worth Preserving If Bundled

These are useful, but probably not worth extracting as one-off public tools.
They make sense if bundled into a broader extracted utility module.

#### Synthetic Split/Merge Staging Helpers

- `loft.py:_expand_split_merge_stations`
- `loft.py:_needs_split_merge_staging`
- `loft.py:_interpolate_station`
- `loft.py:_shrunken_loop`
- `loft.py:_shrunken_region`
- `loft.py:_synthetic_seed_scale`

These are valuable if we keep explicit synthetic-staging tooling for:

- ambiguity exploration
- split/merge debugging
- staged preview generation

#### Ambiguity Parsing / Reporting Helpers

- `loft.py:_ambiguity_failure_stage`
- `loft.py:_ambiguity_failure_candidate_count`
- `loft.py:_candidate_scalar_cost`
- `loft.py:_ambiguity_candidate_id`
- `loft.py:_hole_ambiguity_candidate_id`
- `loft.py:_predicted_actions_for_assignment`

These are useful if we keep a standalone loft diagnostics/reporting layer.

#### World-Space Loop / Patch Helpers

- `loft.py:_station_loop_world`
- `loft.py:_loft_planar_patch_from_station_loops`

These are useful if we build explicit surfaced or analysis-facing “station loop
to world geometry” helpers.

### Tier 3: Mostly Legacy-Flow Internal

These do not look like good standalone tooling targets by themselves.

They should mostly stay internal or disappear with legacy modeling code.

- `extrude.py:_linear_extrude_loops`
  - too tightly bound to mesh face construction

- `extrude.py:_cap_faces`
  - useful internally, but too mesh-triangle specific unless adopted by a mesh
    repair/orientation toolkit

- `extrude.py:_apply_quality_samples`
  - better replaced by a more explicit quality policy helper

- `loft.py:_validate_*` family
  - useful internally, but not the first extraction target for tooling

- `loft.py:_validate_loft_plan`
  - important, but belongs more to plan-contract enforcement than standalone
    tooling

- `loft.py:_validate_region_pair_closure_ownership`
  - same

## Extrude-Specific Standalone Tool Candidates

Legacy extrude has fewer extraction-worthy helpers, but a few are still worth
keeping.

- `extrude.py:_rotate_around_axis`
  - useful geometric utility for mesh analysis tools, sectioning helpers, or
    rotate-extrude-related tooling

- `extrude.py:_rotate_vector`
  - same, especially if grouped with rotation/frame helpers

- `extrude.py:_normalize`
  - too small on its own, but worth folding into a geometry utility module if
    these helpers are extracted together

## Suggested Extraction Modules

If we do this work, these functions should not all land in one giant “utils”
file.

Recommended groupings:

### `impression.analysis.loft_correspondence`

Good home for:

- `_align_loops_for_loft`
- `_resolve_transition_loop_start`
- `_minimum_cost_subset_assignment`
- `_minimum_cost_region_assignment`
- `_minimum_cost_hole_assignment`
- `_enumerate_region_ambiguity_candidates`
- `_enumerate_hole_ambiguity_candidates`
- `_score_subset_assignment`
- `_local_assignment_fairness`
- `_count_segment_crossings_2d`
- `_normalized_loop_area_delta`
- `_is_stable_loop_transition`
- `_is_split_merge_ambiguous`
- `_classify_region_transition_ambiguity`

### `impression.analysis.sectioning` or `impression.analysis.stations`

Good home for:

- `_resolve_positions`
- `_resample_path`
- `_normalized_path_parameters`
- `_compute_frames`
- `_orthonormalize_frame`
- `_build_stations`
- `_build_profile_section_stations`
- `_section_to_region_loops`
- `_section_from_region_loops`
- `_canonicalize_section_for_loft`
- `_loops_resampled_anchored`

### `impression.modeling.cap_tools` or `impression.surface.cap_profiles`

Good home for:

- `_resolve_endcap_amounts`
- `_cap_profile_series`
- `_cap_ease`
- `_scale_loop`
- `_loop_half_dims`
- `_build_endcap_sections`
- `_endcap_schedule`

### `impression.analysis.mesh_tools`

Possible home for mesh-era retained utilities such as:

- `extrude.py:_rotate_around_axis`
- `extrude.py:_rotate_vector`
- mesh plane sectioning once added
- mesh standalone hull or CSG if retained as tools

## Best Immediate Extraction Candidates

If we want the shortest high-value list, these are the ones I would extract
first:

1. `_compute_frames`
2. `_resample_path`
3. `_section_to_region_loops`
4. `_canonicalize_section_for_loft`
5. `_align_loops_for_loft`
6. `_resolve_transition_loop_start`
7. `_minimum_cost_subset_assignment`
8. `_enumerate_region_ambiguity_candidates`
9. `_enumerate_hole_ambiguity_candidates`
10. `_build_endcap_sections`

That set would give us the best return for:

- loft correspondence analysis
- twist diagnostics
- reconstruction-style section comparison
- surfaced cap tooling

## Recommendation

The first concrete tooling extraction sequence should probably be:

1. section/path/frame utilities
2. correspondence and ambiguity diagnostics
3. plane-section reconstruction comparison tools
4. cap-profile helpers

That order best supports the loft verification work already identified as the
next big testing opportunity.
