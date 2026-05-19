# Changelog

## v0.4.4 (19/5/2026)

This release incorporates feedback from the PhD dissertation reviewers: dr hab. Andrzej Dudek, dr hab. Joanna Landmesser-Rusek, and dr hab. Paweł Andrzej Strzelecki.

### Added

- Added Brier score and mean P(true class) diagnostics to `cat2cat_ml_run()` results and repr output.
- Added support for categorical/object/string ML features by one-hot encoding levels observed in `ml.data` and the target period.
- Added configurable ML failure handling to `cat2cat_ml` with `on_fail={"freq", "naive", "na", "error"}` and `fail_warn`.
- Added the `occup_panel` packaged dataset and `load_occup_panel()`.
- Added `summary_c2c()` for statsmodels-like result objects, including corrected standard errors, corrected statistics, corrected p-values, and reference distribution labels.
- Added docs pages for Get Started, Choosing Weights And Validating ML, and Advanced Workflows.

### Changed

- Fixed `cat2cat_ml_run()` so `test_prop` controls the scikit-learn train/test split as documented.
- Kept the Python ML API estimator-based: users pass compatible scikit-learn classifiers with `predict_proba()`.
- Updated README and example docs for scikit-learn estimators, categorical features, ML fallback policy, and the new probability diagnostics.
- Improved `get_freqs()` counting to preserve missing-value categories with pandas `value_counts(dropna=False)`.
- Replaced assertion-based runtime validation in public/core paths with explicit `TypeError`/`ValueError` exceptions for library-safe API behavior.

### Optional Dependencies

- Added the `summary` extra for statsmodels-based regression workflows.

## v0.1.7 (26/12/2025)

- Support higher python versions; 3.12 and 3.13.

## v0.1.6 (11/02/2024)

- New `cat2cat_ml_run` function to check the ml models performance before `cat2cat` with ml option is run. Now, the ml models are more transparent.
- Improved the lack of support for NaN and None in the `get_mappings`.
- Fixed a bug that `cat2cat_ml.features` can be only a `list` not a `Sequence`.
- Fixed assertion message and docs for the `freqs` argument in the `cat2cat_mappings`.
- Fixed some typing, and bring the clear `mypy`.
- Replaced poetry with setuptools.

## v0.1.4 (12/09/2022)

### Fix

- Fixed README example, pandas.concat was not imported.
- Fixed some typing, to be more precise.

### Tests

- Added more tests to increase the coverage to 100%.

### Miscellaneous

- Added more comments in code examples.
- Added support in a few places for Sequence, to not limit users only to List.

## v0.1.3 (01/09/2022)

### Fix

- Fixed typing problems, mypy connected.
- Fixed html documentation formatting.

### Tests

- Added more tests for dataclasses.

### Miscellaneous

- Added more validation to dataclasses.
- Improved documentation.
- Removed `get_mappings` method from the `cat2cat_mappings` dataclass.
- Removed not needed import calls.
- Created the project description and sidebar on pypi.
- Added pypi badge.

## v0.1.1 (30/08/2022)

- First test release of `cat2cat`!
