# Changelog

## v0.1.4.9007

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