## What changed?

<!-- Briefly describe the change and link the related issue, if any. -->

Closes #

## Why is it needed?

<!-- Explain the defect, use case, or design decision. -->

## Statistical and API impact

<!-- For numerical changes, identify the methods, settings, objective, and constraints involved. -->

- Affected methods or modules:
- Changes to numerical outputs, defaults, public API, or dependencies:
- Compatibility considerations:

## Validation

- [ ] Ruff passed: `python -m ruff check tempdisagg tests`
- [ ] Full Python suite passed: `python -m pytest -q`
- [ ] Regression tests were added or updated for changed behavior (or not applicable, explained below)
- [ ] Benchmark and other exact constraints were checked where applicable
- [ ] Numerical objective, edge cases, and diagnostics were checked where applicable
- [ ] R parity was considered for comparable methods; CI result reviewed (or not applicable)
- [ ] Documentation / changelog / migration notes were updated if public behavior changed (or not applicable)
- [ ] No secrets or confidential datasets were included

### Test evidence or justification for unchecked items

<!-- Include minimal reproduction, output comparison, tolerance, benchmark residual, objective value, or reason an item does not apply. -->

## Release considerations

<!-- Ordinary PRs should not change the package version. Releases go through main, tagged GitHub Releases, and PyPI Trusted Publishing. See docs/RELEASING.md. -->

- [ ] This is not a release/version-bump PR, or release metadata is intentionally synchronized.
