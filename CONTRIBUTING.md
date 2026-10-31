# Contributing to tempdisagg

Thanks for considering a contribution to `tempdisagg`. The project's priority is **numerical correctness, reproducibility, and a clear public API**. A small, well-tested correction is more valuable than a large change that is difficult to validate.

## Before opening a pull request

- Search [existing issues](https://github.com/jaimevera1107/tempdisagg/issues) and pull requests for related work.
- For a bug, open a **Bug report** with a minimal reproducible example, ideally using synthetic data.
- For a new method or a substantial API change, open a **Feature request** first. Explain the statistical assumptions, intended use, relevant references, and how correctness will be assessed.
- Do not include confidential, copyrighted, or personal datasets in issues, tests, or pull requests unless you have permission to publish them.

## Development setup

Python **3.10 or newer** is required. The standard development installation is:

```bash
git clone https://github.com/jaimevera1107/tempdisagg.git
cd tempdisagg
python -m venv .venv
```

Use the virtual environment's Python to install the development dependencies and run the checks. For example, on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check tempdisagg tests
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or macOS:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m ruff check tempdisagg tests
.venv/bin/python -m pytest -q
```

The CI matrix checks Python **3.10–3.13** and separately validates results against the R `tempdisagg` package. Passing tests under another Python version locally does not replace these CI checks.

## What makes a statistical change acceptable?

When touching an estimator, aggregation rule, constraint, numerical optimization, or preprocessing step, include a targeted regression test. Depending on the change, demonstrate that:

1. **Aggregation/benchmark constraints hold.** Disaggregated values reproduce their low-frequency targets using the selected conversion (`sum`, `average`, `first`, or `last`), within an explicitly justified numerical tolerance.
2. **Other exact constraints hold.** Known high-frequency values, weighted constraints, or cross-sectional reconciliation conditions remain satisfied where applicable.
3. **The stated statistical objective is actually optimized.** For example, a Denton or Denton–Cholette change should verify the constrained criterion, not just the benchmark total.
4. **Results are stable and interpretable.** Cover edge cases such as missing periods, ill-conditioned indicators, boundary `rho`, unit changes, and infeasible constraints when relevant. Warnings and diagnostics must not be silently suppressed.
5. **Existing behavior is preserved unless the change is deliberate.** Document changes to public functions, numerical defaults, result fields, or compatibility behavior.

For an observed numerical discrepancy, provide a small dataset, the method and options, actual versus expected results, and (when relevant) a published formula or reference implementation. Compare against R only for settings that are meaningfully comparable; explain intentional differences.

Useful references in the repository:

- [Testing and numerical contracts](docs/testing.md)
- [Statistical methodology](docs/methodology.md)
- [Configuration and parameters](docs/configuration.md)
- [Architecture](docs/architecture.md)

## Tests and R reference parity

Run the full Python suite and Ruff before requesting review:

```bash
python -m ruff check tempdisagg tests
python -m pytest -q
```

The R parity CI workflow installs the R package, regenerates reference results, and checks them with `tests/test_r_parity.py`. If your change affects an R-comparable method, add or update an appropriate parity case. If R is available locally, you can reproduce the reference workflow:

```bash
Rscript tests/reference/generate_r_reference.R
python -m pytest tests/test_r_parity.py -v
```

Review any generated reference-file changes before committing them. Do not update reference values solely to make failing tests green; explain the mathematical or implementation reason for the change.

## Code and documentation

- Follow the existing module boundaries and public API patterns.
- Use type annotations and NumPy-style docstrings consistent with the project.
- Follow the Ruff configuration in `pyproject.toml` (including the project's 99-character line length).
- Add tests for fixes and new behavior, and update the relevant README, docs, or migration notes when the public behavior changes.
- Keep pull requests focused; avoid unrelated reformatting or dependency churn.

## Pull request workflow

1. Create a branch from the latest `main` (for example, `fix/denton-criterion` or `feat/new-method`).
2. Make the change, add tests, and run the local checks.
3. Open a pull request against `main`, describing the reason, statistical impact, and evidence.
4. Address CI failures and review comments before merging.

Do not bump package versions in ordinary feature or fix pull requests unless coordinated as a release change. **GitHub `main` is the source of truth**; published packages are built from a versioned GitHub Release through trusted publishing. See [the release procedure](docs/RELEASING.md). Do not upload packages manually to PyPI.

## Security issues

Please do **not** disclose suspected security vulnerabilities in public issues. Follow [SECURITY.md](SECURITY.md) instead.
