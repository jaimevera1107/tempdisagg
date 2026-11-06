# Publishing tempdisagg from GitHub

**GitHub `main` is the authoritative source for the code.** Published PyPI distributions are immutable outputs of an identifiable commit and release tag, not a second editable repository.

## One-time setup

1. Merge `.github/workflows/publish.yml` into `main` using a pull request.
2. On GitHub, open **Settings → Environments → New environment**; create an environment named exactly `pypi`. Optionally restrict deployments to tags matching `v*` and configure approvals. Do not enable an approval rule that prevents the only maintainer from approving their own releases.
3. On PyPI, sign in and open **tempdisagg → Manage → Publishing → Add a new publisher**. Select GitHub Actions with exactly:

   | Field | Value |
   | --- | --- |
   | Owner | `jaimevera1107` |
   | Repository | `tempdisagg` |
   | Workflow filename | `publish.yml` |
   | Environment name | `pypi` |

4. Confirm that PyPI lists the trusted publisher. Never paste PyPI tokens into GitHub secrets or source files for this workflow.

Setting up the workflow does **not** republish 0.6.2. Only a **new, non-prerelease GitHub Release** triggers an upload.

## How to publish a new version

One command does everything that can be automated:

```powershell
.\scripts\release.ps1 0.6.5
```

It bumps the version in `tempdisagg/_version.py`, opens the `CHANGELOG.md`
section for it, reinstalls the package, runs pytest, ruff and mypy plus
`tools/check_release.py`, commits, pushes, waits for the `tests` workflow on
that exact commit, asks you to type `PUBLISH`, and creates the tag and the
GitHub Release that triggers `publish.yml`.

It stops at the first failure and does nothing further; every step is
skipped when already done, so after fixing a problem you run the same
command again and it resumes. `-DryRun` runs every check and stops before
committing; `-SkipTests` skips the local suite.

`publish.yml` then verifies tag and version, requires the commit to be on
`main`, runs the Python matrix and the R parity tests, builds the
distributions, and waits for approval of the `pypi` environment under
**Actions → publish-to-pypi → Review deployments**. That approval stays
manual because PyPI never allows a version to be replaced or reused.

Confirm the upload with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_pypi.ps1
```

### Doing it by hand

`scripts/publish_release.ps1` performs only the tag-and-release part, for
when the version and the changelog are already committed and pushed. The
individual tools (`tools/bump_version.py`, `tools/open_changelog_section.py`,
`tools/check_release.py`) can also be run on their own.

### Before the first automated release

The workflow uses the GitHub environment `pypi` and requires a matching Trusted Publisher configured **on PyPI**. If the release job fails to authenticate, check owner, repo, workflow filename and environment name before trying again. If an upload is partial, check which distributions PyPI accepted: already-uploaded filenames cannot be uploaded again, even on retry.

### Standard local development commands (PowerShell)

```powershell
cd 'G:\My Drive\Librerías\tempdisagg'
git switch main
git pull --ff-only origin main
git switch -c fix/describe-change
# ...edit files...
.\.venv\Scripts\python.exe -m ruff check tempdisagg tests
.\.venv\Scripts\python.exe -m pytest -q
git add .
git commit -m 'Describe change'
git push -u origin fix/describe-change
```

Open a pull request rather than pushing directly to `main`. A normal `git push` **does not** publish to PyPI.

Official documentation: https://docs.pypi.org/trusted-publishers/using-a-publisher/ and https://docs.pypi.org/trusted-publishers/adding-a-publisher/.
