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

The version lives in **one** place, `tempdisagg/_version.py`; `pyproject.toml`
reads it dynamically and `tempdisagg.__version__` re-exports it.

1. Merge the work into `main` through pull requests, with CI green.
2. On `main`, set the new version and describe it:

   ```powershell
   python tools/bump_version.py 0.6.5
   # rename "## Unreleased" in CHANGELOG.md to "## 0.6.5 — YYYY-MM-DD" and start a new empty "## Unreleased"
   python tools/check_release.py          # PASS, Version: 0.6.5
   git commit -am "release: 0.6.5"
   git push origin main
   ```

3. Wait until the `tests` workflow is green on that commit (`gh run watch`).
4. Create the release from the local checkout:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\publish_release.ps1
   ```

   The script re-runs the checklist, requires a clean tree synchronised with
   `origin/main`, an unused tag and a successful CI run for the exact commit,
   then asks you to type `PUBLISH`. It creates the tag `v0.6.5` and the GitHub
   Release, which triggers `publish.yml`.
5. `publish.yml` verifies tag and version, requires the commit to be on
   `main`, runs the Python matrix with ruff, mypy and the suite, regenerates
   the R references and runs the parity tests, builds the distributions and
   then waits for approval of the `pypi` environment. Approve it under
   **Actions → publish-to-pypi → Review deployments**.
6. Verify from a fresh environment:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\verify_pypi.ps1
   ```

Never reuse a tag or a version: PyPI refuses replacement files. For a
correction, release the next patch version.

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
