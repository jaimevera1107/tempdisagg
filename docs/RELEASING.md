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

1. Work on a feature, fix or documentation branch. Open a pull request. Ensure existing Python and R checks pass, then merge to `main`.
2. Update `pyproject.toml` (`[project].version`), `tempdisagg/__init__.py` (`__version__`), `CHANGELOG.md`, and any version badge in `README.md` **in a commit on `main`**. Use a new version number such as `0.6.3`: PyPI refuses replacement files for an already published version.
3. Ensure the version-bump pull request passes CI and merge it to `main`.
4. In GitHub go to **Releases → Draft a new release**. Create a tag `v0.6.3` **from `main`**, enter release notes and publish (not as a pre-release).
5. GitHub Actions `publish-to-pypi` will check that the release tag, `pyproject.toml` and `__version__` match, that the tag points to a commit in `main`, and that all tests and R parity pass. Only then does it build the exact tagged source and upload the distributions to PyPI using Trusted Publishing (OIDC).
6. Verify the new version on PyPI and install it into a *fresh* environment. Do not republish, delete/reuse the tag, or retag a published version. For any corrections, make the next version (for example `0.6.4`).

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
