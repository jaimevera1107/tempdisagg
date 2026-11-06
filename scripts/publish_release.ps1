param([string]$Owner = 'jaimevera1107', [string]$Repo = 'tempdisagg')
# Native commands (git, gh) write progress to stderr; under 'Stop' PowerShell 5
# turns that into a terminating error. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $ProjectRoot
try {
    python tools/check_release.py
    if ($LASTEXITCODE -ne 0) { throw 'Release check failed.' }
    $version = (python tools/read_project_version.py).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the version.' }
    $tag = "v$version"
    if (@(git status --porcelain).Count -gt 0) { throw 'Uncommitted changes. Commit and push first.' }
    git fetch origin main --tags
    if ($LASTEXITCODE -ne 0) { throw 'Could not reach the remote.' }
    $head = (git rev-parse HEAD).Trim()
    $remoteHead = (git rev-parse origin/main).Trim()
    if ($head -ne $remoteHead) { throw 'HEAD and origin/main differ. Push or update main before releasing.' }
    if (@(git tag --list $tag).Count -gt 0) { throw "Tag $tag already exists. Versions are never reused." }
    gh auth status
    if ($LASTEXITCODE -ne 0) { throw 'Sign in to GitHub CLI with gh auth login.' }
    $runs = gh run list --repo "$Owner/$Repo" --workflow tests.yml --branch main --limit 20 --json headSha,conclusion | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read CI runs.' }
    $verified = @($runs | Where-Object { $_.headSha -eq $head -and $_.conclusion -eq 'success' })
    if ($verified.Count -eq 0) { throw "No successful CI run for commit $head. Check GitHub Actions." }
    Write-Host "Version: $version; validated commit: $head"
    Write-Host 'Confirm FIRST: the Trusted Publisher on PyPI (workflow publish.yml, environment pypi).'
    $confirmation = Read-Host "Type PUBLISH to create release $tag, which triggers the PyPI upload"
    if ($confirmation -cne 'PUBLISH') { Write-Host 'Cancelled; nothing published.'; exit 0 }
    git tag -a $tag -m "tempdisagg $version"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the tag.' }
    git push origin $tag
    if ($LASTEXITCODE -ne 0) { throw 'Local tag created but not pushed; check before retrying.' }
    gh release create $tag --repo "$Owner/$Repo" --verify-tag --title "tempdisagg $version" --generate-notes
    if ($LASTEXITCODE -ne 0) { throw 'Tag is on the remote, but the GitHub Release could not be created.' }
    Write-Host "Release started: https://github.com/$Owner/$Repo/actions/workflows/publish.yml"
    Write-Host 'The PyPI upload happens only after the release tests and the environment approval.'
}
finally {
    Pop-Location
}
