<#
.SYNOPSIS
    Release a new version end to end, stopping at the first problem.

.DESCRIPTION
    One command replaces the nine manual steps: bump the version, open the
    CHANGELOG section, reinstall, run the checks, commit, push, wait for CI,
    and create the GitHub Release that triggers the PyPI upload.

    The upload itself still waits for a human: PyPI never lets a version be
    replaced or reused, so the 'pypi' environment approval in GitHub is the
    last point where the release can be stopped. The script prints the link.

    Every step is skipped if it is already done, so a re-run after fixing a
    failure continues where it left off instead of starting over.

.EXAMPLE
    .\scripts\release.ps1 0.6.5
    .\scripts\release.ps1 0.6.5 -DryRun
#>
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Owner = 'jaimevera1107',
    [string]$Repo = 'tempdisagg',
    # Stop before any commit, push or tag: useful to see what would happen.
    [switch]$DryRun,
    # Skip the local test run when CI has already verified this commit.
    [switch]$SkipTests
)

# Native commands write progress to stderr; exit codes are checked instead.
$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $ProjectRoot

function Step([string]$Message) { Write-Host "`n=== $Message" -ForegroundColor Cyan }
function Run([string]$Message, [scriptblock]$Command) {
    Step $Message
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Message failed (exit $LASTEXITCODE)." }
}

try {
    if ($Version -notmatch '^\d+\.\d+\.\d+((a|b|rc)\d+)?$') {
        throw "Expected a version like 0.6.5, got '$Version'."
    }

    Step 'Checking the working tree and the branch'
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    if ($branch -ne 'main') { throw "Releases are cut from main; you are on '$branch'." }
    $dirty = @(git status --porcelain)
    if ($dirty.Count -gt 0 -and -not $DryRun) {
        throw "Uncommitted changes:`n$($dirty -join "`n")`nCommit or stash them first."
    }
    git fetch origin main --tags
    if ($LASTEXITCODE -ne 0) { throw 'Could not reach the remote.' }
    if (@(git tag --list "v$Version").Count -gt 0) {
        throw "Tag v$Version already exists. PyPI never allows a version to be reused."
    }

    Run "Setting the version to $Version" { python tools/bump_version.py $Version }
    Run 'Opening the CHANGELOG section' { python tools/open_changelog_section.py $Version }
    Run 'Reinstalling so the metadata match the new version' {
        python -m pip install -q -e ".[dev]"
    }

    if (-not $SkipTests) {
        Run 'Running the test suite' { python -m pytest -q }
        Run 'Linting' { python -m ruff check . }
        Run 'Type checking' { python -m mypy tempdisagg }
    }
    Run 'Release checklist' { python tools/check_release.py }

    if ($DryRun) {
        Write-Host "`nDry run: everything passed. Nothing was committed." -ForegroundColor Green
        Write-Host 'Re-run without -DryRun to publish.' -ForegroundColor Green
        exit 0
    }

    Step 'Committing and pushing'
    if (@(git status --porcelain).Count -gt 0) {
        git commit -q -am "release: $Version"
        if ($LASTEXITCODE -ne 0) { throw 'Could not commit the release.' }
    } else {
        Write-Host 'Version and CHANGELOG were already committed.'
    }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Could not push to origin/main.' }

    $head = (git rev-parse HEAD).Trim()
    Step "Waiting for CI on $($head.Substring(0, 7)) (several minutes)"
    # The run takes a moment to appear after the push.
    Start-Sleep -Seconds 12
    gh run watch --repo "$Owner/$Repo" --exit-status 2>$null | Out-Null
    $runs = gh run list --repo "$Owner/$Repo" --workflow tests.yml --branch main `
        --limit 20 --json headSha,conclusion | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the CI runs.' }
    $ok = @($runs | Where-Object { $_.headSha -eq $head -and $_.conclusion -eq 'success' })
    if ($ok.Count -eq 0) {
        throw "CI has not succeeded for $head. Inspect it with: gh run view --log-failed"
    }
    Write-Host 'CI is green.' -ForegroundColor Green

    Step 'Creating the release'
    Write-Host "This creates tag v$Version and starts the PyPI workflow."
    $answer = Read-Host "Type PUBLISH to continue, anything else to stop"
    if ($answer -cne 'PUBLISH') {
        Write-Host 'Stopped. The version and CHANGELOG are committed and pushed;' -ForegroundColor Yellow
        Write-Host 'run this script again with the same version to resume.' -ForegroundColor Yellow
        exit 0
    }
    git tag -a "v$Version" -m "tempdisagg $Version"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the tag.' }
    git push origin "v$Version"
    if ($LASTEXITCODE -ne 0) { throw 'The tag was created locally but not pushed.' }
    $options = @()
    if ($Version -match '(a|b|rc)\d+$') { $options += '--prerelease' }
    gh release create "v$Version" --repo "$Owner/$Repo" --verify-tag `
        --title "tempdisagg $Version" --generate-notes @options
    if ($LASTEXITCODE -ne 0) { throw 'The tag is pushed but the GitHub Release failed.' }

    Write-Host "`nRelease created." -ForegroundColor Green
    Write-Host 'Last step, in the browser: approve the pypi environment at'
    Write-Host "  https://github.com/$Owner/$Repo/actions/workflows/publish.yml"
    Write-Host 'Then confirm the upload with:'
    Write-Host '  powershell -ExecutionPolicy Bypass -File .\scripts\verify_pypi.ps1'
}
catch {
    Write-Host "`n$($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Nothing further was done. Fix the problem and run the same command again.' -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
