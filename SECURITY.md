# Security policy

`tempdisagg` is a scientific Python package. Security reports are distinct from ordinary numerical bugs, incorrect estimates, or requests for new methods.

## Supported versions

Security fixes are ordinarily considered for the **latest release published on PyPI**. Older releases are not guaranteed to receive backported fixes. Support decisions depend on the reported issue and maintainer capacity.

## Reporting a vulnerability

**Do not publish exploit details, credentials, tokens, or sensitive data in a public GitHub issue or pull request.**

Use GitHub's private vulnerability reporting:

**[Report a vulnerability privately](https://github.com/jaimevera1107/tempdisagg/security/advisories/new)**

This requires **Private vulnerability reporting** to be enabled in the repository's GitHub security settings. If the private reporting option is unavailable, open a public issue **only to ask for a private contact channel**; do not disclose the vulnerability itself.

A useful private report includes:

- The affected `tempdisagg` version and Python environment.
- A clear description of the vulnerability and possible impact.
- The relevant code path, configuration, or dependency.
- Minimal, safe reproduction steps or a proof of concept.
- Any known mitigation or workaround.

The maintainer will review reports as availability permits; no guaranteed response or remediation timeline is implied.

## Ordinary defects

Incorrect estimates, aggregation violations, numerical instability, and API errors **without a security impact** should be reported using the public [Bug report](https://github.com/jaimevera1107/tempdisagg/issues/new/choose) form.

## Release provenance

Official Python releases are published to PyPI from tagged GitHub Releases through GitHub Actions and PyPI Trusted Publishing. See [docs/RELEASING.md](docs/RELEASING.md). Never share publishing tokens or credentials in issues or pull requests.
