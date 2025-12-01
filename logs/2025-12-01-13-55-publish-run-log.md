Actions:
- Created and pushed annotated tag v0.1.1 to origin.
- Observed publish workflow run triggered for v0.1.1; followed logs.

Result:
- Build step succeeded and created distributions (vectalab-0.1.0.tar.gz + wheel).
- Publish step failed early with "Trusted publishing exchange failure: OpenID Connect token retrieval failed: missing or insufficient OIDC token permissions" error.

Diagnosis:
- The workflow attempted Trusted Publishing (OIDC) but the job lacked id-token write permissions in the workflow, causing the OIDC exchange to fail.
- Also, `PYPI_API_TOKEN` is not configured in repository secrets — without a token the action falls back to trusted publishing and needs id-token permissions.
- pyproject.toml version remains 0.1.0 while tag was v0.1.1 (this mismatch can be confusing — typically bump pyproject version to match tag before publish).

Next steps:
1. Add PYPI_API_TOKEN secret (recommended) and re-run the failed workflow or push a new tag (v0.1.2) to trigger a fresh run.
2. If you prefer Trusted Publishing (OIDC) instead of token secrets, update workflow permissions to include `permissions: id-token: write` and configure PyPI Trusted Publishers (requires additional setup).
3. Optionally bump the version in pyproject.toml to match tag before publishing.

Commands to re-run after adding secret:
- Re-run existing run: `gh run rerun 19812778762 --repo raphaelmansuy/vectalab`
- Or push a new tag: `git tag -a v0.1.2 -m "Release v0.1.2" && git push origin v0.1.2`
