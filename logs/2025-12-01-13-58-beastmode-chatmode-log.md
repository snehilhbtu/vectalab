Actions:
- Fixed GitHub Actions publish workflow to use pypa/gh-action-pypi-publish@release/v1 (previously an invalid @release ref) and updated README docs accordingly.
- Committed and pushed the fix to origin/main.

Result:
- This resolves the CI error "Unable to resolve action `pypa/gh-action-pypi-publish@release`, unable to find version `release`." Use of `release/v1` or an explicit tag avoids that problem.

Next steps:
- After adding a PYPI_API_TOKEN secret, pushing a tag like `vX.Y.Z` will trigger the workflow successfully.
