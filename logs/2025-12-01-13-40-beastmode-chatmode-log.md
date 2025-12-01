Actions:
- Documented how to trigger the publish workflow to PyPI using a Git tag and a PYPI_API_TOKEN secret.
- Included local alternatives (scripts/publish_to_pypi.py) and safety guidelines.

Next steps:
- User can either add PYPI_API_TOKEN to GitHub Secrets and push a tag to trigger CI, or run publish locally with TWINE credentials.
