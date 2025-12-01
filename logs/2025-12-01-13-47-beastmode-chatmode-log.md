Actions:
- Documented how to trigger the GitHub Actions publish workflow by pushing tags and by using manual triggers (workflow_dispatch) as an option.
- Explained safe test flow for TestPyPI and local dry-run alternative using scripts/publish_to_pypi.py.

Next steps:
- User can add PYPI_API_TOKEN to GitHub Secrets and push a v* tag to trigger production publish.
- Optionally allow manual `workflow_dispatch` runs or test on TestPyPI first.
