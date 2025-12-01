Actions:
- Staged and committed all local changes (publish script, tests, README updates, CI workflow, logs).
- Pushed commit to origin/main.

Remote response:
- Remote indicated a branch protection bypass in effect for the push ("Changes must be made through a pull request"). That can happen for admin pushes or tokens with permissions.

Next steps:
- Consider enforcing stricter protections (enforce_admins=true) to prevent bypasses and require PRs from all users.
- Optionally open a PR-based release workflow and use protected branches.
