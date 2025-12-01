Actions:
- Updated .vscode/settings.json with terminal + Copilot-related settings for zsh + tmux.
- Added .vscode/shell-integration.md with step-by-step instructions for adding integration to ~/.zshrc and a tmux config snippet to preserve TERM_PROGRAM.

Decisions:
- Kept terminal.integrated.shellIntegration.enabled true (workspaces previously set it).
- Set decorations to `whenFocus` to reduce noise in busy tmux workflows.
- Enabled inline suggestions and workspace Copilot keys in settings.json to favor Copilot users.

Next steps:
- If user still sees slowness, audit extensions and check ~/.zshrc for slow initializers.
- Optionally add a helper script to validate TERM_PROGRAM propagation to tmux.

Lessons/insights:
- Automatic shell integration is convenient but manual, conditional sourcing is safest when using tmux across different hosts.
- Propagating TERM_PROGRAM into tmux is necessary for reliable VS Code shell integration inside tmux panes.

Files changed:
- .vscode/settings.json (updated)
- .vscode/shell-integration.md (new)
