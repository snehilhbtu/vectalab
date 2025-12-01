# VS Code shell integration with zsh + tmux (macOS)

Short, tested steps to make VS Code shell integration work nicely when you use zsh and tmux heavily — and keep Copilot inline suggestions fast.

## Goals
- Keep VS Code shell-integration features (cwd detection, command decorations, terminal IntelliSense) working reliably.
- Avoid leaking VS Code-specific sequences into native tmux shells or remote sessions.
- Make sure tmux sessions started in VS Code preserve the environment needed for integration.

## 1) Manual, safe zsh setup (recommended)
Add this to your `~/.zshrc` (or put it at the top of the file):

```sh
# Load VS Code shell integration only when we're actually running inside the VS Code terminal
if [[ "$TERM_PROGRAM" == "vscode" ]]; then
  INTEGRATION_PATH=$(code --locate-shell-integration-path zsh 2>/dev/null)
  if [[ -n "$INTEGRATION_PATH" && -r "$INTEGRATION_PATH" ]]; then
    # source the integration script
    . "$INTEGRATION_PATH"
  fi
fi
```

Why this is good:
- `code --locate-shell-integration-path zsh` finds the installed integration script path.
- Only sources it when inside VS Code, avoiding accidental injection into unrelated shells.
- If you care about micro-startup time, resolve `code --locate-shell-integration-path zsh` once and inline the absolute path in your config.

## 2) Make tmux sessions cooperate
If you want tmux shells to keep the VS Code environment (so integration stays working when you start tmux inside VS Code):

- Start tmux from a VS Code terminal (the session inherits TERM_PROGRAM).
- Or make `TERM_PROGRAM` propagate into tmux windows by adding this in `~/.tmux.conf` and reloading or restarting tmux:

```
# keep TERM_PROGRAM available to panes
set -g update-environment "TERM_PROGRAM"
```

- For an existing tmux server, run from a VS Code shell before attaching:
```
# ensures new windows inherit the var
tmux set-environment -g TERM_PROGRAM vscode
```
- Verify inside the tmux pane: `echo $TERM_PROGRAM` should show `vscode`.

Notes:
- Do not blindly copy-paste shell integration scripts into remote shells (e.g. ssh) unless you understand the side effects.
- When using long-lived tmux servers launched outside VS Code, prefer launching a fresh tmux from the VS Code terminal for the best integration.

## 3) Copilot and inline suggestions (workspaces)
- The workspace has `editor.inlineSuggest.enabled` turned on to keep inline AI completions available.
- If you use GitHub Copilot, make sure the extension is installed and enabled for this workspace (you can toggle per workspace).

## 4) Quick validation steps
1. Open a VS Code terminal (ensure profile is `zsh`), run `echo $TERM_PROGRAM` → should output `vscode`.
2. Start `tmux` from that terminal and in a newly created pane run `echo $TERM_PROGRAM` → should still be `vscode` if environment propagation works.
3. Try a few shell commands to validate decorations and the "run recent command" feature.
4. In the editor, verify Copilot inline suggestions appear (try typing a function); use `Cmd+.` or Copilot's commands to open chat.

## 5) Troubleshooting and tips
- If you see decorations or sequences in other terminals, remove the shell integration sourcing line from your `~/.zshrc` (or wrap it in the `TERM_PROGRAM == vscode` guard above).
- If shell startup is slow, inline the integration path into `~/.zshrc` (avoid running `code` repeatedly to resolve path).
- If Copilot suggestions are slow: check installed extensions, disable or limit other inline suggestion providers, and make sure `editor.inlineSuggest.enabled` is `true`.

---

If you'd like, I can prepare a small `checks.sh` script that validates your `$TERM_PROGRAM` and whether the integration file is present (and can optionally attempt to auto-run `tmux set-environment -g TERM_PROGRAM vscode`). Tell me if you'd like me to add that.
