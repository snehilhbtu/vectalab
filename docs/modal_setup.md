# Cloud (Modal) — short guide

Use Modal.com for remote SAM (Segment Anything Model) runs when local hardware cannot run the heavy segmentation models (e.g., vit_h).

## Quick checklist

- Modal account and API token
- (Optional) `modal` CLI installed for `modal setup`

How to enable

1. Configure Modal credentials (interactive): `modal setup` — or export `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in CI/headless environments.
2. Use the `--use-modal` flag together with `convert --method sam`:

```bash
vectalab convert input.png --method sam --use-modal
```

## Notes

- Vectalab will stream images to a Modal container and return segmentation masks; this flow is purely optional and controlled by `--use-modal`.
- First run may be slower due to cold-start and weight downloads. Repeated runs benefit from Modal volume caching.
- If you don't have Modal, use `--method sam` locally (if your machine supports it), or prefer `hifi` / `bayesian` methods.

## Troubleshooting

- Authentication errors: re-run `modal setup` or confirm env vars.
- Timeouts: large images may exceed default limits — split or reduce resolution.
