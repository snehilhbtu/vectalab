## Model weights (sam_vit_b.pth, sam_vit_h.pth)

These large model files were intentionally removed from the git history to keep the repository small and fast to clone. If you need to run features that require the segmentation / SAM models, download them separately and place the files in a local `models/` directory.

Recommended hosting & distribution options

- GitHub Releases (recommended if you want a centrally-located place tied to the repo):
  1. Create a release in the GitHub UI and upload the `.pth` files as release assets.
  2. Users can then download the assets using curl/wget or the GitHub Releases downloads page.

- Cloud storage (S3, GCS, Azure Blob):
  - Upload to a private or public bucket and provide a signed URL or public URL for downloads.

- External model hosting (huggingface.co, Zenodo, or similar model registries):
  - Useful for public models with versioning and checksums.

Where to put the files locally

1. Create a `models/` directory at the repository root (this repository already ignores `models/*.pth` and `*.pth`).
2. Place the downloaded files in `models/`:

```text
models/sam_vit_b.pth
models/sam_vit_h.pth
```

Example: Download release assets with curl

Replace the example URLs below with the actual release asset URL or object store URL where you uploaded the files.

```bash
mkdir -p models
curl -L -o models/sam_vit_b.pth "https://example.com/your-releases/sam_vit_b.pth"
curl -L -o models/sam_vit_h.pth "https://example.com/your-releases/sam_vit_h.pth"
```

If you prefer wget:

```bash
mkdir -p models
wget -O models/sam_vit_b.pth "https://example.com/your-releases/sam_vit_b.pth"
wget -O models/sam_vit_h.pth "https://example.com/your-releases/sam_vit_h.pth"
```

Security and integrity

- We recommend publishing checksums (sha256) alongside the assets and validating them after download:

```bash
sha256sum models/sam_vit_b.pth models/sam_vit_h.pth
# Compare value to published sha256s
```

Other tips

- Keep the model files outside of the repository to avoid accidentally committing them again.
- If you want to keep model files under version control, use Git LFS and follow the `git lfs migrate` workflow — but note LFS has storage/transfer costs.

If you'd like, I can help upload the models to a GitHub Release or add a tiny helper script to download them from a URL you provide.
