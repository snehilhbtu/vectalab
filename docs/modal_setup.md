# Modal.com Integration Setup

Vectalab integrates with [Modal.com](https://modal.com) to run heavy segmentation models (SAM - Segment Anything Model) in the cloud. This allows you to use the most powerful models (like `vit_h`) even if your local machine (e.g., macOS with MPS) doesn't support them efficiently or lacks sufficient VRAM.

## Prerequisites

- A Modal.com account.
- Python 3.10+ installed.

## Authentication (Setting Model Keys)

To use the cloud-based SAM model, you must authenticate with Modal. This involves setting up your "Model Keys" (Modal API Token).

1.  **Interactive Setup (Recommended):**
    Run the following command and follow the interactive prompts:
    ```bash
    modal setup
    ```
    This command will open your web browser to the Modal dashboard where you can create a new token. The token will be automatically saved to `~/.modal.toml`.

2.  **Manual Setup (CI/CD or Headless):**
    If you cannot use the interactive browser flow, you can set the environment variables manually. Generate a token in your Modal dashboard settings and export them:
    ```bash
    export MODAL_TOKEN_ID=ak-...
    export MODAL_TOKEN_SECRET=as-...
    ```

## Usage

Once configured, you can use the `--use-modal` flag with the `convert` command:

```bash
vectalab convert input_image.png --method sam --use-modal
```

## How it Works

1.  **Local Client**: Vectalab runs locally on your machine.
2.  **Cloud Execution**: When `--use-modal` is passed, the image is sent to a Modal container running on a high-performance GPU (A10G).
3.  **Model Caching**: The SAM model weights (`vit_h`) are downloaded and cached in a Modal Volume, so subsequent runs are faster.
4.  **Result**: The segmentation masks are returned to your local machine for vectorization and SVG generation.

## Troubleshooting

-   **Authentication Error**: Ensure you have run `modal setup` and that your token is valid.
-   **Timeout**: Large images might take longer to process. The current timeout is set to 600 seconds.
-   **Cold Start**: The first run might take a minute to spin up the container and download weights. Subsequent runs will be much faster.
