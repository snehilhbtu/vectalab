#!/usr/bin/env bash
set -euo pipefail

# Simple helper script to download model files into ./models
# Customize MODEL_BASE_URL or pass explicit URLs as args.

: ${MODEL_BASE_URL:-"https://example.com/your-releases"}

mkdir -p models

if [ "$#" -eq 0 ]; then
  echo "No arguments provided; using default filename list and MODEL_BASE_URL=$MODEL_BASE_URL"
  urls=(
    "$MODEL_BASE_URL/sam_vit_b.pth"
    "$MODEL_BASE_URL/sam_vit_h.pth"
  )
else
  urls=("$@")
fi

for url in "${urls[@]}"; do
  filename=$(basename "$url")
  echo "Downloading $filename from $url → models/$filename"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "models/$filename" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "models/$filename" "$url"
  else
    echo "neither curl nor wget found — cannot download $url" >&2
    exit 1
  fi
done

echo "Done — downloaded $(( ${#urls[@]} )) files to models/"
