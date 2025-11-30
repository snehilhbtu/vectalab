# Task Log - Logo Vectorization Optimization

## Actions
- Analyzed `vectalab/cli.py` and `vectalab/quality.py` to understand the current logo vectorization pipeline.
- Identified that `reduce_to_palette` was using `PIL.Image.quantize` with `MedianCut` and default dithering (Floyd-Steinberg).
- Created a reproduction script `reproduce_issue.py` to compare MedianCut (with and without dithering) vs K-means.
- Confirmed that K-means provides better color selection and avoids dithering noise which is detrimental to vectorization.
- Replaced `reduce_to_palette` in `vectalab/quality.py` with a K-means implementation using `cv2.kmeans`.
- Added a fallback to `PIL.Image.quantize` (with `dither=Image.Dither.NONE`) in case `cv2.kmeans` fails.
- Added unit tests in `tests/test_quality.py` to verify the new implementation.

## Decisions
- Switched from MedianCut to K-means clustering for palette reduction. K-means is superior for logos as it finds the dominant colors (centroids) rather than splitting the color space based on frequency, and it naturally handles anti-aliasing by clustering similar shades.
- Disabled dithering in the fallback method. Dithering creates noise that `vtracer` interprets as small paths, reducing quality and increasing file size.
- Kept the `get_optimal_palette_size` logic as is, assuming that better color selection will improve SSIM without needing more colors.

## Next Steps
- Monitor user feedback on logo vectorization quality.
- Consider adding a "strict" mode for logos that enforces a very small palette (e.g. 4-8 colors) for simpler icons.
