# Task Log - SOTA Optimization

## Actions
- Created `scripts/test_bayesian.py` to test the experimental Bayesian method (failed due to missing model weights).
- Created `scripts/optimize_hifi_params.py` to systematically test different `vtracer` parameter configurations.
- Tested 3 configurations: `ultra` (baseline), `ultra_spline`, and `ultra_max` (max detail).
- Identified `ultra_max` as the best performer (94.72% SSIM on subset, faster execution).
- Updated `vectalab/optimize.py` to upgrade the standard `ultra` preset with the `ultra_max` settings.
- Ran the full test suite to verify the improvements.

## Results
- **Complex Scenes**:
  - `tiger.svg`: Improved from 95.38% to 95.96% SSIM.
  - `tommek_Car.svg`: Improved from 95.79% to 96.03% SSIM.
  - `rg1024_metal_effect.svg`: Improved from 89.82% to 90.08% SSIM.
  - Overall average SSIM remained stable (~93.4%), but key difficult images showed improvement.
- **Performance**: The new settings are slightly faster despite higher precision, likely due to simpler layer handling (`layer_difference=0`).

## Lessons/Insights
- `mode='polygon'` generally outperforms `mode='spline'` for complex realistic images in `vtracer`, likely due to better edge preservation.
- Reducing `layer_difference` to 0 allows for maximum detail capture, which is crucial for SOTA performance on complex scenes.
- The Bayesian method requires proper model weight setup to be a viable alternative.

## Next Steps
- To reach true SOTA (>98% on complex scenes), we likely need to move beyond `vtracer` to a differentiable renderer that supports gradients (like the Bayesian method or DiffVG).
- Fix the Bayesian method's model loading issue.
