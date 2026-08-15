# Engineering result tables

## Matched cascade versus monolithic comparisons

| comparison | thermal ratio | control | lower median endpoints | separated seed ranges | geometric median reduction |
|---|---:|:---:|---:|---:|---:|
| FNO, no baseline | 1.04x | pass | 10/10 | 4/10 | 15.9x |
| FNO-COD | 1.26x | pass | 10/10 | 10/10 | 29941.0x |
| MIONet, no baseline | 1.04x | pass | 10/10 | 10/10 | 157.4x |
| MIONet-COD | 1.11x | pass | 10/10 | 7/10 | 13.5x |
| PI-DeepONet, no baseline | 2.58x | confounded | 8/10 | 5/10 | not interpreted |
| PI-COD | 1.73x | pass | 10/10 | 10/10 | 428.4x |
| S-DeepONet, no baseline | 1.00x | pass | 10/10 | 10/10 | 161.5x |
| S-DeepONet-COD | 1.08x | pass | 10/10 | 6/10 | 17.2x |

## Baseline-equipped in-cascade implementations

| implementation | thermal MAE, degC | swing gate | swing ratio | K=0.95 gas-error range, ppm | K=1.10 gas-error range, ppm |
|---|---:|:---:|---:|---:|---:|
| PI-COD | 0.405 [0.306-0.476] | 7/7 | 1.0112 [1.0055-1.0176] | 0.001599-0.1233 | 0.1892-1.108 |
| FNO-COD | 0.397 [0.272-0.564] | 7/7 | 1.0356 [1.0310-1.0465] | 0.005115-0.2894 | 0.6616-3.219 |
| MIONet-COD | 0.931 [0.490-1.275] | 7/7 | 1.1275 [1.0212-1.3245] | 0.068-3.877 | 6.352-25.54 |
| S-DeepONet-COD | 0.997 [0.675-4.737] | 7/7 | 1.1220 [1.0789-1.2673] | 0.07517-4.423 | 5.479-23.05 |
