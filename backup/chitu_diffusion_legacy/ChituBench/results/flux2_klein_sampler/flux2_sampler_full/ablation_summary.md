| layout | GPUs | variant | DiT forward (s) | denoise (s) | DiT speedup vs 1 GPU | rel. to pair |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CP2 (2 GPU) | 2 | Parallel DiT only | 0.727 | 0.818 | 1.856 |  |
| CP2 (2 GPU) | 2 | Parallel DiT + Sampler | 0.726 | 0.817 | 1.858 | +0.1% DiT |
| CP4 (4 GPU) | 4 | Parallel DiT only | 0.401 | 0.457 | 3.366 |  |
| CP4 (4 GPU) | 4 | Parallel DiT + Sampler | 0.401 | 0.456 | 3.366 | -0.0% DiT |
| CP8 (8 GPU) | 8 | Parallel DiT only | 0.278 | 1.002 | 4.860 |  |
| CP8 (8 GPU) | 8 | Parallel DiT + Sampler | 0.273 | 0.999 | 4.938 | +1.6% DiT |
