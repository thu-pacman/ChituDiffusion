| layout | GPUs | variant | DiT forward (s) | denoise (s) | DiT speedup vs 1 GPU | rel. to pair |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CP2 (2 GPU) | 2 | Parallel DiT only | 80.708 | 81.439 | 1.729 |  |
| CP2 (2 GPU) | 2 | Parallel DiT + Sampler | 80.640 | 81.380 | 1.730 | +0.1% DiT |
| CFP2+CP2 (4 GPU) | 4 | Parallel DiT only | 40.442 | 40.838 | 3.449 |  |
| CFP2+CP2 (4 GPU) | 4 | Parallel DiT + Sampler | 40.418 | 40.839 | 3.452 | +0.1% DiT |
| CFP2+CP4 (8 GPU) | 8 | Parallel DiT only | 25.287 | 25.669 | 5.517 |  |
| CFP2+CP4 (8 GPU) | 8 | Parallel DiT + Sampler | 25.188 | 25.578 | 5.539 | +0.4% DiT |
