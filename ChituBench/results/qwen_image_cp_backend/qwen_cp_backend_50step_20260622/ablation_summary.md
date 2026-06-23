| layout | GPUs | variant | DiT forward (s) | denoise (s) | DiT speedup vs 1 GPU | rel. to pair |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CP2 (2 GPU) | 2 | AGCP (all-gather KV) | 80.846 | 81.614 | 1.724 |  |
| CP2 (2 GPU) | 2 | UCP (ring / Ulysses) | 81.373 | 82.128 | 1.712 | -0.7% DiT |
| CFP2+CP2 (4 GPU) | 4 | AGCP (all-gather KV) | 40.565 | 40.980 | 3.435 |  |
| CFP2+CP2 (4 GPU) | 4 | UCP (ring / Ulysses) | 41.071 | 41.460 | 3.393 | -1.2% DiT |
| CFP2+CP4 (8 GPU) | 8 | AGCP (all-gather KV) | 25.257 | 25.645 | 5.517 |  |
| CFP2+CP4 (8 GPU) | 8 | UCP (ring / Ulysses) | 27.023 | 27.465 | 5.157 | -7.0% DiT |
