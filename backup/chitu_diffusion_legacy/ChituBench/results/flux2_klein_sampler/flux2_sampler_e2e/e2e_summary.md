| case | TextEncode (s) | Denoise (s) | VAEDecode (s) | End-to-end (s) | E2E speedup vs 1 GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_1gpu | 0.050 | 1.364 | 0.171 | 1.586 | 1.00x |
| cp2_dit_only | 0.050 | 0.730 | 0.094 | 0.873 | 1.82x |
| cp2_dit_sampler | 0.049 | 0.729 | 0.094 | 0.872 | 1.82x |
| cp4_dit_only | 0.049 | 0.404 | 0.063 | 0.517 | 3.07x |
| cp4_dit_sampler | 0.049 | 0.404 | 0.063 | 0.517 | 3.07x |
| cp8_dit_only | 0.049 | 0.273 | 0.044 | 0.366 | 4.33x |
| cp8_dit_sampler | 0.049 | 0.272 | 0.044 | 0.365 | 4.34x |
