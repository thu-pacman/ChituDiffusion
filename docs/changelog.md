# Changelog

All notable changes to Smart-Diffusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-16

### Changed
- Unified launch entry to `run.sh` only
- Moved startup/system parameter control to `system_config.yaml`
- Replaced CFG parallel toggle with explicit `cfp -> infer.diffusion.cfg_size` flow
- Updated documentation to use `bash run.sh system_config.yaml ...` examples

## [0.1.1] - 2026-02-16 Chuxi

### Added
- Comprehensive English docstrings for core modules
- Enhanced README.md with:
  - Improved structure and formatting
  - Comprehensive feature descriptions
  - Detailed installation instructions with uv support
  - Usage examples and configuration guide
  - Contributing guidelines and roadmap
- Complete documentation website using MkDocs Material
  - Installation guide
  - Quick start tutorial
  - Architecture overview
  - FAQ section
  - Configuration guide
- GitHub Actions workflow for automatic documentation deployment
- Support for documentation search and code highlighting

### Changed
- Reorganized documentation structure with clear sections
- Improved code documentation standards across the codebase

## [0.1.0] - 2026-01-27

### Added
- Initial release of Smart-Diffusion
- Support for Wan-T2V series models (1.3B, 14B, A14B)
- Multiple attention backend support:
  - FlashAttention (default)
  - SageAttention (quantized)
  - SpargeAttention (sparse)
- Memory optimization features:
  - Low memory mode with model offloading
  - VAE tiling support
  - Multi-level memory management (0-3)
- FlexCache system for feature reuse:
  - TeaCache strategy
  - Pyramid Attention Broadcast (PAB) strategy
- Parallelism support:
  - Context Parallelism (CP)
  - Classifier-Free Guidance (CFG) parallelism
- Evaluation support:
  - VBench custom-mode evaluation
- Configuration system:
  - Hydra-based configuration
  - Three-tier parameter system (Model/User/System)
- Task management:
  - Task pool and scheduler
  - Request serialization for distributed execution

### Known Issues
- Data parallelism not yet implemented
- Limited model support (Wan-T2V only)
- Documentation incomplete in some areas

## Future Roadmap

### Planned Features
- [ ] Models
  - [ ] Flux-2
  - [ ] FireRed-Image-edit
  - [ ] Longcat
- [ ] AutoVideoParallel
  - [ ] DiTango
  - [ ] Hybrid parallelism combinations
- [ ] More acceleration algorithms
  - [ ] Additional cache strategies
  - [ ] Quantization improvements
- [ ] Production features
  - [ ] HTTP API server
  - [ ] Batching and request queuing
  - [ ] Monitoring and metrics
- [ ] Better operator implementations
  - [ ] Custom CUDA kernels
  - [ ] Triton implementations
- [ ] Comprehensive benchmarks
  - [ ] Performance comparisons
  - [ ] Quality metrics

### Documentation Improvements
- [ ] Complete API reference for all modules
- [ ] More usage examples
- [ ] Video tutorials
- [ ] Community contributions guide

## Contributing

See [Contributing Guide](contributing/developer-guide.md) for how to contribute to Smart-Diffusion.

---

For detailed commit history, see [GitHub Commits](https://github.com/chen-yy20/SmartDiffusion/commits/main).
