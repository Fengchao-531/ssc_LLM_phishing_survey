# Detectors

This directory contains the academic and industrial detector implementations/configurations used in the benchmark.

- `academic/`: academic detector implementations and local model artifacts where already included in the public repository.
- `industrial/`: industrial/open-source detector wrappers and configuration.
- `examples/`: small example inputs for smoke testing.
- `requirements.txt`: public Python dependencies used by the detector wrappers.

Dataset-construction and cleaning helpers are not part of this release. Benchmark-ready inputs are under `../data/benchmark/`, and released detector outputs used by the paper analyses are under `../results/detector_predictions/`.

Some detectors require local model checkpoints, external services, or environment-specific configuration. API keys and local private paths must be supplied by the user and are not committed to the repository.
