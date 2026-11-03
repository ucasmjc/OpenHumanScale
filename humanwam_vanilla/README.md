# HumanWAM-Vanilla

[Open-HumanScale](../README.md) · [Code guide](docs/code_guide.md) ·
[Experiments](docs/experiments.md) · [Release plan](../ROADMAP.md)

HumanWAM-Vanilla is the model pretrained on human egocentric experience and
adapted to robot manipulation. Its **full open-source release** covers data
processing, pretraining, mid-training, post-training, evaluation, and experiments.

The current package contains the model, trainer, RoboTwin loader and fine-tuning
entry point, configuration references, and CPU checks. Remaining workflows and
checkpoint bundles are being prepared; see the [release plan](../ROADMAP.md).

## Layout

```text
humanwam_vanilla/
├── src/             Model, datasets, training, and configuration utilities
├── configs/         Portable configurations and historical references
├── scripts/         Training entry point and source checks
├── tests/           CPU configuration and video tests
├── docs/            Setup, experiments, and manuscript alignment
├── release/         Source provenance and artifact availability
└── pyproject.toml   Python package and dependencies
```

## Getting started

From the repository root:

```bash
cd humanwam_vanilla
python3 scripts/check_release.py
```

For installation and fine-tuning, follow the [code guide](docs/code_guide.md).
All model-specific commands in that guide run from this directory. Internal
`fastwam` import and checkpoint parameter names are retained for compatibility.

## License

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Model assets and datasets retain their own release terms.
