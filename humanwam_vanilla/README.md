# HumanWAM-Vanilla

The HumanWAM-Vanilla implementation for [Open-HumanScale](../README.md),
covering human-experience pretraining and robot post-training.

🚧 Code and documentation are being organized for release.

## Getting Started

- [Weight Download](../README.md#vanilla-weight-download) — pretrained and RoboTwin post-trained checkpoints; uploading.
- [Environment Setup & Quick Start](../README.md#vanilla-quick-start) — work in progress.
- [Training](../README.md#vanilla-training) — local and distributed launch commands.
- [Inference](../README.md#vanilla-inference) — work in progress.
- [Reproduction Guide](../README.md#vanilla-reproduction-guide) — work in progress.

## Experiment Configurations

Each series has a directory under `configs/`. `pretrain_*.yaml`,
`midtrain_*.yaml`, and `posttrain_*.yaml` describe the corresponding stages;
shared settings live in `_base_/`. A recipe's `_base_` paths are relative to
that YAML file, and command-line overrides apply last.

| Series | Configurations |
| --- | --- |
| [Scaling — isolated attention](configs/scaling_ia/) | 100, 500, 1,000, 5,000, 10,000 hours; common RoboTwin post-training and No-Ego |
| [Scaling — IDM](configs/scaling_idm/) | 100, 500, 1,000, 5,000, 20,500, 50,000 hours; common RoboTwin post-training and No-Ego |
| [Task and scene diversity](configs/diversity/) | Matched-diversity pretraining; scaled-diversity reuses the IA scaling configs |
| [Demonstration efficiency](configs/demonstration_efficiency/) | No-Ego / 500h / 5,000h initialization × 5 / 10 / 25 / 50 demonstrations per task |
| [Human action supervision](configs/action_supervision/) | 0%, 50%, or 100% supervision during pretraining; full supervision during robot post-training |
| [Multi-view adaptation](configs/multiview/) | Head only, wrist conditioning, or wrist prediction; with or without EgoPro mid-training |
| [Attention and action representation](configs/attention/) | Isolated, Mutual, IDM, and Optional-IDM attention; relative or absolute actions |
| [Action-expert initialization](configs/action_initialization/) | Interpolated video weights or random initialization |

🚧 These recipes are being prepared. Missing implementations and resource
bindings are left as `???`; the pretraining loader, several ablation processors,
and the exploration evaluator are still in progress. Source-mix alignment and
real-robot pretraining are excluded from this release work.

The [corpus index catalog](configs/corpus_indices.yaml) links to the published
[Open-HumanScale-50K indices](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50K/tree/main/indices).
The 100h–5,000h files describe historical scaled-diversity selections; their
mapping to the released LeRobot parts is pending. The 20.5kh and 50kh source
selections are mapped. The published 10kh subset was frozen on November 4 and
has not been bound to the report's historical 10kh experiment. Matched-diversity,
four-source ablation, and EgoPro multi-view selections remain pending.

For the head-view RoboTwin recipes, fill in the local paths and inspect the
configuration from `humanwam_vanilla/`:

```bash
cp configs/paths.exploration.example.yaml configs/paths.local.yaml
python scripts/train.py --config configs/scaling_ia/posttrain.yaml \
  --paths configs/paths.local.yaml --print-config

# After the corresponding checkpoints, statistics, and data are available:
NPROC_PER_NODE=8 bash scripts/train.sh \
  --config configs/scaling_ia/posttrain.yaml --paths configs/paths.local.yaml
```

The reference launch uses **8 GPUs**, a per-GPU batch of **64**, and the recipe's
gradient accumulation. When changing hardware, preserve
`world_size × batch_size × gradient_accumulation_steps`. Post-training uses a
global batch of 512 and all selected training episodes (`val_set_proportion: 0.0`).
Low-demonstration recipes require the exact per-task exclusion manifest and
statistics computed on that selected corpus. The 50% action-supervision recipe
requires a fixed original-episode mask, not random masking on each batch.

The [experiment index](configs/experiments.yaml) records the exploration
evaluation protocol: 50 tasks × 50 episodes under Clean, Scene-OOD, and Random,
selecting the best checkpoint separately for each setting. Preserve the full
learning-rate schedule when evaluating an intermediate checkpoint. Historical
matched-diversity runs include both 12k and 15k schedules; their selected
checkpoints are listed in [diversity/series.yaml](configs/diversity/series.yaml).
`configs/robotwin.yaml` remains the separate main-result recipe.

## Code

- [src/humanwam](src/humanwam/) — models, datasets, training, and utilities.
- [configs](configs/) — training configurations and local resource paths.
- [scripts/train.sh](scripts/train.sh) — training launcher.
- [evaluation](evaluation/) — RoboTwin evaluation integration.

See the project README for the [release plan](../README.md#release-plan),
[license](../README.md#license), and [citation](../README.md#citation).
