# RoboTwin code guide

## Release scope

This repository focuses on **RoboTwin post-training**. It includes the model,
trainer, robot dataset loader, a portable fine-tuning configuration, and five
historical configuration references. Pretraining and mid-training pipelines,
their initialization checkpoints, RoboDojo, and real-robot/GoalWAM workflows are
outside this release. The manuscript covers a broader research program.

The project name is **Open-HumanScale**; the policy described in the manuscript's
model section is **HumanWAM**. Internal `fastwam` module and parameter names are
retained to preserve compatibility with the recorded checkpoints.

## Current manuscript result

| Policy | Clean2Clean | Clean2Random | Average |
| --- | ---: | ---: | ---: |
| HumanWAM | 92.08% | 68.88% | 80.48% |

These are **manuscript-reported results**, not evaluations run from this checkout.
Both settings use the same checkpoint and unseen instructions, with 50 tasks ×
50 episodes each: 2,302/2,500 and 1,722/2,500 successful episodes.
The recorded lineage is **50kh egocentric pretraining / pre40k / W0 mid10k /
post21k**, from a post-training run scheduled for 24k steps with A2 augmentation.
The manuscript's provisional final 100kh recipe and 12k post-training budget are
separate from this evaluated checkpoint. See [paper alignment](../docs/paper_alignment.md)
and [machine-readable provenance](../release/paper_alignment.json).

## Code and experiments

```text
src/fastwam/            HumanWAM model, RoboTwin loader, trainer
src/openhumanscale/     Portable configuration validation
configs/robotwin.yaml   Fine-tuning recipe based on the current HumanWAM profile
configs/reference/     Five historical settings with external resource bindings
scripts/train.py       Config validation and training entry point
scripts/check_release.py  Static source and release checks
release/               Source fingerprints, model metadata, remaining work
```

The default source/config profile is `idm_w0_a2_50kh`. It preserves the recorded
main-result model implementation, A2 augmentation probabilities, and training
settings. The other profiles are references pending source compatibility work;
they are not advertised as executable reproductions.

[The experiment index](../docs/experiments.md) covers scaling, diversity, alignment,
action supervision, multi-view adaptation, demonstration efficiency, robot versus
ego initialization, and the newly added model ablations. Related RoboTwin
post-trained weights are planned under **one OpenHumanScale Collection in
`DAGroup-PKU`**, with one model repository per experiment group. No weight
repository or Collection has been published by this code release.

## Installation

Use Python 3.10 or newer. Dependency pins come from a historical training snapshot;
clean installation and GPU compatibility remain to be validated. The historical
baseline is PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.8. Install a matching
CUDA build of PyTorch before installing this project. FlashAttention is optional
and environment-specific.

```bash
python -m pip install -e '.[dev]'
python -m pytest
python scripts/check_release.py
```

The recorded CPU checks used a different installed environment; see
[verification](../release/verification.json) for exact versions and limitations.

## Configure fine-tuning

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` and fill every
`???` with a real local resource. Comments identify the 50 task directories;
[dataset bindings](../release/dataset_bindings.json) preserve which tasks used
repaired video views. Supply the matching normalization statistics, prompt
manifest, and text/video caches. Data preparation and cache-building entry points
are still being packaged.

`paths.resume` must identify a compatible **full RoboTwin post-trained policy**.
Checkpoint downloads are not yet published, so this preview is not a runnable
quickstart with downloadable assets. Private pretraining initializers are not
needed by the fine-tuning entry point and are not included.

```bash
python scripts/train.py --paths configs/paths.local.yaml --check-config
```

This checks configuration constraints, resource existence, and dataset metadata
without loading weights. It does not certify actual sample selection, cache
coverage, or checkpoint compatibility. New training uses the full intended
training corpus (`val_set_proportion: 0.0`), separate ego action tokenizers, and
normalization provenance validation. Rebuild statistics and caches when changing
the corpus or transforms; do not disable provenance to reuse stale statistics.

Once the corresponding assets and GPU checks are complete, use:

```bash
accelerate launch scripts/train.py --paths configs/paths.local.yaml
```

This starts a new fine-tuning schedule from post-trained weights, rather than
recreating the full trajectory from private pretraining initializers. Exact
training-state resume is outside this first entry point. Simulator evaluation
commands will be added after binding the adapter revision to the released policy.

## License and acknowledgments

The inherited FastWAM MIT license is retained in [LICENSE](../LICENSE). Vendored
LeRobot files retain their Apache 2.0 headers, with the license text included in
[licenses/Apache-2.0.txt](../licenses/Apache-2.0.txt). We acknowledge the inherited
FastWAM, Wan, and LeRobot implementations; see
[third-party provenance](../THIRD_PARTY_NOTICES.md). Code licenses do not assign a
license to weights, upstream model assets, or datasets.
