<div align="center">

# Open-HumanScale

### An Open Recipe for Scaling Human and Robot Experience<br>for Embodied Foundation Models

**Human Data Scaling · Zero-Shot Robot Learning**

[![arXiv](https://img.shields.io/badge/arXiv-Coming_soon-B31B1B?style=flat&logo=arxiv&logoColor=white)](#-paper-and-citation)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90_Project_Page-Coming_soon-4285F4?style=flat)](https://dagroup-pku.github.io/open-humanscale/)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging_Face-Dataset_%C2%B7_Coming_soon-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k)
[![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-Model_Collection-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/collections/DAGroup-PKU/open-humanscale)
[![ModelScope Dataset](https://img.shields.io/badge/ModelScope-Dataset_%C2%B7_Coming_soon-624AFF?style=flat&logo=modelscope&logoColor=white)](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3)

[**Dataset**](#-open-humanscale-50k) · [**Scaling**](#-scaling-human-experience) · [**Zero-Shot**](#-zero-shot-robot-learning) · [**Getting Started**](#-getting-started) · [**Release Plan**](ROADMAP.md) · [**Experiments**](humanwam_vanilla/docs/experiments.md)

</div>

![Open-HumanScale overview: 50k-hour egocentric corpus, pure and hybrid scaling, RoboTwin results, and zero-shot real-robot generalization](assets/teaser.png)

**Open-HumanScale** presents an open recipe for scaling human and robot experience,
built on **Open-HumanScale-50k**, a curated **50,000-hour** egocentric corpus with
hand-motion and language annotations. **HumanWAM-Vanilla** connects
future video prediction with action generation, then adapts to robot manipulation
through post-training on robot demonstrations. **HumanWAM-Goal** explores
zero-shot execution through hybrid human–robot training and visual goals.

The **HumanWAM-Vanilla full release** covers data processing, pretraining,
adaptation, evaluation, and related experiments; packaging is in progress.
**HumanWAM-Goal** will be added as a separate model implementation.

| Model | Code | Status |
| --- | --- | --- |
| **HumanWAM-Vanilla** | [humanwam_vanilla/](humanwam_vanilla/) | Current code; remaining workflows and checkpoints are being packaged |
| **HumanWAM-Goal** | [humanwam_goal/](humanwam_goal/) | Coming soon |

See each model directory for setup and the [release plan](ROADMAP.md) for availability.

## 🔥 News

- **[2026/11/04]** Added the latest paper teaser, HumanWAM-Goal post-training results,
  hybrid scaling curves, and Open-HumanScale-50k dataset links.
- **[2026/11/03]** Updated scaling figures and results from the latest manuscript.
  The 50k-hour LeRobot v3 dataset conversion is complete; dataset uploads are in progress.
- **[2026/11/02]** The first public code preview is available: HumanWAM model and
  trainer, RoboTwin data loading, configuration references, and CPU checks.
- **[Next]** Complete the HumanWAM-Vanilla workflows and checkpoint release.
  Follow the [release plan](ROADMAP.md) for progress and deliverables.

## 📦 Open-HumanScale-50k

**50,000 hours · 35.9M clips · Video, hand motion, and language**

Curated from HumanNet, EgoSuite-Open100K, EgoDex, Gen-HumanEgo, and EgoVerse,
with hand-visibility, motion-stability, and activity filtering, followed by
cross-source alignment to a unified bimanual action representation.

| Resource | Link | Status |
| --- | --- | --- |
| Dataset | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k) | Paper's release destination; public release pending |
| Dataset mirror | [ModelScope](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) | Partial upload; public release pending |
| Model collection | [Hugging Face](https://huggingface.co/collections/DAGroup-PKU/open-humanscale) | Collection available; checkpoints coming soon |

LeRobot v3 conversion is complete; uploads are in progress. See
[resource details](docs/resources.md) for availability and packaging information.

## 📈 Scaling Human Experience

**More human experience improves downstream generalization.** Scaling egocentric
pretraining from **100 to 50,000 hours** raises Scene-OOD success from **49.1% to
73.2%** and Random success from **21.5% to 42.9%** with inverse-dynamics attention.

**Isolated attention · 100–10,000 hours**

![Egocentric data scaling with isolated attention: Clean, Scene-OOD, and Random success](assets/ego_scaling_ia.png)

**Inverse-dynamics attention · 100–50,000 hours**

![Egocentric data scaling with inverse-dynamics attention: Clean, Scene-OOD, and Random success](assets/ego_scaling_idm.png)

*After RoboTwin post-training. The scaling study reports the best success
rate per evaluation setting; dashed blue curves are empirical log-linear fits,
and gray lines are No-Ego baselines.*

### Hybrid scaling and post-training

With robot data fixed at **14,618 hours**, adding **45,938 hours** of human data
raises post-trained Random success from **70.80% to 74.36%** under the controlled
scaling recipe.

![RoboTwin post-training success as robot-only and hybrid human–robot data scale](assets/hybrid_posttraining_scaling.png)

With a refined post-training recipe and photometric augmentation,
**HumanWAM-Goal reaches 88.65% average success on RoboTwin 2.0**:

| Model | Clean2Clean | Clean2Random | Average |
| --- | ---: | ---: | ---: |
| HumanWAM-Vanilla | 92.08% | 68.88% | 80.48% |
| **HumanWAM-Goal** | **92.70%** | **84.60%** | **88.65%** |

*Paper-reported results after clean-only post-training. The refined Goal result
uses additional augmentation and recipe changes; it is separate from the controlled
scaling curve above and from the zero-shot results below.*

## 🎯 Zero-Shot Robot Learning

HumanWAM-Goal executes tasks **without target-task parameter updates**.

### RoboTwin: success and task coverage

With robot data fixed at **14,618 hours**, adding **45,938 hours** of human data
raises average zero-shot success from **11.48% to 13.94%** and per-checkpoint
task coverage from **18 to 22 of 50 tasks**.

<p align="center">
  <img src="assets/zero_shot_robotwin_success.png" width="49%" alt="RoboTwin zero-shot Clean and Random success as robot and human data scale">
  <img src="assets/zero_shot_robotwin_coverage.png" width="49%" alt="RoboTwin zero-shot task coverage per checkpoint and cumulatively across configurations">
</p>

*50 Clean and 50 Random episodes per executed task; all 50 tasks remain in the
scoring denominator. Coverage requires ≥2% mean success across the two settings.
“Cumulative” is the union across independently trained configurations.*

### Real robots: scaling and per-task results

Across **20 tasks**, average zero-shot success increases from **20.0% → 26.0% →
30.5%** as human data scale from **0 → 15,901 → 45,938 hours**, with robot data fixed.
Task coverage grows from **10/20 → 14/20 → 16/20** tasks with at least one success
in 10 trials.

![Real-robot zero-shot success by task difficulty and training-data scale](assets/zero_shot_real_robot_scaling.png)

![Real-robot zero-shot success and progress scores on 20 tasks](assets/zero_shot_real_robot_tasks.png)

*10 trials per task. The 30.5% result uses the scaling-series configuration.
“Balanced” is a separate 50:50 robot/human sampling variant at the same data scale;
it raises success on the Hard task group from 19% to 27%, with 29% success overall.*

The paper also demonstrates **visual-goal in-context learning**: a frozen policy
changes its behavior when supplied different goal images, including human goal
images. Real-robot demos will be hosted on the
[project page](https://dagroup-pku.github.io/open-humanscale/) (coming soon).
These figures report paper results. For the released HumanWAM code's
post-training benchmark and checkpoint mapping, see [result details](humanwam_vanilla/docs/paper_alignment.md).

## 🗓️ Release Plan

- [x] Organize the current HumanWAM-Vanilla source, configurations, and tests in one model directory.
- [x] Paper teaser, scaling and zero-shot figures, benchmark results, and experiment index.
- [x] Convert the 50k-hour corpus to LeRobot v3.
- [ ] Complete dataset uploads and publish download instructions.
- [ ] Package the full HumanWAM-Vanilla data, pretraining, adaptation, and evaluation workflows.
- [ ] Release checkpoints with matching configurations and normalization statistics.
- [ ] RoboTwin inference adapter and evaluation commands.
- [ ] Data preparation tools and a verified post-training walkthrough.
- [ ] Complete experiment configurations, checkpoints, and result mappings.
- [ ] Add HumanWAM-Goal code and model documentation.

Weights will be organized under the [**Open-HumanScale Collection**](https://huggingface.co/collections/DAGroup-PKU/open-humanscale) in `DAGroup-PKU`,
with **one model repository per experiment group**. Download links will appear
here when the bundles are available.

See [ROADMAP.md](ROADMAP.md) for milestone details and
[CHANGELOG.md](CHANGELOG.md) for completed updates. This is currently a code
preview; the full checkpoint-to-evaluation workflow is being prepared.

## 🚀 Getting Started

Clone the repository and inspect the source release with Python 3.10+:

```bash
git clone https://github.com/ucasmjc/OpenHumanScale.git
cd OpenHumanScale/humanwam_vanilla
python3 scripts/check_release.py
```

This source check needs no GPU or model download. Continue with the
[code guide](humanwam_vanilla/docs/code_guide.md) for installation, local data paths, and the
fine-tuning entry point. Public checkpoint downloads and simulator evaluation
are tracked in the [next milestone](ROADMAP.md#m1--main-policy-and-evaluation-in-progress).

| Resource | Contents |
| --- | --- |
| [Paper, project page, datasets, and models](docs/resources.md) | Publication links and availability |
| [Code guide](humanwam_vanilla/docs/code_guide.md) | Installation and RoboTwin post-training setup |
| [Experiment index](humanwam_vanilla/docs/experiments.md) | Experiment groups and planned weight repositories |
| [Result details](humanwam_vanilla/docs/paper_alignment.md) | Reported policy, evaluation protocol, and provenance |
| [Release plan](ROADMAP.md) | Upcoming deliverables and completion criteria |

## 🤝 Feedback

Please use [GitHub Issues](https://github.com/ucasmjc/OpenHumanScale/issues) for
bugs and reproduction questions. Include your environment, configuration, and
relevant error output when reporting a problem.

## 📄 Paper and Citation

arXiv and the project page are coming soon. The paper link and BibTeX entry will
be added when available. Results and terminology follow the November 4, 2026
manuscript; dataset and model availability is tracked in [Resources](docs/resources.md).

## License and Acknowledgments

We build on FastWAM, Wan, and LeRobot. The inherited source license is
[MIT](LICENSE); vendored components retain their own licenses and notices.
See [third-party acknowledgments](humanwam_vanilla/THIRD_PARTY_NOTICES.md).
