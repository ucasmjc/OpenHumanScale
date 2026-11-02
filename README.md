# Open-HumanScale · HumanWAM

Scaling human egocentric experience for robot learning. This repository releases
**HumanWAM's RoboTwin post-training code**.

[Code guide](docs/code_guide.md) · [Experiments](docs/experiments.md) · [Result details](docs/paper_alignment.md)

## Key results

**RoboTwin 2.0** — trained on 50 clean demonstrations per task.

| Model | Clean2Clean | Clean2Random | Average |
| --- | ---: | ---: | ---: |
| **HumanWAM** | **92.08%** | **68.88%** | **80.48%** |

Same 50kh-initialized checkpoint, 50 tasks × 50 evaluation episodes per setting.

**Real robots** — mean success across cup stacking, towel folding, and desk clearing.

| Model | In-distribution | Out-of-distribution |
| --- | ---: | ---: |
| No-Ego | 56.7% | 3.3% |
| **HumanWAM** | **86.7%** | **66.7%** |

10 trials per task and setting. Results reported in the paper.

## Demos

![HumanWAM: cup stacking, towel folding, and desk clearing](assets/humanwam_demos.png)

*Selected real-robot rollout keyframes and visual conditions from the paper.*

## Code release

RoboTwin code preview available; checkpoints and evaluation integration are being
prepared. Setup and reproduction status are in the [code guide](docs/code_guide.md).

[License](LICENSE) · [Acknowledgments](THIRD_PARTY_NOTICES.md)
