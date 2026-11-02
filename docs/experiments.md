# RoboTwin experiment index

This index follows the November 2, 2026 manuscript. Repository names below are
planned Hugging Face destinations in `DAGroup-PKU`, grouped into one
`OpenHumanScale` Collection; they are not live download links.

| Experiment group | Planned model repository | First code preview |
| --- | --- | --- |
| HumanWAM main result | `OpenHumanScale-RoboTwin` | Current 50kh/W0/A2 source and config; weights/eval pending |
| Data scale | `OpenHumanScale-Data-Scaling` | Historical profile references; complete mapping pending |
| Task/scene diversity | `OpenHumanScale-Data-Diversity` | Experiment index; executable recipes pending |
| Cross-source alignment | `OpenHumanScale-Alignment` | Experiment index; executable recipes pending |
| Human action supervision | `OpenHumanScale-Action-Supervision` | IA reference config; source compatibility pending |
| Head-only / W0 / Future | `OpenHumanScale-Multiview` | IDM reference configs; other routes pending |
| Robot demonstration efficiency | `OpenHumanScale-Demo-Efficiency` | Experiment index; executable recipes pending |
| Robot versus ego pretraining | `OpenHumanScale-Robot-vs-Ego` | Downstream RoboTwin results only; recipes pending |
| Model ablations | `OpenHumanScale-Model-Ablations` | Added from latest manuscript; checkpoint/config binding pending |

The last group includes attention strategy, relative versus absolute actions,
and action-expert initialization. Optional-IDM's first-frame-only and video-to-action
modes share a checkpoint; they must not be counted as independent trained models.
Several historical study tables select the best available complete checkpoint
independently per evaluation setting. Those maxima do not describe one policy
with all reported scores. The HumanWAM main-result table instead uses one shared
checkpoint across Clean2Clean and Clean2Random.

Within a model repository, variants and checkpoints will use separate directories,
with model/config/statistics/evaluation provenance. Shared baseline weights should
be uploaded once and referenced by other experiment manifests. The earlier
495-checkpoint inventory predates the latest main-result and ablation additions;
it is not presented here as a complete current release catalog.

The manuscript's joint human/robot scaling, real-robot results, and GoalWAM/ICL
sections are outside this RoboTwin-only code preview. Their inclusion in the
paper does not imply their code or weights are included here.
