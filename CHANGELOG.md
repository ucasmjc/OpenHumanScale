# Changelog

Completed public updates are recorded here. Future work is tracked in
[ROADMAP.md](ROADMAP.md); no numbered model release has been published yet.

## 2026-11-04

- Add an original blue/violet Open-HumanScale wordmark to the README header.

- Update the title and teaser from manuscript revision `3069a9e`.
- Add hybrid post-training scaling and the HumanWAM-Goal RoboTwin result:
  92.70% Clean2Clean, 84.60% Clean2Random, and 88.65% average.
- Add real-robot zero-shot task coverage alongside the existing success curves.
- Link Open-HumanScale-50k, the ModelScope upload destination, the public model
  collection, and the planned project page with explicit availability labels.
- Refresh figure provenance, resource documentation, and the release plan.

## 2026-11-03

- Organize the repository by model: move the existing package, configuration,
  scripts, tests, and provenance under `humanwam_vanilla/`; reserve `humanwam_goal/`.
- Correct the scope to the full HumanWAM-Vanilla release, remove the static ban
  on pretraining modules, and track remaining workflows explicitly in the roadmap.
- Put arXiv, Project Page, Hugging Face dataset/model, and ModelScope dataset
  logo badges at the top of the README; mark unpublished resources as coming soon.
- Add a resource index distinguishing dataset conversion, upload, and public availability.
- Refresh all six result figures from manuscript revision `9a84e02`; update the
  IDM scaling endpoints to 73.2% Scene-OOD and 42.9% Random and use HumanWAM-Vanilla
  for the pure-egocentric model.

## 2026-11-02

### Project documentation

- Organize the README around project introduction, News, scaling and zero-shot results,
  release progress, and getting started.
- Add a maintained release plan with milestones for the main policy/evaluation,
  reproducible post-training, and all related RoboTwin experiment groups.
- Keep detailed configuration and reproduction information in the code guide.
- Replace the demo-led README with six paper figures: egocentric scaling under
  two attention settings, RoboTwin zero-shot success and coverage, and real-robot
  zero-shot scaling and per-task results. Record figure sources and hashes.
- Reserve real-robot demos for the project page and align the maintained roadmap.

### Initial public code preview

- Publish the HumanWAM source and RoboTwin training entry point.
- Align the default configuration with the current 50kh/W0/A2 manuscript result;
  retain five historical configuration references.
- Record 17 passing CPU tests, core import checks, and successful wheel packaging.
- Add experiment, source, and result provenance with MIT/Apache license notices.
- Weights and the simulator evaluation interface are not included in this preview.

Source: [initial code](https://github.com/ucasmjc/OpenHumanScale/commit/f39b1e1e17d5fb4971f4efda4d178103e70cc150)
and [result/demo presentation](https://github.com/ucasmjc/OpenHumanScale/commit/c9c7bd9a4bf09b566514845be2065ffc6b2d9152).
