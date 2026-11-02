# Open-HumanScale Release Plan

**Last updated: 2026-11-02 · Current stage: public code preview · Next: main policy and evaluation**

This is the maintained plan for the **RoboTwin post-training** release. It covers
code, post-trained weights, related experiments, and evaluation. Pretraining and
mid-training pipelines/initialization weights, RoboDojo, and real-robot/GoalWAM
code are outside this plan.

Milestones are ordered by delivery priority. Dates are announced in README News
when confirmed; the items below do not imply a promised release date.

## M0 — Code preview (complete)

- [x] Publish the main HumanWAM source and RoboTwin post-training entry point.
- [x] Separate local resource paths from five configuration references.
- [x] Add the experiment index and manuscript-result provenance.
- [x] Run CPU configuration/video tests, import checks, and wheel packaging.
- [x] Publish key results, demo keyframes, and project documentation.

Evidence: [initial code commit](https://github.com/ucasmjc/OpenHumanScale/commit/f39b1e1e17d5fb4971f4efda4d178103e70cc150),
[verification record](release/verification.json), and [source manifest](release/source_manifest.json).
Passing these checks does not establish GPU or simulator reproduction.

## M1 — Main policy and evaluation (in progress)

**Deliverable:** a user can download one policy bundle and run a documented
RoboTwin evaluation with its corresponding adapter.

- [x] Identify the paper's main policy: 50kh / pre40k / W0 mid10k / post21k, A2.
- [ ] Verify the actual checkpoint bytes against the recorded SHA-256 and test full loading.
- [ ] Bind the checkpoint to its exact model configuration and normalization statistics.
- [ ] Package the inference adapter, action decoding, and observation preprocessing.
- [ ] Pin the simulator/adapter revisions, task settings, and evaluation seeds.
- [ ] Verify GPU inference and a simulator smoke rollout.
- [ ] Run the full Clean2Clean and Clean2Random evaluations; publish episode counts
  and results, distinguishing new reproductions from the paper's records.
- [ ] Create the Hugging Face Collection and publish `DAGroup-PKU/OpenHumanScale-RoboTwin`
  with a model card, download instructions, checksums, and asset license information.
- [ ] Add playable rollout demos with their task, setting, and checkpoint identified.

**Complete when:** public downloads, commands, version information, and evaluation
records form a working path from a fresh checkout to reported reproduction results.
Any difference from the paper's result must be documented.

## M2 — Reproducible post-training (planned)

**Deliverable:** a documented workflow from RoboTwin demonstrations to a new
post-trained checkpoint.

- [ ] Package demonstration conversion and data-layout instructions.
- [ ] Package prompt/text-cache and video-cache preparation.
- [ ] Provide normalization-statistics generation with provenance checks.
- [ ] Validate installation in a clean environment and publish tested dependencies.
- [ ] Verify dataset selection and cache coverage, then GPU forward/backward and saving/loading.
- [ ] Document a complete fine-tuning run from the released post-trained policy.

**Complete when:** the walkthrough is run successfully in the documented environment.
New jobs use the full intended training corpus, `val_set_proportion: 0.0`, and
`separate_ego_action_tokenizer: true`. Historical configurations remain records
of their original runs. Fine-tuning released policies does not recreate the
private pretraining-to-post-training trajectory.

## M3 — All related RoboTwin experiments (planned)

**Deliverable:** each experiment group has executable settings, paired policy
artifacts, and an explicit mapping to its evaluation results.

- [ ] Data scaling.
- [ ] Task and scene diversity.
- [ ] Cross-source action alignment.
- [ ] Human action supervision.
- [ ] Multi-view adaptation: Head-only, W0, and Future.
- [ ] Robot demonstration efficiency.
- [ ] Robot-versus-ego initialization.
- [ ] Model ablations: attention, action representation, and action-expert initialization.

For each group:

1. Resolve the appropriate source version; validate any shared implementation
   against the historical behavior before treating it as compatible.
2. Match each policy to its configuration, statistics, checkpoint step, and results.
3. Document selection rules, missing evaluations, and shared controls. Optional-IDM
   inference modes must point to their shared checkpoint; per-setting maxima must
   not be described as one policy's combined result.
4. Publish a model repository in the common Hugging Face Collection with a
   navigable variant/checkpoint index. Reference shared weights rather than
   uploading duplicate controls.

**Complete when:** every included paper experiment maps to downloadable,
compatible post-trained artifacts and runnable evaluation settings. See the
[experiment index](docs/experiments.md) for the proposed repository grouping.

## Publication and documentation (ongoing)

- [ ] Add the public paper URL and canonical BibTeX entry when available.
- [ ] Link the public project page when its release is confirmed.
- [ ] Replace or supplement demo keyframes with attributable rollout videos.
- [ ] Complete upstream revision attribution and model-asset license documentation.
- [ ] Keep known limitations, setup instructions, and troubleshooting current.

## Keeping this plan current

`ROADMAP.md` is the human-readable source of truth for release progress.

- Update the relevant checklist in the same commit that delivers the item.
  An item is complete only when its linked code, artifact, or validation is available.
- Keep the README's short release checklist aligned with these milestones.
  Announce substantive deliveries in **News**, with dates and working links.
- Record completed changes in [CHANGELOG.md](CHANGELOG.md). Planned work belongs
  here; do not list it as a completed update.
- Refresh `release/status.json` when capabilities change. Preserve dated
  verification records and clearly identify the code/assets they validate.
- If scope or priorities change, update this plan and the README together;
  do not silently expand the release to pretraining or other benchmarks.
