# Alignment with the manuscript

The initial code preview was checked against manuscript revision
`99c34327fc900c7087662e4a88a00ca8a8ceeab9` from November 2, 2026.
That manuscript's title was *Open-HumanScale: The Emerging Capability on Embodied
Foundation Model from Large Scale Hybrid Data Scaling*. The policy in the
pure-egocentric model section was named **HumanWAM**.

The README and its six figures were refreshed against revision
`9a84e02f539ea5c928c0852b1c6248e2cd68f7e5` on November 3, 2026. The model is now
named **HumanWAM-Vanilla** in the introduction and model section. The IDM scaling
study now reports 73.2% Scene-OOD and 42.9% Random at 50kh; these study results
are separate from the main policy benchmark below. This documentation refresh
does not change the released source profile or imply new reproduction tests.

The README was refreshed again against revision `3069a9e8166b72df04e0d31c71af4a0b41cc7b94`
on November 4, 2026, titled *Open-HumanScale: An Open Recipe for Scaling Human
and Robot Experience for Embodied Foundation Models*. It now includes the new
teaser, hybrid post-training scaling curve, and HumanWAM-Goal's refined
post-training result (92.70% Clean2Clean, 84.60% Clean2Random, 88.65% average).
The controlled hybrid scaling recipe reaches 74.36% Random; the 84.60% result
uses additional augmentation and post-training refinements. The Goal zero-shot
result remains 13.94% average on RoboTwin and 30.5% across 20 real-robot tasks.
These Goal results do not describe the Vanilla checkpoint below. Source figures
and hashes are tracked in [the asset manifest](../../assets/SOURCES.json).

The main Vanilla RoboTwin evaluation is the **50kh / pre40k / W0 mid10k / post21k A2**
policy, with recorded SHA-256
`470e8914cf87c98c94b0e157c6de603dd17267239dac92e1aadacd885b1081cf`.
Its source run is `rtw-e50-p40-w0-s10k-a2-p24-b64x8-1028-r1`.
The checkpoint hash is recorded in the manuscript's evaluation provenance; this
export has not independently read or hashed the checkpoint bytes.

The configuration describes a 24k-step training schedule. Selection of post21k
for the paper does not change that schedule to 21k. A2 light-field and geometric
jitter probabilities are both 0.8. Each of the two reported evaluation settings
uses 2,500 episodes and unseen instructions, with the same checkpoint:

- Clean2Clean: 2,302 successes, 92.08%.
- Clean2Random: 1,722 successes, 68.88%.
- Arithmetic mean: 80.48%.

The first internal draft used the earlier 20.5kh/pre50k/mid10k/post24k policy.
That setting remains a historical reference; it is no longer the default or the
checkpoint associated with the current main-result numbers.

The manuscript also describes a provisional 100kh recipe and a 12k RoboTwin
post-training budget. These are not relabeled as the lineage of the evaluated
50kh/post21k policy. HumanWAM-Vanilla is now scoped for a full release, including pretraining.
The current packaged entry point still covers RoboTwin fine-tuning; additional
workflows and checkpoint assets are being prepared.

The new attention/action-representation and action-initialization ablations have
been added to the experiment index. Their executable profiles and exact weight
bindings remain pending. The published preview preserves this distinction
instead of claiming that every paper experiment is already reproduced.
