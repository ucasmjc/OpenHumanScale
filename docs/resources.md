# Project resources

Last checked: November 4, 2026. Links follow the latest manuscript; availability
is listed separately because some destinations are still being prepared.

## Paper

**arXiv: coming soon.** The current title is *Open-HumanScale: An Open Recipe for
Scaling Human and Robot Experience for Embodied Foundation Models*.
The public paper URL and canonical citation are not yet available.

## Project page

[Open-HumanScale project page](https://dagroup-pku.github.io/open-humanscale/)
— **coming soon**. This is the URL specified by the manuscript; it currently
returns HTTP 404. Real-robot demos will be hosted here.

## Datasets

**Open-HumanScale-50k** is the paper's curated 50,000-hour egocentric corpus,
with 35.9 million clips, hand-motion annotations, and language descriptions.
Its five sources are HumanNet, EgoSuite-Open100K, EgoDex, Gen-HumanEgo, and EgoVerse.

| Platform | Destination | Availability |
| --- | --- | --- |
| Hugging Face | [Open-HumanScale-50k](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k) | Canonical destination in the manuscript; not publicly accessible yet |
| ModelScope | [EgoCorpus-50kH-LeRobot-v3](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) | Existing upload destination; partial upload paused while Hugging Face is prioritized |

LeRobot v3 conversion is complete for 53,533 source shards and 31,396,799 physical
episodes, producing 26.516 TB of video, training data, and metadata. This packaging
count uses physical episodes, whereas the manuscript reports clips; the two
counts should not be used interchangeably. Videos remain compressed MP4;
captions are retained and text embedding caches are excluded.

The active Hugging Face uploader still targets the private preparation repository
[`EgoCorpus-50kH-LeRobot-v3`](https://huggingface.co/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3).
The paper's new public name does not mean that those uploads have already moved
or that a complete download is available. Publication under the final name,
download instructions, and source-specific release terms are pending.

The existing [HumanNet dataset repository](https://huggingface.co/datasets/DAGroup-PKU/HumanNet)
is a separate resource, not the converted Open-HumanScale-50k release.

## Models

The [Open-HumanScale Hugging Face Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanscale)
is public and currently empty. **Checkpoints are coming soon.** We plan one model
repository per experiment group, with matching configurations and normalization
statistics. See the [release plan](../ROADMAP.md).

The HumanWAM-Vanilla code lives under [humanwam_vanilla/](../humanwam_vanilla/),
with individual workflows and checkpoints marked as available when packaged.
[HumanWAM-Goal](../humanwam_goal/) will be released separately within this repository;
its code and weights are not available yet.

## Result and figure provenance

The README follows manuscript revision `3069a9e8166b72df04e0d31c71af4a0b41cc7b94`
from November 4, 2026. Figure paths, source hashes, and rendering details are
recorded in [assets/SOURCES.json](../assets/SOURCES.json). These are paper-reported
results, not newly reproduced evaluations of this code preview.
