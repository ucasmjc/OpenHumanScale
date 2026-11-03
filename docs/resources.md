# Project resources

Last checked: November 3, 2026. The README badges link here until public releases
are available; they will link directly to each resource when published.

## Paper

**arXiv: coming soon.** The current manuscript is titled *Open-HumanScale: The
Emerging Capability on Embodied Foundation Model from Large Scale Hybrid Data
Scaling*. The public paper URL and canonical citation are not yet available.

## Project page

**Coming soon.** The Open-HumanScale project page will host real-robot demos.
The existing HumanNet website is a separate dataset project.

## Datasets

**50k-hour LeRobot v3 collection: upload in progress; full public release pending.**
Conversion is complete for 53,533 source shards and 31,396,799 physical episodes,
producing 26.516 TB of video, training data, and metadata. Videos remain compressed
MP4; captions are retained and text embedding caches are excluded.

| Platform | Destination | Availability |
| --- | --- | --- |
| Hugging Face | `DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3` | Private upload in progress |
| ModelScope | `DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3` | Partial upload paused while Hugging Face is prioritized |

These are preparation statuses, not complete dataset releases. Public download
links and source-specific release terms will be added after publication.
The existing [HumanNet dataset](https://huggingface.co/datasets/DAGroup-PKU/HumanNet)
is a separate resource, not the converted 50k-hour collection.

## Models

**Hugging Face: coming soon.** RoboTwin post-trained policies are planned under
one OpenHumanScale Collection in `DAGroup-PKU`, with one repository per experiment
group. The main policy bundle and evaluation interface are the next code/model
milestone; see the [release plan](../ROADMAP.md).

This repository releases RoboTwin post-training code and policies. Showing the
paper's HumanWAM-Goal zero-shot results does not mean its pretrained weights or
training pipeline are included in this preview.
