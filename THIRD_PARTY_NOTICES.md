# Third-party provenance

The existing FastWAM MIT license is preserved in `LICENSE`. Per-file copyright
and license headers are retained, including the Hugging Face LeRobot Apache 2.0
headers under `src/fastwam/datasets/lerobot/lerobot/`. The complete Apache 2.0
license text is included at `licenses/Apache-2.0.txt`. Local modifications to
vendored video readers are identified by the original/exported hashes in
`release/source_manifest.json`.

Wan model implementations and component-loading code are inherited from the
historical FastWAM source snapshot and retain that snapshot's licensing and
any per-file notices. Exact upstream Wan revisions remain a provenance task;
this preview does not claim a newly verified upstream source lineage.
The source manifest records original and exported SHA-256 hashes for every
copied file; it does not certify upstream licensing.

Model weights, the Wan backbone/VAE/tokenizer assets, and RoboTwin data have
separate provenance and licensing requirements. No new weight license is assigned
by this code draft.
