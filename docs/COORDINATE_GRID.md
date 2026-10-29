# Independent coordinate coding

`coordinate_depth` defaults to 16 and must be in [1, 20]. Coordinate coding is
always independent of `octree_depth`, the trained positional MLP input depth.
All geometry modes use checkpoint v6 and archive v10. Historical formats and
zero-depth coordinate coding are unsupported; no migration is performed.

`coordinate_depth=18, coordinate_tail_fraction=0` uses a single global depth-18 tree.
`coordinate_depth=16, coordinate_tail_fraction=0.001` uses two depth-16 trees: the
per-axis 0.1%–99.9% quantile box and a global box for points outside that box.
Quantiles use all learned point coordinates at freezing, without GT or test metrics.
Exterior points are retained, subject to the same per-region collision policy as
interior points. An empty region has no occupancy stream. Region membership is
implicit in archive order, without storing an integer label per point.

Coordinate cells/regions are distinct from `cells`, which continues to supply the
positional network at the original `octree_depth` and original normalization bounds.
After snapping, positional cells are computed from reconstructed coordinates under
those original bounds; local-grid snapping can cross original positional cells.
Hash normalization bounds also remain unchanged. Positional network weights are
never resized. The last training stage adapts appearance and other attributes to
the snapped positions and deduplication.

Every region has one representative per occupied cell. `--coordinate-selection
opacity` selects using all training-view opacity times projection visibility, without
reading GT; `first` uses original index order. This is not an occlusion-aware score.
Topology changes migrate parameters, optimizer moments, and VQ assignment caches.
Region buffers are also reindexed during topology changes and checkpoint restoration.

Version-10 ZIPs contain occupancy/Huffman coordinate streams, binary spatial hash,
network weights, arithmetic-coded features, ECVQ codebooks and rANS-coded geometry,
plus all coordinate/normalization bounds and stream metadata. Independent decoding
requires the code/runtime and archive, without a training checkpoint or source scene.
Source images are used only for quality evaluation, after decoding.

## Matched full-pipeline experiment

From the repository root:

```bash
.env/bin/python scripts/prepare_coordinate_experiment.py \
  --checkpoint <current-v6-checkpoint-at-step-30000> \
  --output outputs/tandt/train/coordinate_full_pipeline_v1
.env/bin/python scripts/run_coordinate_pipeline.py \
  --experiment outputs/tandt/train/coordinate_full_pipeline_v1
```

Use a new output directory. Preparation restores the same source optimizer and RNG
state for each branch, scores all training views, freezes and deduplicates, and saves
continuation checkpoints before immediate evaluation. Both branches continue through
exactly step 35000, encode complete archives, independently decode in a separate
process, validate rendering against export decisions on every test view, and evaluate
all 38 test views from the compressed archive. Source snapshots, hashes, exact commands,
per-view metrics and byte counts are retained in the experiment directory.

This is a single-scene exploratory comparison. Smaller snapping error alone does not
establish final compressed quality or bitrate; report the final decoded metrics and
the complete ZIP size. The earlier global16 opacity branch is a historical reference,
not a new rerun under the current encoder.

Tests:

```bash
PYTHONPATH=tests:. .env/bin/python -m unittest \
  test_core test_coordinate_selection test_ecvq_codec test_coordinate_grid
```
