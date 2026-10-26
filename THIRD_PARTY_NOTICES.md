# Sources and attribution

This is an independent implementation of the supplied Smol-GS manuscript, not code
released by the paper's authors. The original Inria research license is preserved
in `LICENSE.md`; individual vendor copyright notices remain in their files.

* Inria 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
  Local revision `54c035f7834b564019656c3e3fcc3646292f727d` supplies
  `scene/colmap_loader.py`. The camera and rendering conventions follow Inria.
* HAC: https://github.com/YihangChen-ee/HAC
  Local revision `110dcf0a2d1603f65f94827fbdc58b9c9c3fcfbb` supplies
  `submodules/{diff-gaussian-rasterization,simple-knn,gridencoder,arithmetic}` and
  `utils/encodings_cuda.py`. These are source copies, excluding cached binaries,
  build directories and training outputs. HAC's hash-grid CUDA ABI is used directly.
* The embedded GLM sources retain their MIT license in
  `submodules/diff-gaussian-rasterization/third_party/glm/copying.txt`.
* Scaffold-GS, HAC++, LocoGS were inspected for project layout, argument names,
  learning-rate conventions and experimental launchers. Their model-specific
  anchors, offsets, masks and losses are not part of the Smol-GS model.
* Authors' official Smol-GS repository:
  https://github.com/AaltoML/Smol-GS
  `official_snapshot/` records revision
  `3270c9d0a9807566d173bf9d1bc295aee9b31764`, retrieved 2026-10-21.
  Its README says `Codes to be released.` Its CSVs are published reference results,
  not measurements from this implementation.

`docs/vendor_manifest.json` records SHA-256 hashes of the copied source files.
It is a provenance record, not a license replacement.
