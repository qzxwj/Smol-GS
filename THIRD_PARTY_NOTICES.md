# Sources and Attribution

This is an independent implementation of the supplied Smol-GS manuscript. The
Inria research license is preserved in `LICENSE.md`; vendor copyright and
license notices remain in their source files.

- Inria 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
  Revision `54c035f7834b564019656c3e3fcc3646292f727d` supplies
  `scene/colmap_loader.py`. Camera and rendering conventions follow Inria.
- HAC: https://github.com/YihangChen-ee/HAC
  Revision `110dcf0a2d1603f65f94827fbdc58b9c9c3fcfbb` supplies
  `submodules/{diff-gaussian-rasterization,simple-knn,gridencoder,arithmetic}`
  and `utils/encodings_cuda.py`. The binary mixed hash-grid backend and
  arithmetic codec are retained.
- Embedded GLM sources retain their MIT license in
  `submodules/diff-gaussian-rasterization/third_party/glm/copying.txt`.
- NVlabs tiny-cuda-nn v2.0 is an optional external dependency under its
  BSD-3-Clause license. The installation script pins revision
  `3c7931caab7b691c6f30dffe20952ed727d935b9`. Current models use its native
  FullyFusedMLP; the entropy encoder uses HAC binary mixed grids.
- Scaffold-GS, HAC++, and LocoGS were consulted for conventions and comparisons.
  Their anchor/offset models are not part of this implementation.
- Authors' Smol-GS repository: https://github.com/AaltoML/Smol-GS
  `official_snapshot/` records revision
  `3270c9d0a9807566d173bf9d1bc295aee9b31764`. Its CSVs and the copies in
  `results/` are published reference results, not measurements from this code.

`docs/vendor_manifest.json` records the provenance and SHA-256 hashes of copied
third-party source files. It is not a license replacement.
