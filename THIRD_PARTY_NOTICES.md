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
- USTC-IMCL RDO-Gaussian: https://github.com/USTC-IMCL/RDO-Gaussian
  Local revision `a7bb99b0374efbe3e2a23c5cf3e9109fbf53baef` supplies the geometry
  ECVQ/K-means implementation adapted in `smolgs/vq/ecvq.py`. Original source
  paths, behavior and adaptations are listed in `docs/ECVQ.md`; frozen reference
  copies are no longer bundled. Its license matches `LICENSE.md`.
- RDO-Gaussian's imcl-compression/InterDigital entropy extensions supply
  `submodules/rdo-entropy/cpp_exts/{rans,ops,ryg_rans}`. Original notices are
  retained; Apache-2 and ryg_rans licenses are included. The rANS flush buffer
  reserves two extra words for the final 64-bit state (single-symbol fix).
  `docs/rdo_entropy_manifest.json` records original and adapted file hashes.
- Authors' Smol-GS repository: https://github.com/AaltoML/Smol-GS
  `official_snapshot/` records revision
  `3270c9d0a9807566d173bf9d1bc295aee9b31764`. CSVs in
  `official_snapshot/results/` are published reference results, not measurements
  from this code.

`docs/vendor_manifest.json` records the provenance and SHA-256 hashes of copied
third-party source files. It is not a license replacement.
