# Minimal Native-boundary fixture

This model-free fixture reproduces one numerical boundary observed while
checking RTPA's recurrent-state reference. It contains one 128×128 FP32 head
slice plus its key/value vectors for one token and layer—no prompt text, token
IDs, model weights, full 16-head recurrent state, or user data. The complete
head shape is retained because PyTorch does not guarantee that reducing a
slice follows the same finite-arithmetic path as reducing the batched matrix.

On the recorded PyTorch 2.11.0 + CUDA 12.8 + RTX 5080 environment, the same
stored FP32 products reduce to neighboring values on CPU and CUDA. The one-ULP
projection difference crosses an adjacent BF16 midpoint after cancellation,
so the persisted state takes different BF16 values.

This demonstrates a scoped finite-arithmetic boundary. It is not evidence of
an upstream bug, universal hardware behavior, codec quality, or whole-model
equivalence. The parent experiment and full tensor remain local; the manifest
records their hashes and the narrower provenance of this derived fixture.

Run the CPU check with a compatible PyTorch installation:

```bash
python native_boundary_demo.py --device cpu
```

Run the recorded CUDA comparison on compatible hardware:

```bash
python native_boundary_demo.py --device both
```

Add `--out receipt.json` to preserve the JSON result. The script refuses to
overwrite an existing receipt.

The script verifies the fixture hash and schema before computing, and checks
that CPU and CUDA produce the same elementwise-product bytes before the
reduction diverges. Results on
other PyTorch, CUDA, CPU, or GPU combinations may differ; a mismatch is an
environment observation, not automatically a software defect.

The numeric fixture is derived from Qwen/Qwen3.5-0.8B-Base revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, licensed under Apache-2.0.
Qwen is credited as the source model; no endorsement is implied. Intended
integration is into the Apache-2.0 `Munsik-Kim/rtpa-research` repository with
its existing LICENSE, NOTICE, and third-party attribution.
