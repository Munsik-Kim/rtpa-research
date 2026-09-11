# Architecture support and source fidelity

RTPA separates a **trained-model storage intervention** from an **equation-level operator experiment**. A working recurrent kernel, a model card, and a measured pretrained model are different levels of evidence. Quality and cost belong to the exact checkpoint, layer scope, codec, and run in [Results](RESULTS.md).

| Family | What is verified | Scope and boundary |
|---|---|---|
| GDN | Model-backed Qwen3.5-0.8B-Base adapter; installed native recurrence inspected and checked | The model has 18 GDN layers. New comparison A evaluates allocation at all 18; comparisons B/C evaluate inherited FA_CODE at layers 0/12/22. Historical and new panels remain separate. |
| GDN2 | **OPERATOR_TESTED**: independently authored forward/adjoint equations and synthetic checks | No author-linked trained checkpoint with a verified local cached-inference path was established. Pretrained-model KL/NLL is **NOT_RUN**, not zero. No claim is made that GDN2 cannot fit 16 GB. |

The [machine-readable source audit](../data/evidence/upgrade_sources/source_audit.json) records the 2026-09-11 searches, versions, byte counts, SHA-256 hashes, and CPU probe results. This is a bounded review, not proof that no other checkpoint or implementation exists.

## GDN: the actual Qwen path

The checkpoint is [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68), revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Its GDN layer indices are:

```text
0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22
```

Each has 16 recurrent heads with key-by-value state `128 × 128`. The other six layers use full attention. Weights and the Native recurrent cache are BF16; the native recurrent update is FP32. Native is therefore **not an all-FP32 oracle**. The SDPA attention path, attention KV cache, convolution cache, weights, and activations are not the compressed state payload.

### Frozen method scopes

The new protocol preserves two different comparisons, fixed before TEST:

| Contrast | Methods | Target layers and offline policy |
|---|---|---|
| A: allocation | MATCHED_ENERGY, RTPA_DIAG, DAMP_PAPER_ADAPTED | All 18 GDN layers; new synthetic TRAIN6 masks |
| B: integer-code correction | STORED_NEAREST, FA_CODE_FACTORIZED | Layers 0/12/22; inherited v0.5 MATCHED mask and TRAIN12 FA metric `m2_l1_e005` |
| C: execution representation | FA_CODE_REFERENCE, FA_CODE_FACTORIZED | Same inherited three-layer policy; reference retained for conformance/timing |

The all-18-layer stored-nearest and FA paths failed during the pre-TEST CAL applicability probe. Their [actual failure fixtures](../data/evidence/upgrade_failures/README.md) are retained. B/C therefore use the **pre-existing three-layer policy**, not a new layer subset selected for favorable TEST results. Its masks are exactly the included v0.5 MATCHED_ENERGY8 masks (original TRAIN9); its FA metric was fitted on the separate historical TRAIN12. The new all-layer FA metric is not used for B/C. The allocation paths completed 64 CAL tokens, which is not a full-length or TEST stability guarantee.

Mixed payload remains 19,328 bytes/head within each matched comparison. That does **not** make the all-layer allocation paths and three-layer code-correction paths equal-total-storage controls. Do not interpret their cross-scope KL or timing differences as isolated mask/encoder effects; report each against its registered baseline and layer scope. The source audit records both policy hashes and the method-layer mapping.

The inspected implementation is Transformers 5.9.0, `models/qwen3_5/modeling_qwen3_5.py`, SHA-256 `d67880b98f47d55a9d40679c901a740aa0618b97458824cab2abe9a1f7861be6`. The exact native arithmetic matters: q/k normalization precedes conversion to FP32, and query scaling uses multiplication by `1 / sqrt(128)`. Storage encoding occurs **after the current local readout**, so its perturbation first enters subsequent recurrent updates. Per-token prompt writes are a deliberate local runtime condition, not optimized chunk-prefill behavior.

### CPU boundary probe, not a model benchmark

Two synthetic CPU cases checked the installed Torch functions: a nonzero BF16 incoming state through the recurrent path, and a zero-start first-token chunk path. The independently reconstructed FP32 update matched the returned state exactly in both cases. The reconstructed `zᵀq` matched the native returned readout **after its BF16 cast**. Before casting, normalized differences were approximately `1.6704e-3` and `1.6577e-3`.

This distinguishes a calibration **FP32 pre-output-cast objective** from an actual BF16 returned readout. Changing reciprocal multiplication into division changed a BF16 rounding decision in an exploratory probe; that expression is not an interchangeable byte-level contract. The audit includes the executed probe program and results. It establishes two CPU tensor-point checks, not arbitrary-input, CUDA, or full-model equivalence.

## GDN2: different gates, nonsymmetric transition

The [GDN2 paper, v1](https://arxiv.org/html/2605.22791v1) directly links [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2). We inspected commit `a5552fe3c67e0ebc7ef1220df68ae8896ec62d56`, including the layer, recurrent kernel, model wrapper, and configuration.

For state `S ∈ R^(key × value)`, let `e = b ⊙ k` and `D = diag(α)`. The reviewed equation is:

```text
A       = (I − k eᵀ) D
S_new   = A S_old + k (w ⊙ v)ᵀ
output  = S_newᵀ q
Aᵀ x    = α ⊙ [x − (b ⊙ k)(kᵀx)]
```

The erase gate `b` and decay `α` act on key coordinates; the write gate `w` acts on value coordinates. In general **A is not symmetric**. The adjoint applies decay after the rank-one operation; a scalar-decay GDN shortcut is not valid for arbitrary GDN2 gates. Tying both gates to one scalar β gives the KDA form; additionally tying decay across key coordinates gives GDN.

The source supports key-by-value state and an optional transposed layout. Its recurrent accumulator/final state is FP32, with output cast to the value dtype before subsequent gated normalization/projection. The inspected layer chooses a recurrent path for nontraining short sequences and otherwise uses its chunk path. The training-oriented top-level model wrapper does not itself establish a verified cache-forwarding generation interface.

The local [operator implementation](../src/rtpa_research/operators.py) is independently authored from the equations. [CPU tests](../tests/test_operators.py) check dense versus structured updates, nonsymmetric adjoints, tied-gate reductions, source-axis propagation, and direct impulse versus Gramian SSE. A synthetic operator SSE or demonstration is **not pretrained language quality**, and these checks are not official Triton-kernel parity.

### Checkpoint and license boundary

No checkpoint link was found in the inspected official README; the checked official release/tag endpoints and NVIDIA GDN model query returned empty lists. A **third-party** model associated with the paper in Hugging Face metadata does exist: [LLM-OS-Models/gdn2-1.3B-fineweb-edu-100b](https://huggingface.co/LLM-OS-Models/gdn2-1.3B-fineweb-edu-100b/tree/86327354dd1e6874fe6407878087079326fee851), revision `86327354dd1e6874fe6407878087079326fee851`. Its card describes in-progress pretraining and links a different training repository. This is not an outgoing author-paper checkpoint link; author endorsement was not verified.

The checked file listing contains 19 `.pth` checkpoints through 95B training tokens, each reported as 17,401,727,659 bytes, without tokenizer files or a standalone configuration. These are **remote listing observations**: weights were neither downloaded nor loaded, and file size is not measured VRAM demand. The correct conclusion is limited model-port provenance and execution coverage, not “no GDN2 weights exist” or “16 GB is impossible.”

Official GDN2 code is under the [NVIDIA Source Code License-NC](https://github.com/NVlabs/GatedDeltaNet-2/blob/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56/LICENSE), which states non-commercial research/evaluation restrictions. **No official GDN2 source was vendored or relabeled under RTPA's Apache-2.0 license.** A third-party model card's license field does not override upstream code restrictions.

## DAMP: paper-adapted means paper-adapted

The current bounded review of [DAMP v1](https://arxiv.org/html/2608.27513v1), paper links, and exact-title/ID repository searches did not establish author-code provenance. The status remains **AUTHOR_CODE_NOT_VERIFIED**, not a claim that author code is unpublished. arXiv's GitHub feedback links are not algorithm repositories.

| Component | Paper contract | Local comparison |
|---|---|---|
| Codec | UINT8 affine; normalized value-axis H32; group32; FP16 metadata/high rows | Same broad format; explicit local P_PRE ordering and edge-case conventions |
| Protected budget | 16/128 rows; 9.875 bits/value | 8/128 rows; 9.4375 bits/value including metadata, excluding indices/static/scratch |
| Calibration | Four Pile validation domains × eight documents, 256 tokens, stride8, balanced | New synthetic TRAIN6 and native stride8 in this upgrade; historical calibration variants remain distinct |
| Risk | Low-only error energy × geometric-mean-decay persistence | Paper-adapted scalar-head calculation, not author-code execution |
| Runtime | FP32 recurrent reference; fused decode; chunk-boundary prefill writes | Native BF16 cache with FP32 recurrence; eager per-token storage writes |

For scalar-decay GDN and a fixed per-head budget, persistence is a common positive factor within a head, so it cannot change the error-energy top-k ordering. That identity does not transfer to arbitrary channel-wise GDN2 decay. Exact cast-before-code, tie, floor, constant-group, and overflow details are not completely established by the checked paper. Local choices and the retained FP16 metadata failure are not silently promoted to official DAMP behavior. See [Method](METHOD.md) and [Limitations](LIMITATIONS.md).

## Prior-art boundary: Qronos

[Qronos v3](https://arxiv.org/html/2505.11695v3) directly links its [Brevitas implementation](https://github.com/i-colbert/brevitas/tree/fa88103f1f2b7d04aac3eedc66a274dcfda1c1fa/src/brevitas_examples/llm). We inspected `src/brevitas/graph/qronos.py` at commit `fa88103f1f2b7d04aac3eedc66a274dcfda1c1fa` (BSD-3-Clause SPDX). Its offline weight-column rounding uses float/quantized activation covariances and least-squares error correction.

Output-aware reconstruction, data-dependent rounding, fixed grids, error correction, and error diffusion are prior principles—not RTPA novelty claims. FA_CODE's examined scope is bounded **runtime recurrent-write code correction** using a frozen state metric; RTPA-DIAG selects a **fixed precision mask offline**. We present recurrent-state implementation choices and measured evidence, not first-ever priority, SOTA, a Qronos reproduction, or a DAMP refutation. No Qronos benchmark was executed here.

## Reproduce the operator check

From an installed checkout, without model weights:

```bash
python -m rtpa_research demo
```

The NumPy demonstration is model-free. Torch-dependent equation/native probes require the recorded optional environment, but do not load a model. Source hashes identify reviewed bytes; numerical benchmark results, provenance, and verification levels remain separate.
