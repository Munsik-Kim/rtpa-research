# Same-byte codec revision R2

This is a **new numerical contract**, not a repair retroactively applied to
P_PRE/P_STORE. Historical implementations, masks, receipts, failures, and
expectations remain unchanged. The two candidates below are fixed before CAL
selection. This document does not select a winner or claim universal safety.

## Common representation and arithmetic

State has shape `[heads, key_rows, value_columns]`. Each low key row is split
into contiguous groups of 32 on the **value axis**. A normalized, deterministic
Sylvester H32 is applied on the right, in FP32. High rows are stored in their
**original coordinates**, without H32, as FP16 values. The next-token persistent
information is only the integer/FP16 payload; FP32 decode/transform/update
buffers are temporary. Immutable H32 and row indices are separate static data.

Each low group stores 32 UINT8 codes plus two FP16 numbers. There is no
FP32 persistent metadata, separate correction state, or hidden master state.
For 128x128 per head and high8:

`120 * (128 + 4 * (2 + 2)) + 8 * 128 * 2 = 19,328 bytes/head`

Uniform-low is 18,432 bytes/head. High-only is 32,768 bytes/head. These are
payload bytes, not allocated GPU memory, mask bytes, or allocator scratch.

FP32 arithmetic is used to create metadata, form code coordinates, and decode.
FP16 casts use IEEE round-to-nearest, ties-to-even (RNE). `floor16(x)` means
cast to FP16, then take the preceding FP16 value if the cast rounded upward.
`ceil16(x)` analogously takes the following FP16 value if the cast rounded
downward. Both are implemented with FP16 `nextafter`. Subnormals are retained.
The minimum stored nonzero scale is `s_min = 2^-24`, not a normal-only floor.
Reduction is min/max, not an approximate sampled range. Tensor division by
255, subtraction, addition and multiplication are separate FP32 operations.
No fused expression or compiler reordering is part of this reference contract.

`q = uint8(clamp(RNE(coordinate), 0, 255))` in both candidates. This is the
canonical nearest code on the declared **stored-metadata** grid using the
specified FP32 code-coordinate calculation; it is not P_PRE's pre-cast grid.
UINT8 saturation is an explicit quantizer operation, not a repair of invalid
inputs. Endpoint occupancy is not the same as saturation. Floating operation
rounding may affect exact halfway cases; cross-backend parity must be measured,
not inferred from a real-arithmetic identity.

## Candidate A: `R2_OFFSET`

For a transformed group `x`, compute:

```
lo = min(x); hi = max(x)
o16 = floor16(lo)
raw_s = (hi - float32(o16)) / 255
s16 = ceil16(max(raw_s, 2^-24))
coordinate = (x - float32(o16)) / float32(s16)
q = uint8(clamp(RNE(coordinate), 0, 255))
x_hat = float32(s16) * float32(q) + float32(o16)
```

Decode is FP32 multiplication followed by FP32 addition. The keys are
`low_codes`, `low_scales`, **`low_offsets`**, and optional `high_values`.
The physical offset is not a dimensionless zero point. Adjusting the range
*after* offset storage avoids assuming that rounding the original minimum
does not change the range. For exact-zero groups, use `(s16,o16,q)=(1,0,0)`.
Nonzero constant groups use the same general formula: their physical offset
is rounded downward and their nonnegative residual determines the scale.
They are not assigned a separate off-grid constant decode exception.

## Candidate B: `R2_ZERO_INCLUSIVE`

```
lo = min(min(x), 0); hi = max(max(x), 0)
raw_s = (hi - lo) / 255
s16 = ceil16(max(raw_s, 2^-24))
z16 = float16(RNE(-lo / float32(s16)))
coordinate = x / float32(s16) + float32(z16)
q = uint8(clamp(RNE(coordinate), 0, 255))
x_hat = float32(s16) * (float32(q) - float32(z16))
```

The zero point is an integer in `[0,255]`, represented exactly in FP16.
`RNE(x/s + z)` is intentional; it is **not** `RNE(x/s) + z`, whose halfway
behavior can differ. Decode is FP32 subtraction followed by multiplication.
The keys are `low_codes`, `low_scales`, **`low_zeros`**, and optional
`high_values`. Exact-zero groups use `(s16,z16,q)=(1,0,0)`. A nonzero constant
group includes zero in its range and uses the general formula. This can lose
substantial same-sign-group resolution and is a candidate to measure, not a
free numerical fix. Rounding the zero point can shift the endpoint grid by
up to approximately half a step; the declared code saturation remains active.

## Supported range and explicit failure

Before creating a returned payload, every transformed low value must be FP32,
finite and satisfy `abs(x) <= 65,504`. Every original-coordinate high value must
separately be finite with `abs(z_high) <= 65,504`. Testing only the untransformed
low input is insufficient: H32 changes entry magnitudes. These are conservative
supported ranges, not a claim that every FP32 input is representable.

For accepted low groups, the scale is positive FP16 and bounded by the ceiling
of `131008/255`, well below 65,504. Candidate A's offset is finite FP16, including
at both extrema. Candidate B's stored zero is an integer in `[0,255]`. Decoding
and inverse H32 use FP32; the decoded value need not itself remain within the
*input* bound at every component. A later recurrence write is checked anew.

Invalid input raises `CodecRangeError` with its boundary before the caller can
commit the returned payload. NaN/Inf replacement, input clipping, precision
promotion, metadata widening, or silent fallback are forbidden. Bounds are
mandatory even when a backend removes observation counters. Invalid externally
supplied payload dtype/shape/fields also raise, and floating metadata/high values
are checked before decode. UINT8 codes need no redundant finite test.

One reduction/synchronization is used per guard, never a host `.item()` for each
group. `guard_input` and `guard_payload` are reusable. Internal
`_affine_groups_unchecked` and `_decode_groups_unchecked` contain arithmetic only;
an optimized caller must perform equivalent guards before payload commit/use.
`CodecR2.assert_finite()` has no delayed counter: public encode/decode methods
already validate and raise before returning. A supplied shared H32 tensor is
trusted immutable policy data, not request state.

## Independent CPU validation and limits

`numpy_groups_reference` is a separate scalar-per-group NumPy implementation,
with explicit FP32 intermediates, FP16 directional conversion, RNE and decode.
It does not call the Torch encoder, transform, or decoder to produce expected
values. Common transformed groups are compared exactly; an independent binary
parity H32 construction checks axes and normalization. Small H32 floating
reduction differences are separated from same-group codec parity.

The three preserved failure fixtures are tested without modifying their bytes:
the historical v0.5 overflow group and the all18 stored-nearest/FA_CODE heads.
Historical overflow is rechecked separately from R2 finite decode and error.
Zero/constant, narrow same-sign, mixed-sign, subnormal scale, halfway rounding,
FP16 input/high limits, invalid metadata, payload bytes, and protected rows are
tested. CPU fixture success is not a CUDA parity claim, full CAL completion,
universal future-input safety, or end-to-end quality evidence. CAL selection,
all18 replay, policy freeze, and TEST belong to the separate run protocol.
