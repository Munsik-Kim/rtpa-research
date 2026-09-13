"""Small diagnostic arm adapter. Historical codecs and payloads are unchanged.

HOLD arms retain the first FP16 grid, not a CAL-learned deployment policy.
Instrumentation reads operands after the actual write; it never edits payloads.
"""
import torch
from .codec import Codec
from .codec_r2 import CodecR2, affine_groups, guard_input
from .layout import tensor_bytes

ARMS = ('LEGACY_P_PRE', 'R2_OFFSET', 'R2_ZERO_INCLUSIVE', 'R2_HOLD_INITIAL_GRID')
CONTROLS = ('R2_HOLD_OFFSET', 'R2_HOLD_SCALE')


class Arm:
    def __init__(self, name):
        if name not in ARMS + CONTROLS:
            raise ValueError(name)
        self.name = name
        self.codec = Codec('P_PRE', 'cpu') if name == ARMS[0] else CodecR2(
            'R2_ZERO_INCLUSIVE' if name == ARMS[2] else 'R2_OFFSET', 'cpu')
        self.grid = None

    def groups(self, x):
        """Affine control, using exactly the appropriate decoder convention."""
        from .codec import affine
        from .codec_r2 import decode_groups
        if self.name == ARMS[0]:
            q, s, m, _ = affine(x, 'P_PRE')
            p = dict(low_codes=q, low_scales=s, low_zeros=m)
            return p, s.float() * (q.float() - m.float())
        profile = self.codec.profile
        p = affine_groups(x, profile)
        if self.name.startswith('R2_HOLD'):
            if self.grid is None:
                self.grid = {k: p[k].clone() for k in ('low_scales', 'low_offsets')}
            if self.name != 'R2_HOLD_OFFSET':
                p['low_scales'] = self.grid['low_scales']
            if self.name != 'R2_HOLD_SCALE':
                p['low_offsets'] = self.grid['low_offsets']
            p['low_codes'] = torch.round((x - p['low_offsets'].float()) /
                                        p['low_scales'].float()).clamp(0, 255).to(torch.uint8)
        return p, decode_groups(p, profile)

    def write(self, z, layout):
        if self.name.startswith('R2_HOLD'):
            low = z.gather(1, layout.indices('low'))
            x = self.codec.transform(low).reshape(*low.shape[:-1], -1, 32)
            p, _ = self.groups(x)
            high = z.gather(1, layout.indices('high'))
            guard_input(high, boundary='original_high')
            p['high_values'] = high.half()
        else:
            p = self.codec.encode(z, layout)
        try:
            out = self.codec.decode(p, layout)
            self.codec.assert_finite()
            if not torch.isfinite(out).all():
                raise FloatingPointError('NONFINITE_DECODE')
        except Exception as exc:
            exc.grid_payload=p
            raise
        if tensor_bytes(p) != len(z) * 19328:
            raise ValueError('PAYLOAD_BYTES')
        return p, out

    def inspect(self, z, p, layout):
        low = z.gather(1, layout.indices('low'))
        x = self.codec.transform(low).reshape(*low.shape[:-1], -1, 32)
        if self.name == ARMS[0]:
            lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
            const = lo == hi
            s = (hi - lo) * float(torch.tensor(1 / 255, dtype=torch.float32))
            s = torch.where(const, torch.ones_like(s), s)
            s = torch.where((~const) & (s > 0) & (s.half() == 0), torch.full_like(s, 2**-24), s)
            m = torch.where(const, -lo, torch.round(-lo / s))
            rounded = torch.where(const, torch.zeros_like(x), torch.round(x / s) + m)
            decoded = p['low_scales'].float() * (p['low_codes'].float() - p['low_zeros'].float())
        elif self.name == ARMS[2]:
            rounded = torch.round(x / p['low_scales'].float() + p['low_zeros'].float())
            decoded = p['low_scales'].float() * (p['low_codes'].float() - p['low_zeros'].float())
        else:
            rounded = torch.round((x - p['low_offsets'].float()) / p['low_scales'].float())
            decoded = p['low_scales'].float() * p['low_codes'].float() + p['low_offsets'].float()
        return x, decoded, ((rounded < 0) | (rounded > 255)).sum((1, 2, 3))
