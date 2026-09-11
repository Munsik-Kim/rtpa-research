"""Payload row layout, relocated without numerical changes from RTPA v0.1."""
from dataclasses import dataclass
import torch

@dataclass
class Layout:
    low: torch.Tensor
    high: torch.Tensor
    keys: int
    values: int

    @classmethod
    def from_mask(cls, mask, values=128):
        mask = mask.bool()
        counts = mask.sum(-1)
        if not bool((counts == counts[0]).all()):
            raise ValueError('each head must have the same number of high rows')
        ids = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
        return cls(ids[~mask].reshape(len(mask), -1).clone(),
                   ids[mask].reshape(len(mask), -1).clone(), mask.shape[-1], values)

    def indices(self, which):
        return getattr(self, which).unsqueeze(-1).expand(-1, -1, self.values)



def tensor_bytes(payload):
    return sum(t.numel() * t.element_size() for t in payload.values())
