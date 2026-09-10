from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path

import torch

from .model_adapter import text_model


def _cpu(tensor):
    return tensor.detach().to("cpu") if isinstance(tensor, torch.Tensor) else tensor


class GDNTraceCapture(AbstractContextManager):
    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.records: dict[int, dict] = {index: {} for index in layer_indices}
        self._restores = []
        self._handles = []

    def __enter__(self):
        layers = text_model(self.model).layers
        for index in self.layer_indices:
            module = layers[index].linear_attn
            original = module.chunk_gated_delta_rule
            record = self.records[index]

            def wrapped(query, key, value, *args, _original=original, _record=record, **kwargs):
                result = _original(query, key, value, *args, **kwargs)
                _record["query"] = _cpu(query)
                _record["key"] = _cpu(key)
                _record["value"] = _cpu(value)
                _record["g"] = _cpu(kwargs["g"])
                _record["beta"] = _cpu(kwargs["beta"])
                _record["initial_state"] = _cpu(kwargs.get("initial_state"))
                _record["native_core_output"] = _cpu(result[0])
                _record["native_final_state"] = _cpu(result[1])
                return result

            module.chunk_gated_delta_rule = wrapped
            self._restores.append((module, original))

            def pre_hook(_module, args, kwargs, _record=record):
                hidden = args[0] if args else kwargs["hidden_states"]
                _record["hidden_input"] = _cpu(hidden)

            def post_hook(_module, _args, _kwargs, output, _record=record):
                _record["native_layer_output"] = _cpu(output)

            self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, original in self._restores:
            module.chunk_gated_delta_rule = original
        for handle in self._handles:
            handle.remove()
        return False


def trace_text_sequence(model, input_ids: torch.Tensor, layer_indices: list[int]) -> tuple[dict, torch.Tensor]:
    language_model = text_model(model)
    with torch.inference_mode(), GDNTraceCapture(model, layer_indices) as capture:
        output = language_model(input_ids=input_ids, use_cache=True, return_dict=True)
    return capture.records, output.last_hidden_state.detach().cpu()


def save_trace(path: Path, record: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "trace": record}, path)
