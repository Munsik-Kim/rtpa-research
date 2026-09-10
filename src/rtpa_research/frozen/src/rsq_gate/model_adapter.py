from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from .discovery import sha256


def load_model(model_path: Path, device: str = "cuda"):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer


def text_model(model):
    return model.model.language_model


def gdn_indices(model) -> list[int]:
    return [index for index, layer in enumerate(text_model(model).layers) if hasattr(layer, "linear_attn")]


def selected_layer_indices(indices: list[int]) -> list[int]:
    return [indices[0], indices[len(indices) // 2], indices[-1]]


def _revision(model_path: Path) -> str | None:
    metadata = model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    return metadata.read_text().splitlines()[0] if metadata.exists() else None


def build_model_manifest(model, model_path: Path) -> dict:
    language_model = text_model(model)
    config = language_model.config
    indices = gdn_indices(model)
    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    first_module = language_model.layers[indices[0]].linear_attn
    source = Path(inspect.getsourcefile(first_module.__class__) or "")
    recurrence_source = Path(inspect.getsourcefile(first_module.recurrent_gated_delta_rule) or "")
    weight = model_path / "model.safetensors-00001-of-00001.safetensors"
    return {
        "model_identifier": "Qwen/Qwen3.5-0.8B-Base",
        "actual_local_path": str(model_path.resolve()),
        "revision": _revision(model_path),
        "weight_sha256": sha256(weight),
        "dtype": str(next(model.parameters()).dtype),
        "parameter_count_total": sum(p.numel() for p in model.parameters()),
        "parameter_count_text_model": sum(p.numel() for p in language_model.parameters()),
        "num_layers": config.num_hidden_layers,
        "layer_types": list(config.layer_types),
        "gdn_layer_indices": indices,
        "selected_gdn_layer_indices": selected_layer_indices(indices),
        "full_attention_layer_indices": full,
        "num_recurrent_key_heads": config.linear_num_key_heads,
        "num_recurrent_value_heads": config.linear_num_value_heads,
        "key_dimension": config.linear_key_head_dim,
        "value_dimension": config.linear_value_head_dim,
        "conv_kernel_size": config.linear_conv_kernel_dim,
        "state_shape_per_layer": [
            "batch",
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ],
        "native_model_implementation_path": str(source),
        "actual_recurrence_function_path": str(recurrence_source),
        "recurrent_function": first_module.recurrent_gated_delta_rule.__name__,
        "chunk_function": first_module.chunk_gated_delta_rule.__name__,
        "fla_fast_path_active": first_module.recurrent_gated_delta_rule.__module__.startswith("fla."),
        "cache_class": "transformers.cache_utils.DynamicCache",
        "selected_model_is_gdn2": False,
    }


def write_model_reports(workspace: Path, manifest: dict) -> None:
    (workspace / "artifacts" / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = [
        ("Identifier", manifest["model_identifier"]),
        ("Local path", manifest["actual_local_path"]),
        ("Revision", manifest["revision"]),
        ("Dtype", manifest["dtype"]),
        ("Total/text parameters", f"{manifest['parameter_count_total']:,} / {manifest['parameter_count_text_model']:,}"),
        ("Layers", manifest["num_layers"]),
        ("GDN layers", manifest["gdn_layer_indices"]),
        ("Selected GDN layers", manifest["selected_gdn_layer_indices"]),
        ("Full-attention layers", manifest["full_attention_layer_indices"]),
        ("Heads / d_k / d_v", f"{manifest['num_recurrent_value_heads']} / {manifest['key_dimension']} / {manifest['value_dimension']}"),
        ("Native implementation", manifest["native_model_implementation_path"]),
        ("Recurrence implementation", manifest["actual_recurrence_function_path"]),
        ("FLA active", manifest["fla_fast_path_active"]),
    ]
    table = "\n".join(f"| {name} | `{value}` |" for name, value in rows)
    report = f"# Model Introspection\n\n| Field | Value |\n|---|---|\n{table}\n\n"
    report += "The downloaded checkpoint was selected only after bounded local cache discovery found no Qwen3.5 or other Gated DeltaNet pretrained LM. Existing Qwen2.5 checkpoints were rejected because their `config.json` files describe softmax-attention models, not recurrent GDN state.\n\n"
    report += "The selected model is Qwen3.5 Gated DeltaNet, not GDN2. E7 is therefore `NOT_APPLICABLE_TO_SELECTED_MODEL`.\n"
    (workspace / "reports" / "model_introspection.md").write_text(report)
