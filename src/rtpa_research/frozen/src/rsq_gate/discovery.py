from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _command(args: list[str], cwd: Path) -> dict:
    completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "argv": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_environment(workspace: Path) -> dict:
    conda = Path("__USER_HOME__/miniforge3/bin/conda")
    commands = {
        "nvidia_smi": _command(["nvidia-smi"], workspace),
        "python_version": _command([sys.executable, "--version"], workspace),
        "which_python": _command(["which", "python"], workspace),
        "conda_env_list": _command([str(conda), "env", "list"], workspace),
        "pip_freeze": _command([sys.executable, "-m", "pip", "freeze"], workspace),
        "git_status": _command(["git", "status", "--short", "--branch"], workspace),
        "git_head": _command(["git", "rev-parse", "HEAD"], workspace),
    }
    cuda_available = torch.cuda.is_available()
    manifest_path = workspace / "configs" / "gate_manifest.json"
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        },
        "gate_manifest_sha256": sha256(manifest_path),
        "commands": commands,
        "package_installations_performed": [],
    }
    (workspace / "artifacts" / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")
    nvidia = commands["nvidia_smi"]
    git_note = (
        commands["git_head"]["stdout"]
        if commands["git_head"]["returncode"] == 0
        else "NOT_A_GIT_REPOSITORY"
    )
    markdown = "# Environment\n\n"
    markdown += f"Captured (UTC): `{payload['captured_at_utc']}`  \n"
    markdown += f"Python: `{payload['python_executable']}` (`{payload['python_version']}`)  \n"
    markdown += f"PyTorch: `{torch.__version__}`; torch CUDA: `{torch.version.cuda}`  \n"
    markdown += f"GPU: `{payload['torch']['device_name']}`; capability: `{payload['torch']['device_capability']}`  \n"
    markdown += f"Git HEAD: `{git_note}`  \n"
    markdown += f"Frozen manifest SHA-256: `{payload['gate_manifest_sha256']}`\n\n"
    markdown += "No packages were installed or upgraded for this experiment. The existing `attention` environment is reused.\n\n"
    markdown += "## nvidia-smi\n\n```text\n" + (nvidia["stdout"] or nvidia["stderr"]) + "\n```\n\n"
    markdown += "## Conda environments\n\n```text\n" + commands["conda_env_list"]["stdout"] + "\n```\n\n"
    markdown += "The complete `pip freeze`, command return codes, stderr, and raw Git probes are retained in `artifacts/environment.json`.\n"
    (workspace / "reports" / "environment.md").write_text(markdown)
    return payload


def discover_hf_cache() -> dict:
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    repos = []
    for repo in sorted(info.repos, key=lambda item: item.repo_id.lower()):
        repos.append(
            {
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "repo_path": str(repo.repo_path),
                "size_on_disk": repo.size_on_disk,
                "revisions": [revision.commit_hash for revision in repo.revisions],
            }
        )
    return {
        "size_on_disk": info.size_on_disk,
        "repos": repos,
        "warnings": [str(warning) for warning in info.warnings],
    }
