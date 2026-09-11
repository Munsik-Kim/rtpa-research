"""Resolve bundled public evidence without a researcher-specific filesystem."""
from pathlib import Path


def is_evidence_root(path):
    path = Path(path)
    return (path / "configs/model_manifest.json").is_file() and (
        path / "data/masks/v05.json"
    ).is_file() and (path / "results/file_manifest.json").is_file()


def evidence_root(root=None):
    """Use an explicit checkout, the source checkout, or wheel-bundled evidence.

    Never searches arbitrary working directories, original absolute paths, model
    caches, network services, or private Drive locations.
    """
    if root is not None:
        selected = Path(root).expanduser().resolve()
        if not is_evidence_root(selected):
            raise FileNotFoundError(f"Not an RTPA evidence root: {selected}")
        return selected
    package = Path(__file__).resolve().parent
    for candidate in (package.parent.parent, package / "_evidence"):
        if is_evidence_root(candidate):
            return candidate
    raise FileNotFoundError(
        "RTPA evidence is unavailable. Install the project wheel, or supply "
        "--root /path/to/rtpa-research to use a public checkout."
    )
