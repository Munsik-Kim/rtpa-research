"""Include the curated repository evidence in wheels without duplicating Git data.

The installed CPU verifier checks historical paths and source receipts. Therefore
the wheel carries the same evidence-root layout, including the frozen numerical
source, alongside the importable package. No research cache or model is copied.
"""
from pathlib import Path
import shutil
from setuptools.command.build_py import build_py


ROOT_FILES = (
    ".gitattributes", ".gitignore", "CITATION.cff", "LICENSE", "NOTICE",
    "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
)
TREE_ROOTS = ("configs", "data", "docs", "results", "scripts", "examples", "tests", "src", ".github")


def evidence_files(root):
    """Return only explicit publication roots; never cwd caches or Git history."""
    for name in ROOT_FILES:
        path = root / name
        if path.is_file():
            yield path
    for name in TREE_ROOTS:
        tree = root / name
        for path in sorted(tree.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in {"__pycache__", "_evidence", ".git", ".pytest_cache"}
                   or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


class BuildEvidence(build_py):
    def get_source_files(self):
        root = Path(__file__).resolve().parents[2]
        sources = super().get_source_files()
        sources.extend(str(path.relative_to(root)) for path in evidence_files(root))
        return sorted(set(sources))

    def run(self):
        super().run()
        root = Path(__file__).resolve().parents[2]
        destination = Path(self.build_lib) / "rtpa_research" / "_evidence"
        # A repeated local build must not retain evidence removed from the tree.
        # This is a build output owned by this command, not a source directory.
        if destination.exists():
            shutil.rmtree(destination)
        for source in evidence_files(root):
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        root = Path(__file__).resolve().parents[2]
        destination = Path(self.build_lib) / "rtpa_research" / "_evidence"
        outputs.extend(str(destination / source.relative_to(root))
                       for source in evidence_files(root))
        return outputs
