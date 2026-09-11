"""Model-free resource and command boundary checks for source and wheel use."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from rtpa_research.resources import evidence_root, is_evidence_root


class InstallationContract(unittest.TestCase):
    def test_default_evidence_exists(self):
        root = evidence_root()
        self.assertTrue(is_evidence_root(root))
        self.assertTrue((root / "data/tokenizer/tokenizer.json").is_file())
        self.assertTrue((root / "src/rtpa_research/frozen").is_dir())

    def test_missing_explicit_root_is_not_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                evidence_root(directory)

    def test_resource_import_needs_no_model_library(self):
        command = [sys.executable, "-c", "import sys; from rtpa_research.resources import evidence_root; assert evidence_root(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"]
        p = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"))
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_legacy_cli_help(self):
        p = subprocess.run([sys.executable, "-m", "rtpa_research", "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("verify", p.stdout)
        self.assertIn("--root", p.stdout)


if __name__ == "__main__":
    unittest.main()
