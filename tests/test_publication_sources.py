"""Publication-only initializer proof cannot hide other source-byte changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from rtpa_research.publication_sources import check_frozen_publication_source


ROOT = Path(__file__).resolve().parents[1]
TARGET = 'src/rtpa_research/__init__.py'
ARCHIVE = 'configs/publication_sources/__init__.py.txt'
BINDING = 'configs/publication_sources/version_only_binding.json'
OLD_SHA = '586553eed9a87aed61252d3e7e1228e37489bdb2e0b21ae1ddc10d41ced4f5fa'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    for name in (TARGET, ARCHIVE, BINDING, 'pyproject.toml'):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())


class PublicationSourceTests(unittest.TestCase):
    def test_exact_mapping_keeps_original_expectation_and_nonmatching_byte_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            result = check_frozen_publication_source(root, TARGET, OLD_SHA)
            self.assertEqual(result['status'], 'PUBLICATION_VERSION_ONLY_MAPPING_VERIFIED')
            self.assertEqual(result['expected_sha256'], OLD_SHA)
            self.assertEqual(result['archived_original_sha256'], OLD_SHA)
            self.assertFalse(result['matches_frozen_bytes'])
            self.assertFalse(result['historical_expected_hash_modified'])
            self.assertTrue(result['all_other_bytes_identical'])
            self.assertFalse(result['source_executed'])

    def test_any_other_published_byte_change_fails_even_if_executable_meaning_matches(self):
        for suffix in (b'\n', b'# publication note\n', b'pass\n'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root)
                source = root / TARGET
                source.write_bytes(source.read_bytes() + suffix)
                with self.assertRaisesRegex(ValueError, 'NOT_EXACT_DECLARED_VERSION'):
                    check_frozen_publication_source(root, TARGET, OLD_SHA)

    def test_archive_and_pyproject_changes_fail(self):
        for name in (ARCHIVE, 'pyproject.toml'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root)
                path = root / name
                path.write_bytes(path.read_bytes().replace(b'1.1.0rc1', b'0.0.0rc1')
                                 if name == ARCHIVE else path.read_bytes().replace(b'1.3.0rc1', b'1.4.0rc1'))
                with self.assertRaisesRegex(ValueError, 'ARCHIVED_INITIALIZER|PROJECT_VERSION'):
                    check_frozen_publication_source(root, TARGET, OLD_SHA)

    def test_wrong_authority_or_other_source_has_no_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with self.assertRaisesRegex(ValueError, 'FROZEN_SOURCE_SHA256_MISMATCH'):
                check_frozen_publication_source(root, TARGET, '0' * 64)
            other = root / 'numerical.py'
            other.write_bytes((root / TARGET).read_bytes())
            with self.assertRaisesRegex(ValueError, 'FROZEN_SOURCE_SHA256_MISMATCH'):
                check_frozen_publication_source(root, 'numerical.py', OLD_SHA)
            exact = check_frozen_publication_source(root, 'numerical.py', digest(other))
            self.assertEqual(exact['status'], 'EXACT_FROZEN_BYTES')
            self.assertTrue(exact['matches_frozen_bytes'])

    def test_binding_cannot_redeclare_source_or_silently_repeat_keys(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root)
                path = root / BINDING
                if duplicate:
                    path.write_text(path.read_text().replace('{', '{"schema":"duplicate",', 1))
                else:
                    value = json.loads(path.read_text())
                    value['original_sha256'] = '0' * 64
                    path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'BINDING'):
                    check_frozen_publication_source(root, TARGET, OLD_SHA)

    def test_install_resolution_must_equal_current_bundled_bytes_before_mapping(self):
        spec = importlib.util.spec_from_file_location('r2_install_publication_test', ROOT / 'scripts/check_diag_r2_install.py')
        install = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(install)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            proof = install.source_identity(root, TARGET, OLD_SHA, digest(root / TARGET))
            self.assertEqual(proof['status'], 'PUBLICATION_VERSION_ONLY_MAPPING_VERIFIED')
            # Even the original expected bytes cannot substitute for the actual bundle.
            with self.assertRaisesRegex(ValueError, 'RESOLVED_BUNDLED_SOURCE_MISMATCH'):
                install.source_identity(root, TARGET, OLD_SHA, OLD_SHA)


if __name__ == '__main__':
    unittest.main()
