import hashlib,json,unittest,subprocess,sys
import numpy as np
from rtpa_research.resources import evidence_root
from rtpa_research.single_write_evidence import reconstruct

class Integration(unittest.TestCase):
    def test_included_observations(self):
        r=reconstruct(evidence_root())
        self.assertEqual(r['DEV_totals'],{'NATIVE':94,'ENERGY_PROMOTION':95,'B2_QUERY_PROMOTION':93})
        self.assertEqual(r['native_boundary_receipt_counts']['GPU_readout_bitwise_equal'],18)
    def test_fixture_and_lazy_commands(self):
        p=evidence_root()/'examples/native_boundary';m=json.loads((p/'fixture_manifest.json').read_text())
        raw=(p/'native_boundary_fixture.npz').read_bytes()
        self.assertEqual(len(raw),m['fixture_bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),m['fixture_sha256'])
        with np.load(p/'native_boundary_fixture.npz',allow_pickle=False) as z:
            self.assertEqual(z['initial_state_head'].shape,(128,128));self.assertTrue(np.isfinite(z['initial_state_head']).all())
        for cmd in ([],['native-boundary','--help'],['single-write-evidence','--help']):
            code='import sys\nfrom rtpa_research.cli import main\ntry:main('+repr(cmd)+')\nexcept SystemExit as e:assert e.code==0\nassert "torch" not in sys.modules\nassert "transformers" not in sys.modules'
            subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
