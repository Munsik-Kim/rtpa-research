"""CPU fixture and independent-reference audit; no model/CUDA work.

Run from a source checkout with PYTHONPATH=src. The report measures only saved
points and deterministic synthetic groups, not all18 CAL or model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import unittest

import numpy as np
import torch

from rtpa_research.codec import Codec, affine
from rtpa_research.codec_r2 import CodecR2, PROFILES, affine_groups, decode_groups, numpy_groups_reference
from rtpa_research.layout import Layout, tensor_bytes

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_error(actual, target):
    a, z = actual.astype(np.float64), target.astype(np.float64)
    error = a-z
    energy = float(np.square(z).sum(dtype=np.float64))
    sse = float(np.square(error).sum(dtype=np.float64))
    return {"SSE":sse,"input_energy":energy,"NMSE":sse/energy if energy>0 else None,
            "NMSE_reason":None if energy>0 else "ZERO_INPUT_ENERGY",
            "max_absolute_error":float(np.abs(error).max()),"decoded_finite":bool(np.isfinite(a).all())}


def saturation(groups,payload,profile):
    s=payload["low_scales"].float()
    if profile=="R2_OFFSET":coordinate=(groups-payload["low_offsets"].float())/s
    else:coordinate=groups/s+payload["low_zeros"].float()
    rounded=torch.round(coordinate)
    return {"rounded_code_outside_0_255":int(((rounded<0)|(rounded>255)).sum()),
            "endpoint_code_count":int(((payload["low_codes"]==0)|(payload["low_codes"]==255)).sum()),
            "code_count":groups.numel(),
            "scale_subnormal_count":int(((s>0)&(s<2**-14)).sum()),
            "scale_min":float(s.min()),"scale_max":float(s.max())}


def audit(out):
    started=time.monotonic()
    torch.set_num_threads(2)
    fixture_paths=[ROOT/"data/evidence/fixtures/overflow.npz",
                  ROOT/"data/evidence/upgrade_failures/stored_nearest.npz",
                  ROOT/"data/evidence/upgrade_failures/fa_code_factorized.npz"]
    fixture_hashes={str(p.relative_to(ROOT)):sha(p) for p in fixture_paths}
    source_paths=[ROOT/"src/rtpa_research/codec_r2.py",ROOT/"tests/test_codec_r2.py",
                  ROOT/"docs/CODEC_R2_CONTRACT.md",Path(__file__).resolve()]
    rows=[]
    for path in fixture_paths:
        with np.load(path,allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        if "group" in arrays:
            groups=torch.from_numpy(arrays["group"])
            for historical in ["P_PRE","P_STORE"]:
                old=affine(groups,historical)
                rows.append({"fixture":str(path.relative_to(ROOT)),"profile":historical,
                    "role":"HISTORICAL_EXPECTED_FAILURE","metadata_nonfinite":int((~torch.isfinite(old[2])).sum()),
                    "status":"EXPECTED_OVERFLOW_REPRODUCED" if not bool(torch.isfinite(old[2]).all()) else "FAIL_NOT_REPRODUCED"})
            z=None;layout=None
        else:
            z=torch.from_numpy(arrays["z_head"])[None]
            layout=Layout.from_mask(torch.from_numpy(arrays["high_mask"])[None])
            for historical in ["P_PRE","P_STORE"]:
                legacy=Codec(historical,"cpu");old=legacy.encode(z,layout)
                rows.append({"fixture":str(path.relative_to(ROOT)),"profile":historical,
                    "role":"HISTORICAL_EXPECTED_FAILURE","metadata_nonfinite":int((~torch.isfinite(old["low_zeros"])).sum()),
                    "status":"EXPECTED_OVERFLOW_REPRODUCED" if not bool(torch.isfinite(old["low_zeros"]).all()) else "FAIL_NOT_REPRODUCED"})
        for profile in PROFILES:
            codec=CodecR2(profile)
            if z is None:
                p=affine_groups(groups,profile);decoded=decode_groups(p,profile)
                observed=decoded.numpy();target=groups.numpy()
            else:
                p=codec.encode(z,layout);decoded=codec.decode(p,layout)
                groups=codec.transform(z.gather(1,layout.indices("low"))).reshape(1,120,4,32)
                observed=decoded.numpy();target=z.numpy()
            expected=numpy_groups_reference(groups.numpy(),profile)
            exact=all(np.array_equal(p[k].numpy(),expected[k]) for k in p if k!="high_values")
            same_group_decode=decode_groups({k:v for k,v in p.items() if k!="high_values"},profile)
            decode_exact=np.array_equal(same_group_decode.numpy(),expected["decoded_groups"])
            rows.append({"fixture":str(path.relative_to(ROOT)),"profile":profile,
                "role":"NEW_CONTRACT_AT_SAVED_POINT","independent_group_payload_exact":exact,
                "independent_group_decode_exact":decode_exact,"payload_bytes":tensor_bytes(p),
                "payload_scope":"one128x128head_high8" if z is not None else "one_transformed_group32",
                **summarize_error(observed,target),**saturation(groups,p,profile),
                "status":"PASS_AT_INCLUDED_POINT" if exact and decode_exact and np.isfinite(observed).all() else "FAIL"})
    logs=io.StringIO()
    result=unittest.TextTestRunner(stream=logs,verbosity=2).run(
        unittest.TestLoader().discover(str(ROOT/"tests"),pattern="test_codec_r2.py"))
    unchanged=all(sha(ROOT/p)==h for p,h in fixture_hashes.items())
    passed=result.wasSuccessful() and unchanged and all(not r["status"].startswith("FAIL") for r in rows)
    report={"status":"PASS_CPU_CONTRACT_AT_TESTED_POINTS" if passed else "FAIL_CPU_CONTRACT",
        "profiles":list(PROFILES),"execution":"CPU_TORCH_AND_INDEPENDENT_NUMPY;NO_MODEL;NO_GPU",
        "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
        "cpu_threads":torch.get_num_threads(),"cuda_initialized":torch.cuda.is_initialized(),
        "source_manifest":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in source_paths],
        "fixture_initial_hashes":fixture_hashes,"fixtures_unchanged":unchanged,"rows":rows,
        "tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped)},
        "unit_log":logs.getvalue(),"elapsed_cpu_wall_seconds":time.monotonic()-started,
        "limits":["No CUDA arithmetic equivalence test in this receipt","No all18 CAL replay or quality selection",
                  "Finite support is abs<=65504 after low H32 and separately before high cast",
                  "No claim of all finite FP32 input safety"]}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"tests":report["tests"],"out":str(out),"sha256":sha(out)},allow_nan=False))
    if not passed:raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    audit(parser.parse_args().out)
