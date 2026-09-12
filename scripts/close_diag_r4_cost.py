"""Seal a stopped R4 cost artifact set; never start work or infer experiment closure.

Invoke only after the operator has ended all cost work and completed its CPU
analysis. An incomplete result is publishable, but a running attempt is not.
The receipt freezes every file identity; no source or observation is rewritten.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from check_diag_r4_public import read, sha
from curate_diag_r4 import validate_cost_closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--confirm-cost-ended', action='store_true', required=True,
                        help='Operator assertion that all cost processes have ended; not inferred from statistics')
    args = parser.parse_args()
    stage = args.run.resolve()
    destination = stage / 'publication_closure.json'
    if destination.exists():
        raise FileExistsError('COST_PUBLICATION_ALREADY_SEALED')
    artifacts = {}
    for path in sorted(stage.rglob('*')):
        if path.is_symlink():
            raise ValueError('SYMLINK_IN_COST_ARTIFACTS')
        if path.is_file():
            if path.suffix not in ('.json', '.py', '.npy', '.npz', '.log', '.txt'):
                raise ValueError('UNREVIEWED_COST_ARTIFACT_TYPE:' + path.name)
            if path.suffix == '.json':
                read(path)  # Strict duplicate-key and recursive finite checks.
            artifacts[str(path.relative_to(stage))] = sha(path)
    receipt = {'schema': 'R4_COST_PUBLICATION_CLOSURE_V1', 'status': 'CLOSED',
               'UTC': datetime.now(timezone.utc).isoformat(), 'operator_asserted_cost_ended': True,
               'closure_builder_sha256': sha(Path(__file__)),
               'model_forwards': 0, 'GPU_forwards': 0, 'artifacts': artifacts}
    for name, key in (('freeze.json', 'cost_freeze_sha256'),
                      ('cost_wiring_revision.json', 'cost_wiring_revision_sha256'),
                      ('analysis_contract.json', 'analysis_contract_sha256'),
                      ('analysis/summary.json', 'analysis_summary_sha256')):
        receipt[key] = artifacts[name]
    receipt['scientific_status'] = validate_cost_closure(stage, receipt)
    with destination.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'CLOSED', 'scientific_status': receipt['scientific_status'],
                      'closure_sha256': sha(destination), 'artifact_count': len(artifacts)}))


if __name__ == '__main__':
    main()
