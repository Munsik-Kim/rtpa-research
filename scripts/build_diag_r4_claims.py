"""Build the R4 claim-to-evidence index from closed summaries.

This is a presentation/indexing step, not independent numerical verification.
It never selects a codec, changes an expected result, or imports a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from aggregate_diag_r4 import read


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(root):
    raw = root / 'data/benchmarks/diag_r4'
    result = root / 'results/diag_r4'
    model = read(result / 'phase_b_model/summary.json')
    assessment = read(result / 'assessment.json')
    frozen = read(raw / 'phase_b_model/freeze.json')['binding']
    selection = read(raw / 'codec_selection/selection.json')
    cost = read(raw / 'cost/analysis/summary.json')
    if assessment['source_summary_sha256'] != sha(result / 'phase_b_model/summary.json'):
        raise ValueError('MODEL_ASSESSMENT_SOURCE_MISMATCH')
    common = {
        'run_id': model['run_id'],
        'checkpoint_revision': frozen['manifest']['model_revision'],
        'model_source_sha256': frozen['native_source_sha256'],
        'model_numeric_source_files': frozen['package'],
        'experiment_protocol_sha256': frozen['manifest_sha256'],
        'checkpoint_config_sha256': next(x['sha256'] for x in frozen['model_files'] if x['file'] == 'config.json'),
        'panel_sha256': frozen['panel_sha256'],
        'panel_role': 'Fixed source-file TEST; not an answer task or broad-domain sample',
        'independent_unit': 'Source document; paired resampling within software project',
        'n_documents': len(frozen['panel']['items']),
        'profile': 'LEGACY_P_PRE; all18 GDN; high8; actual UINT8/FP16 storage',
        'logical_alias': 'B4 = LEGACY_DIAG only after whole-mask and path identity',
    }
    claims = []

    def add(identity, statement, status, observations, paths, level, limitations):
        refs = [{'path': path, 'sha256': sha(root / path)} for path in paths]
        claims.append({'claim_id': identity, 'statement': statement, 'status': status,
            'observed': observations, 'evidence': refs, 'reproduction_level': level,
            'limitations': limitations, 'mechanical_PASS_is_scientific_success': False})

    add('R4_A_CODEC_MASK', 'The codec disadvantage remains when either mask is fixed.',
        'SUPPORTED_IN_DEV_SCOPE',
        {'public_summary': 'results/diag_r4/phase_a_model/summary.json',
         'n_documents': 3, 'role': 'Reused synthetic DEV, nested prefixes',
         'physical_forwards': 15360},
        ['results/diag_r4/phase_a_model/summary.json', 'scripts/aggregate_diag_r4.py'],
        'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
        ['Not a complete numerical-cause attribution or an additional independent TEST panel.'])
    add('R4_A_REPAIR', 'The preregistered rounding candidate is not a promoted default.',
        selection['status'],
        {'candidate': selection['rules']['candidate'], 'pooled_KL_ratio': selection['pooled_KL_ratio'],
         'mean_delta_NLL': selection['mean_delta_NLL'], 'selected_codec': selection['selected_codec']},
        ['data/benchmarks/diag_r4/codec_selection/selection.json',
         'results/diag_r4/phase_a_repair_model/summary.json'],
        'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
        ['Bounded fixture success is not whole-model quality or universal range closure.',
         'The selected legacy reference retains its known FP16 zero-point failure.'])
    for contrast in assessment['primary_and_secondary']:
        add('R4_B4_VS_' + contrast['baseline'],
            'Coherent DIAG versus ' + contrast['baseline'] + ' under one codec and payload.',
            contrast['quality_status'], {**common, **contrast},
            ['results/diag_r4/phase_b_model/summary.json',
             'results/diag_r4/phase_b_model/bootstrap_draws.npz',
             'data/benchmarks/diag_r4/policy.npz', 'scripts/aggregate_diag_r4.py'],
            'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            ['KL preservation and next-token NLL are distinct; no accuracy endpoint.',
             'B4/B1 changes the score procedure; B4/B3 isolates the registered cross-write quadratic term, not a whole-model causal share.',
             'Primary 95% and two secondary 97.5% intervals; shared-project dependence remains.'])
    add('R4_C_CALIBRATION', 'Exact adjoints and row chunks reproduce tested frozen responses; sketches are not promoted.',
        'EXACT_POINT_CHECKS_WITH_UNCERTIFIED_SKETCHES',
        {'selected_layers': [0, 12, 22], 'heads_per_layer': 1,
         'model_extension': 'NOT_RUN', 'certified_masks': 0,
         'accounting_wrapper_status': 'FAILED_WITH_SEPARATE_POSTRUN_AUDIT'},
        ['data/benchmarks/diag_r4/phase_c_selected_layers/execution.json',
         'data/benchmarks/diag_r4/phase_c_selected_layers/layer0/summary.json',
         'data/benchmarks/diag_r4/phase_c_selected_layers/layer12/summary.json',
         'data/benchmarks/diag_r4/phase_c_selected_layers/layer22/summary.json',
         'data/benchmarks/diag_r4/phase_c_selected_layers/postrun_accounting_audit.json',
         'src/rtpa_research/diag_r4_adjoint.py'],
        'INCLUDED_TENSOR_POINT_CHECKS',
        ['Pointwise exact masks at some probe sizes do not certify selection.',
         'Core timing excludes capture, fitting I/O and model evaluation; no end-to-end savings claim.',
         'The original one-ULP alpha-accounting discrepancy is retained, not relabeled as an exact nominal bound.'])
    add('R4_RUNTIME', 'Equal-work one-cache cost, separate from quality instrumentation.',
        cost['status'], {'timing': cost['timing'], 'attempts': cost['attempts'],
                         'optimization': cost['conditional_optimization']},
        ['data/benchmarks/diag_r4/cost/analysis/summary.json',
         'scripts/analyze_diag_r4_cost.py'],
        'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS', cost['limitations'])
    add('R4_MEMORY', 'Actual state payload and fresh-process peaks are separate quantities.',
        cost['memory']['status'], cost['memory'],
        ['data/benchmarks/diag_r4/cost/analysis/summary.json',
         'src/rtpa_research/diag_r4_cost.py'],
        'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
        [('Tensor byte arithmetic is reproducible from receipts; process peaks are retained measurements, not a CPU rerun.'
          if cost['memory']['rows'] else 'No fresh-process peak or memory receipt was measured in this run.'),
         'Policy indices/H32, shared counters, scratch, KV/conv cache and model weights are not target payload.',
         'No multi-request serving capacity or total-VRAM proportional saving is inferred.'])
    return {'schema_version': 1, 'run_id': model['run_id'], 'claims': claims,
        'mechanical_verification': 'SEPARATE results/diag_r4/verification_manifest.json',
        'old_decisions_and_expectations': 'UNCHANGED', 'new_stable_default': False,
        'production_readiness': 'NOT_ESTABLISHED',
        'index_source_sha256': sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    if args.out.exists():
        raise ValueError('OUTPUT_EXISTS_USE_NEW_PATH')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'INDEXED_CLOSED_SUMMARIES', 'claims': len(result['claims']),
                      'verification': 'Separate verify_diag_r4 command required; this command indexes rather than verifies.'}))


if __name__ == '__main__':
    main()
