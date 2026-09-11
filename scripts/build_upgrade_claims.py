"""Link reviewed upgrade observations to claims without changing prior decisions."""
import argparse
import hashlib
import json
from pathlib import Path

from rtpa_research.io import read


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    root=Path(root)
    expected=root/'results/upgrade/reproduction_expected.json'
    obj=read(expected);g=obj['gdn']['summary'];op=obj['gdn2']['summary']
    op_protocol=read(root/'data/benchmarks/gdn2/protocol.json')
    assert g['verification']['structural_integrity']=='PASS'
    common={
        'run_id':g['run_id'],'checkpoint_revision':g['model_revision'],
        'aggregate_path':str(expected.relative_to(root)),'aggregate_sha256':sha(expected),
        'source_manifest':'configs/upgrade_source_freeze.json',
        'source_manifest_sha256':sha(root/'configs/upgrade_source_freeze.json'),
        'protocol_path':'data/benchmarks/gdn/protocol.json',
        'protocol_sha256':sha(root/'data/benchmarks/gdn/protocol.json'),
        'panel_sha256':sha(root/'data/benchmarks/gdn/test_panel.json'),
        'raw_scalar_paths':'data/benchmarks/gdn/tokens/',
        'raw_hash_authority':'results/upgrade/evidence_manifest.json',
        'n_independent_units':g['planned_documents'],
        'independent_unit':'synthetically generated document; shared generator families limit generalization',
        'reproduction_level':'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
        'reproduction_command':'python scripts/reproduce_upgrade.py --root . --write-generated recomputed-upgrade --verify',
        'table_path':'results/upgrade/benchmark_tables.md',
        'mechanical_verification':'Included scalar and receipt reconstruction, not full-logit replication'}
    labels={('RTPA_DIAG','MATCHED_ENERGY'):'ALLOCATION',
            ('RTPA_DIAG','DAMP_PAPER_ADAPTED'):'ADAPTED_DAMP',
            ('FA_CODE_FACTORIZED','STORED_NEAREST'):'FA_CODE'}
    claims=[]
    for c in g['comparisons']:
        key=(c['candidate'],c['baseline'])
        if key not in labels or not c['same_intervention_scope']:continue
        claims.append({**common,'claim_id':f"GDN_{labels[key]}_{c['context']}",
            'hypothesis_axis':'B_output_transfer','evaluation_level':'MODEL_EVALUATED',
            'policy_sha256':g['policy_sha256'] if labels[key]=='FA_CODE' else g['allocation_policy_sha256'],
            'scope':{'architecture':'GDN','quantized_layers':c['candidate_quantized_layers'],
                     'context':c['context'],'same_codec_payload':True,'profile':'P_PRE',
                     'mixed_payload_bytes_per_head':19328,'high_rows_per_head':8},
            'status':c.get('quality_status','UNRESOLVED'),
            'observed':c,'unit':'Native-reference KL / next-token NLL in nat/token; relative gain is a fraction, not accuracy',
            'limitations':['Small synthetic panel; prefixes share documents',
                           'No new task-accuracy endpoint',
                           'Allocation contrasts retain residual/readout/sampling differences described in METHOD',
                           'No causal decomposition of final nonlinear logits']})
    for c in op['quality']['comparisons']:
        label='ALLOCATION' if c['candidate']=='RTPA_DIAG' else 'FA_CODE'
        ci=c['CI95'];state='SUPPORTED_IN_SCOPE' if ci[0]>0 else 'ADVERSE_QUALITY_OBSERVATION' if ci[1]<0 else 'UNRESOLVED'
        claims.append({'claim_id':'GDN2_OPERATOR_'+label,'run_id':op_protocol['run_id'],
            'parent_stage_run_id':g['run_id'],
            'hypothesis_axis':'B_operator_transfer','evaluation_level':'OPERATOR_TESTED',
            'checkpoint_revision':None,'checkpoint_reason':'OFFICIAL_TRAINED_CHECKPOINT_NOT_VERIFIED',
            'protocol_path':'data/benchmarks/gdn2/protocol.json',
            'protocol_sha256':sha(root/'data/benchmarks/gdn2/protocol.json'),
            'source_manifest':'data/benchmarks/gdn2/freeze.json',
            'source_manifest_sha256':sha(root/'data/benchmarks/gdn2/freeze.json'),
            'raw_scalar_paths':'data/benchmarks/gdn2/operator_scalars.json',
            'raw_sha256':sha(root/'data/benchmarks/gdn2/operator_scalars.json'),
            'policy_sha256':sha(root/'data/benchmarks/gdn2/policy_and_stats.npz'),
            'aggregate_path':str(expected.relative_to(root)),'aggregate_sha256':sha(expected),
            'scope':'Four128x128heads, synthetic TRAIN3/TEST12,256tokens; scored16:256',
            'n_independent_units':12,'unit':'Readout SSE/token summed over four heads; not language KL',
            'status':state,'observed':c,'reproduction_level':'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'reproduction_command':common['reproduction_command'],
            'limitations':['No learned-model KL/NLL or task evidence','No official-kernel parity','No model peak-VRAM measurement']})
    costs={'GDN':g['timing'],'GDN2_ISOLATED_OPERATOR':obj['gdn2_isolated_cost']}
    return {'schema_version':1,'role':'New implementation/evidence mapping; historical claims and frozen decisions remain separate',
            'historical_claims_path':'results/claims.json','claims':claims,
            'cost_evidence':costs,'cost_scope':'GDN is eager model token-step timing; GDN2 is a four-head operator. Never pool them.',
            'initial_GDN2_timing':'CONFOUNDED_OTHER_GPU_WORKER_RETAINED_NOT_POOLED',
            'no_combined_DIAG_FA_gain':True,'production_readiness':'NOT_ESTABLISHED',
            'historical_decisions_overwritten':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=build(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'claims':len(value['claims']),'historical_decisions_overwritten':False}))
