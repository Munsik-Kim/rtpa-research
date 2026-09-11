"""CPU synthetic checks of cohort, window, bootstrap, failure, and cost semantics."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rtpa_research.benchmark_analysis import aggregate, analyze, bootstrap_indices


METHODS=['NATIVE','MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED','STORED_NEAREST','FA_CODE_FACTORIZED']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def token_file(root,item,raw,policy='synthetic-policy'):
    path=root/'tokens'/f'{item["id"]}.jsonl.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as stream:
        for row in raw:stream.write(json.dumps(row,allow_nan=False)+'\n')
    save(path.with_suffix('.receipt.json'),{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'rows':len(raw),'physical_forwards':sum(r['status'] in ('OK','NUMERICAL_FAILURE') for r in raw),
         'policy_sha256':policy,'status':'COMPLETE'})


def fixture(root):
    labels=METHODS+['FA_CODE_REFERENCE','STORED_NEAREST_REPEAT']
    protocol={'run_id':'SYNTHETIC_ANALYSIS_TEST','architecture':'GDN','scope':'synthetic scalar test, no model',
              'methods':METHODS,'TEST_documents':12,'model_revision':'test-only','layers':[0,12,22],
              'policy_sha256':'synthetic-policy','windows':[20,24],'max_context':24,'warmup_tokens':16,
              'quality_primary':['RTPA_DIAG/MATCHED_ENERGY','FA_CODE_FACTORIZED/STORED_NEAREST'],
              'bootstrap_draws':2000,'bootstrap_seed':611204,'cost_target':1.05,
              'timing_labels':labels,'timing_measured_blocks':8,'timing_prefix':128,'timing_decode':32}
    save(root/'protocol.json',protocol)
    panel=[{'id':f'doc{i}','domain':['code','natural_language','associative_recall'][i//4],'input_ids':list(range(24))} for i in range(12)]
    save(root/'test_panel.json',{'split':'TEST','items':panel})
    base={'NATIVE':0.,'MATCHED_ENERGY':.01,'RTPA_DIAG':.008,'DAMP_PAPER_ADAPTED':.011,'STORED_NEAREST':.009,'FA_CODE_FACTORIZED':.007}
    allraw={}
    for item in panel:
        raw=[]
        for t in range(24):
            for method in METHODS:
                raw.append({'document':item['id'],'domain':item['domain'],'method':method,'token':t,
                            'input_id':t,'target_id':t+1 if t<23 else None,'status':'OK','reason':None,
                            'KL':base[method]*(1+t/100),'NLL':2+base[method] if t<23 else None})
        token_file(root,item,raw);allraw[item['id']]=raw
    timing=[]
    costs={m:1+METHODS.index(m)/10 for m in METHODS};costs.update(FA_CODE_REFERENCE=2.,STORED_NEAREST_REPEAT=1.4)
    for block in range(8):
        for order,method in enumerate(labels):
            unit=costs[method]*(1+block/100)
            timing.append({'block':block,'order':order,'method':method,'prompt_tokens':128,'decode_tokens':32,
                           'prefill_seconds':unit*128/1000,'TTFT_seconds':unit*128/1000,'decode_seconds':unit*32/1000,
                           'decode_ms_per_token':unit,'tokens_per_second':1000/unit,
                           'payload_bytes':48*19328,'shared_policy_layout_H_bytes':100000,'cache_tensor_bytes':1100000,
                           'peak_allocated_bytes':2000000,'peak_reserved_bytes':3000000,
                           'gpu_before':{'used_MiB':4000},'gpu_after':{'used_MiB':4001}})
    save(root/'timing.json',{'rows':timing,'timing_scope':'synthetic fixed-workload timing scalars'})
    return panel,allraw


class BenchmarkCPUAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.panel,self.raw=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()

    def primary(self,result,length=24):
        return next(r for r in result['summary']['comparisons'] if r['candidate']=='RTPA_DIAG' and r['baseline']=='MATCHED_ENERGY' and r['context']==length)

    def test_full_cohort_pooled_gain_alignment_and_counts(self):
        result=aggregate(self.root);primary=self.primary(result)
        self.assertAlmostEqual(primary['gain'],.2)
        self.assertEqual(primary['planned_documents'],12)
        self.assertEqual(primary['paired_KL_tokens'],12*8)
        self.assertEqual(primary['paired_NLL_tokens'],12*7)
        self.assertAlmostEqual(primary['delta_NLL'],-.002)
        self.assertEqual((primary['wins'],primary['ties'],primary['losses']),(12,0,0))
        self.assertAlmostEqual(primary['beneficial_KL_mass']-primary['harmful_KL_mass'],-primary['mean_KL_difference']*96)
        np.testing.assert_array_equal(result['bootstrap']['draw_indices'],bootstrap_indices([r['domain'] for r in self.panel]))
        self.assertEqual(result['summary']['physical_quality_forwards_from_rows'],12*24*6)

    def test_missing_document_never_successful_subset_mean(self):
        path=self.root/'tokens/doc0.jsonl.gz';path.unlink()
        result=aggregate(self.root);primary=self.primary(result)
        self.assertIsNone(primary['gain']);self.assertIsNone(primary['candidate_mean_KL'])
        self.assertEqual(primary['planned_documents'],12)
        self.assertEqual(result['coverage'][0]['status'],'NOT_RUN_TOKEN_FILE_MISSING')
        self.assertEqual(result['summary']['verification']['quality_completion'],'INCOMPLETE_OR_FAILED')

    def test_failure_retained_and_earlier_prefix_stays_valid(self):
        raw=copy.deepcopy(self.raw['doc0'])
        for row in raw:
            if row['method']=='RTPA_DIAG' and row['token']>=22:
                row.update(status='NUMERICAL_FAILURE' if row['token']==22 else 'NOT_RUN',
                           KL=None,NLL=None,reason='TEST_FAILURE' if row['token']==22 else 'AFTER_FIRST_FAILURE')
        token_file(self.root,self.panel[0],raw)
        result=aggregate(self.root)
        self.assertIsNone(self.primary(result,24)['gain'])
        self.assertAlmostEqual(self.primary(result,20)['gain'],.2)
        self.assertEqual(result['summary']['numerical_failures'][0]['token'],22)
        self.assertEqual(result['summary']['physical_quality_forwards_from_rows'],12*24*6-1)

    def test_duplicate_row_blocks_numerical_claim(self):
        raw=self.raw['doc0']+[self.raw['doc0'][0]]
        token_file(self.root,self.panel[0],raw)
        result=aggregate(self.root)
        self.assertEqual(result['summary']['verification']['structural_integrity'],'FAIL')
        self.assertIsNone(self.primary(result)['gain'])

    def test_zero_baseline_is_null_not_epsilon(self):
        for item in self.panel:
            raw=copy.deepcopy(self.raw[item['id']])
            for row in raw:
                if row['method']=='MATCHED_ENERGY':row['KL']=0.
            token_file(self.root,item,raw)
        primary=self.primary(aggregate(self.root))
        self.assertEqual(primary['status'],'COMPUTED')
        self.assertIsNone(primary['gain']);self.assertIsNone(primary['gain_CI95'])
        self.assertEqual(primary['ratio_null_reason'],'ZERO_OR_NONPOSITIVE_BASELINE')
        self.assertGreater(primary['mean_KL_difference'],0)

    def test_timing_paired_cost_and_missing_block(self):
        result=aggregate(self.root)
        comp=next(c for c in result['summary']['timing']['comparisons'] if c['candidate']=='FA_CODE_FACTORIZED' and c['baseline']=='FA_CODE_REFERENCE')
        self.assertAlmostEqual(comp['median_ratio'],.75)
        self.assertEqual(comp['status'],'COST_TARGET_MET')
        path=self.root/'timing.json';data=json.loads(path.read_text())
        data['rows']=[r for r in data['rows'] if not (r['block']==7 and r['method']=='FA_CODE_FACTORIZED')];save(path,data)
        result=aggregate(self.root)
        comp=next(c for c in result['summary']['timing']['comparisons'] if c['candidate']=='FA_CODE_FACTORIZED' and c['baseline']=='STORED_NEAREST')
        self.assertIsNone(comp['median_ratio'])
        self.assertEqual(comp['status'],'UNDEFINED_INCOMPLETE_TIMING')

    def test_different_layer_scopes_are_explicit_and_never_primary(self):
        path=self.root/'protocol.json';protocol=json.loads(path.read_text())
        protocol['method_layers']={m:(list(range(18)) if m in ('RTPA_DIAG','MATCHED_ENERGY','DAMP_PAPER_ADAPTED') else [] if m=='NATIVE' else [0,12,22]) for m in METHODS}
        save(path,protocol);result=aggregate(self.root)
        self.assertEqual(result['memory']['calculated_mixed_target_state_bytes_by_method']['RTPA_DIAG'],18*16*19328)
        self.assertEqual(result['memory']['calculated_mixed_target_state_bytes_by_method']['FA_CODE_FACTORIZED'],3*16*19328)
        contrast=next(c for c in result['summary']['comparisons'] if c['candidate']=='FA_CODE_FACTORIZED' and c['baseline']=='MATCHED_ENERGY')
        self.assertFalse(contrast['same_intervention_scope']);self.assertFalse(contrast['primary'])
        self.assertTrue(self.primary(result)['same_intervention_scope'])

    def test_hash_target_mismatch_is_not_silently_accepted(self):
        raw=copy.deepcopy(self.raw['doc0']);raw[0]['target_id']=999
        token_file(self.root,self.panel[0],raw)
        result=aggregate(self.root)
        self.assertTrue(any('INPUT_TARGET_ALIGNMENT' in e['reason'] for e in result['summary']['verification']['input_errors']))
        self.assertIsNone(self.primary(result)['gain'])

    def test_write_outputs_deterministic_and_no_gpu_import(self):
        import subprocess,sys,os
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
        script='import sys; from rtpa_research.benchmark_analysis import aggregate; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        p=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        analyze(self.root);out=self.root/'analysis'
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        analyze(self.root)
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        self.assertEqual(before,after)
        self.assertTrue((out/'sequence_metrics.csv').exists())
        self.assertEqual(len(json.loads((out/'bootstrap_draws.json').read_text())['draw_indices']),2000)


if __name__=='__main__':unittest.main()
