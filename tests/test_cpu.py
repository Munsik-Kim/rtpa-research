import os
import unittest
from pathlib import Path
import numpy as np
from rtpa_research.io import strict,read,close
from rtpa_research.reproduce import parse,verify_gt,gain,logp
from rtpa_research.boundary import affine_reference
from rtpa_research.verify import mask_audit,source_audit

ROOT=Path(__file__).resolve().parents[1]

class CPUContract(unittest.TestCase):
    def test_strict_json(self):
        for s in ['{"a":NaN}','{"a":Infinity}','{"a":1,"a":2}','{"a":1e999}']:
            with self.assertRaises(ValueError):strict(s)

    def test_parser(self):
        for raw,want in [(' 4827\n\n','4827'),('04827',None),('4827 extra',None),('4827 5000',None),('',None),('possibly 4827',None),('4828','4828')]:self.assertEqual(parse(raw),want)

    def test_latest_gt(self):
        i=dict(record_text='KABCDE = 1256\nKFGHIJ = 4432\nKABCDE = 9317',query_key='KABCDE',ground_truth='9317')
        self.assertEqual(verify_gt(i),'9317')

    def test_nonzero_denominator(self):
        self.assertIsNone(gain(0,0));self.assertEqual(gain(2,1),-1)

    def test_kl_direction(self):
        p=logp([1,2,3]);q=logp([3,1,0]);a=float(np.sum(np.exp(p)*(p-q)));b=float(np.sum(np.exp(q)*(q-p)))
        self.assertGreater(a,0);self.assertNotAlmostEqual(a,b)

    def test_codec_zero_constant_finite(self):
        for profile in ['P_STORE','P_PRE']:
            for x in [np.zeros(32),np.ones(32)*.5,np.linspace(-1,1,32),np.linspace(0,1e-6,32)]:
                a=affine_reference(x,profile);self.assertTrue(np.isfinite(a['decoded']).all());self.assertEqual(a['scale'].dtype,np.float16);self.assertEqual(a['codes'].dtype,np.uint8)

    def test_expected_overflow(self):
        x=np.load(ROOT/'data/evidence/fixtures/overflow.npz',allow_pickle=False)['group']
        for p in ['P_STORE','P_PRE']:
            a=affine_reference(x,p);self.assertTrue(np.isneginf(a['zero']).all());self.assertEqual(a['raw_zero'].item(),-73388)

    def test_frozen_mask_and_score(self):
        x=mask_audit(ROOT);self.assertEqual(x['DIAG_MATCHED_row_overlap'],132);self.assertEqual(x['DIAG_MATCHED_identical_heads'],1)

    def test_source_receipt(self):self.assertEqual(source_audit(ROOT)['GPU_equivalence'],'NOT_RUN')

    def test_synthetic_generator_reproduces_all_contexts(self):
        from tokenizers import Tokenizer
        from rtpa_research.task_generator import generate
        t=Tokenizer.from_file(str(ROOT/'data/tokenizer/tokenizer.json'))
        def adapter(text,add_special_tokens=False):return {'input_ids':t.encode(text,add_special_tokens=add_special_tokens).ids}
        panels=[read(ROOT/f'data/panels/v06_{name}.json') for name in ['eval_panel','cal_panel_template1','cal_panel_template2']]
        for panel in panels:
            for item in panel['items']:
                r=generate(adapter,item['family'],item['length'],int(item['item_id'].rsplit('_',1)[1]),item['role'],item['template'])
                for k in ['text','token_ids','events','ground_truth','seed','text_sha256','token_sha256']:self.assertEqual(r[k],item[k])

    def test_focal_token_boundary(self):
        p=read(ROOT/'data/evidence/v07/candidate_panel.json');f=p['focal_foil'];item=p['items'][0]
        j=f['longest_common_prefix_length'];self.assertEqual(item['canonical_target_ids'][:j],f['token_ids'][:j]);self.assertNotEqual(item['canonical_target_ids'][j],f['token_ids'][j]);self.assertEqual(item['prompt_tokens']-1+j,f['predictor_feed_index'])

    def test_publication_license_and_english_explanation(self):
        import re,tomllib
        license_text=(ROOT/'LICENSE').read_text()
        self.assertIn('Apache License',license_text)
        self.assertIn('Copyright 2026 Munsik-Kim',license_text)
        self.assertIn('Copyright 2026 Alibaba Cloud',(ROOT/'data/tokenizer/LICENSE').read_text())
        self.assertIn('RTPA recurrent-state precision allocation research',(ROOT/'NOTICE').read_text())
        self.assertEqual(tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['license']['file'],'LICENSE')
        for path in [ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md',*sorted((ROOT/'docs').glob('*.md'))]:
            self.assertIsNone(re.search('[\uac00-\ud7a3]',path.read_text()),str(path))
        self.assertIn('not an additional license restriction',(ROOT/'README.md').read_text())
        self.assertIn('NOT_COMPUTED',(ROOT/'docs/RESULTS.md').read_text())
        self.assertIn('COST_UNRESOLVED',(ROOT/'docs/RESULTS.md').read_text())

if __name__=='__main__':unittest.main()
