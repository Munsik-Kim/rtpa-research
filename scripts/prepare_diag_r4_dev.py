"""Fixed three-document retrospective DEV, selected by ID not outputs."""
import argparse
from pathlib import Path
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
panel=read(a.root/'data/benchmarks/gdn/test_panel.json')
ids=['test_natural_language_0','test_code_0','test_associative_recall_0']
items=[next(x for x in panel['items'] if x['id']==i) for i in ids]
assert all(len(x['input_ids'])==1024 for x in items)
methods={'NATIVE':{'codec':'NATIVE'}}
for codec in ('LEGACY_P_PRE','R2_OFFSET'):
    for mask,path in [('LEGACY_DIAG','data/benchmarks/gdn/policy.npz'),('R2_DIAG','data/benchmarks/diag_r2/policy_r2.npz')]:
        methods[codec+'__'+mask]={'codec':codec,'path':path,'sha256':sha(a.root/path),'name':'DIAG8'}
for name, obj in [('panel.json',{'split':'RETROSPECTIVE_DEV','items':items,'parent_panel_sha256':sha(a.root/'data/benchmarks/gdn/test_panel.json')}),
    ('protocol.json',{'run_id':'RTPA_DIAG_ATTRIBUTION_R4_20260912_V1','phase':'A_MODEL_DEV_CODEC_MASK_CROSS','methods':methods,
       'model_revision':'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68','layers':'all18_GDN','prefixes':[256,512,1024],
       'primary_window':'KL[16,L); NLL[16,L-1), own input tokens, output before storage',
       'sampling':'first predefined prior upgrade TEST document per family; now DEV; no new independent test',
       'planned_forwards':15360,'backend':'readable codec references; BF16 native, FP32 recurrence, TF32 off',
       'new_codec_repair_candidates':[],'frozen_before_outputs':True})]:
    path=a.out/name
    if path.exists() and read(path)!=obj:raise ValueError('Do not replace frozen DEV design')
    if not path.exists():atomic(path,obj)
print('Prepared fixed DEV 3 documents x5 paths x1024')
