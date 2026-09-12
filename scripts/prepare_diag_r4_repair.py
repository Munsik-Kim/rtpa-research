"""Register the single paired-rounding DEV intervention, never a new panel."""
import argparse
from pathlib import Path
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--dev-run',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rules=read(a.root/'configs/diag_r4_repair_selection.json');panel=read(a.dev_run/'panel.json')
assert [x['id'] for x in panel['items']]==rules['DEV_documents']
path='data/benchmarks/gdn/policy.npz'
manifest={'run_id':'RTPA_DIAG_ATTRIBUTION_R4_20260912_V1','phase':'A_MODEL_SINGLE_RNE_INTERVENTION',
          'model_revision':'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68','prefixes':[256,512,1024],
          'methods':{'NATIVE':{'codec':'NATIVE'},'RNE_LEGACY_MASK':{'codec':'R4_OFFSET_RNE_V1','path':path,'sha256':sha(a.root/path),'name':'DIAG8'}},
          'selection_rules':rules,'selection_rules_sha256':sha(a.root/'configs/diag_r4_repair_selection.json'),
          'reference_A_freeze_sha256':sha(a.dev_run/'freeze.json'),'planned_forwards':6144,
          'scope':'Exploratory codec intervention on same retrospective DEV; paired-rounding change, not isolated scale or offset effect'}
for name,value in [('panel.json',panel),('protocol.json',manifest)]:
    dest=a.out/name
    if dest.exists() and read(dest)!=value:raise ValueError('Frozen candidate plan cannot be replaced')
    if not dest.exists():atomic(dest,value)
print('Registered one RNE codec x fixed legacy mask, Native rerun,3DEVdocuments')
