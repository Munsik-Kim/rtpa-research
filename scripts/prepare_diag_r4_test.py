"""Select local licensed source documents, before model outputs, without downloads."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sysconfig
import numpy as np
from tokenizers import Tokenizer
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic


def digest(value):return hashlib.sha256(value).hexdigest()
def token_hash(ids):return digest(','.join(map(str,ids)).encode())
def grams(ids):return set(tuple(ids[i:i+5]) for i in range(max(0,len(ids)-4)))


def walk(value):
    if isinstance(value,dict):
        yield value
        for x in value.values():yield from walk(x)
    elif isinstance(value,list):
        for x in value:yield from walk(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--site-packages',type=Path,default=Path(sysconfig.get_paths()['purelib']))
    p.add_argument('--additional-panel-root',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfgpath=a.root/'configs/diag_r4_test_selection.json';cfg=read(cfgpath)
    if (a.out/'panel.json').exists():raise ValueError('Never replace selected panel; use existing freeze')
    tok=Tokenizer.from_file(str(a.root/'data/tokenizer/tokenizer.json'))
    patterns=['*panel*.json','*sources*.json','*input*.json']
    prior=set()
    for pattern in patterns:prior.update((a.root/'data').rglob(pattern))
    if a.additional_panel_root:
        for pattern in patterns:prior.update(a.additional_panel_root.rglob(pattern))
    oldhash=set();oldpaths=set();oldgrams=[];oldreceipts=[];unavailable=[]
    for file in sorted(prior):
        if file.stat().st_size>20*1024*1024:
            unavailable.append({'path':str(file.name),'reason':'INPUT_MANIFEST_OVER20MIB_NOT_PARSED'});continue
        try:obj=read(file)
        except (ValueError,UnicodeError):
            unavailable.append({'path':file.name,'reason':'NOT_STRICT_JSON'});continue
        oldreceipts.append({'path':str(file.relative_to(a.root)) if file.is_relative_to(a.root) else 'external/'+str(file.relative_to(a.additional_panel_root)), 'sha256':sha(file)})
        for rec in walk(obj):
            for key,value in rec.items():
                if 'sha256' in key and isinstance(value,str):oldhash.add(value)
                if key in ('source_id','source_path','path') and isinstance(value,str):oldpaths.add(value.replace('\\','/'))
            ids=rec.get('input_ids')
            if isinstance(ids,list) and ids and all(type(x)is int for x in ids):
                oldgrams.append(grams(ids));oldhash.add(token_hash(ids));oldhash.add(token_hash(ids[:256]))
            text=rec.get('text') or rec.get('prompt')
            if isinstance(text,str):
                oldhash.add(digest(text.encode()));oldhash.add(digest(' '.join(text.split()).encode()))
    rows=[];excluded=[];licenses=[]
    for family in cfg['source_families']:
        dist=md.distribution(family);version=dist.version
        license_paths=[a.site_packages/f for f in dist.files if str(f).endswith(('/LICENSE','/LICENSE.txt','/licenses/LICENSE','/licenses/NOTICE')) and '.dist-info/' in str(f)]
        if not license_paths:raise ValueError('MISSING_DISTRIBUTION_LICENSE:'+family)
        for path in license_paths:
            target=a.out/'licenses'/family/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
            licenses.append({'family':family,'version':version,'path':str(target.relative_to(a.out)),'sha256':sha(target)})
        candidates=[x for x in (a.site_packages/family).rglob('*.py') if cfg['source_min_bytes']<=x.stat().st_size<=cfg['source_max_bytes']
                    and not set(x.relative_to(a.site_packages/family).parts)&set(cfg['excluded_path_components'])]
        candidates.sort(key=lambda x:digest(f'{cfg["seed"]}:{family}:{x.relative_to(a.site_packages/family)}'.encode()))
        selected=0
        for path in candidates:
            relative=str(path.relative_to(a.site_packages));text=path.read_text(encoding='utf-8');ids=tok.encode(text,add_special_tokens=False).ids
            rawsha=digest(text.encode());normsha=digest(' '.join(text.split()).encode());chosen=ids[:cfg['length']]
            identity={'source':relative,'source_sha256':rawsha};reason=None
            if any(x.endswith('/'+relative) or x==relative for x in oldpaths):reason='PRIOR_SOURCE_DOCUMENT'
            elif len(ids)<cfg['length']:reason='TOO_SHORT_NO_PADDING'
            elif {rawsha,normsha,token_hash(chosen),token_hash(chosen[:256])}&oldhash:reason='PRIOR_OR_SELECTED_HASH'
            else:
                g=grams(chosen)
                if any(len(g&other)/len(g|other)>=.5 for other in oldgrams if g|other):reason='TOKEN5GRAM_NEAR_DUPLICATE'
            if reason:excluded.append({**identity,'reason':reason});continue
            rows.append({'id':f'r4_test_{family}_{selected}','domain':family,'split':'TEST','source_family':family,
                         'source_version':version,'source_relative_path':relative,'source_sha256':rawsha,
                         'normalized_text_sha256':normsha,'token_sha256':token_hash(chosen),'prefix256_sha256':token_hash(chosen[:256]),
                         'text':text,'input_ids':chosen,'length':len(chosen),'raw_tokens':len(ids),'offset':0,
                         'synthetic':False,'language':'Python source with comments/docstrings',
                         'bootstrap_unit':relative,'seed':cfg['seed']})
            oldhash.update((rawsha,normsha,token_hash(chosen),token_hash(chosen[:256])));oldpaths.add(relative);oldgrams.append(g)
            selected+=1
            if selected==cfg['documents_per_source_family']:break
        if selected!=cfg['documents_per_source_family']:raise ValueError('INSUFFICIENT_ELIGIBLE_DOCUMENTS:'+family)
    atomic(a.out/'panel.json',{'split':'TEST','items':rows,'scope':cfg['scope'],'selection_config_sha256':sha(cfgpath)})
    atomic(a.out/'selection_receipt.json',{'status':'FROZEN_INPUTS_NO_MODEL_OUTPUT','config':cfg,'config_sha256':sha(cfgpath),
        'source_sha256':sha(Path(__file__)),'panel_sha256':sha(a.out/'panel.json'),'prior_manifests':oldreceipts,
        'excluded':excluded,'unavailable_prior_checks':unavailable,'licenses':licenses,
        'coverage_limit':'Accessible recorded manifests only; does not claim pretraining independence or an unrecorded consumed-input audit'})
    print(json.dumps({'selected':len(rows),'sources':[x['source_relative_path'] for x in rows],'excluded':len(excluded),'prior_manifests':len(oldreceipts)}))


if __name__=='__main__':main()
