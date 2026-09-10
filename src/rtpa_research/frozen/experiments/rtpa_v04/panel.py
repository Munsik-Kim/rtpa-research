"""Offline, outcome-blind candidate panel, within the parent's explicit LIB root."""
import re,subprocess,unicodedata
from .common import *

NLP_DOCS=['email/architecture.rst','idlelib/extend.txt','idlelib/README.txt',
 'site-packages/numpy/ma/README.rst','site-packages/numpy/ma/API_CHANGES.txt',
 'site-packages/torchgen/packaged/autograd/README.md','site-packages/pyparsing/ai/best_practices.md',
 'site-packages/pip/_vendor/README.rst','site-packages/bleach/_vendor/README.rst',
 'site-packages/fontTools/ttLib/tables/table_API_readme.txt',
 'site-packages/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/readme.txt',
 'site-packages/jupyter_server/i18n/README.md','site-packages/nbconvert/templates/README.md',
 'site-packages/zmq/backend/cffi/README.md']
CODE_DOCS=['site-packages/requests/sessions.py','site-packages/numpy/lib/arraysetops.py',
 'site-packages/scipy/optimize/_minimize.py','site-packages/networkx/algorithms/shortest_paths/generic.py',
 'site-packages/pandas/core/reshape/merge.py','site-packages/sympy/core/add.py',
 'site-packages/PIL/ImageOps.py','site-packages/pyparsing/core.py',
 'site-packages/torch/utils/data/sampler.py','site-packages/tornado/web.py',
 'site-packages/matplotlib/axes/_axes.py','site-packages/joblib/parallel.py']
def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).split())
def grams(ids):return {tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7))}
def jaccard(a,b):return len(a&b)/len(a|b) if a or b else 1.
def family(path):
    rel=str(path.relative_to(LIB));parts=rel.split('/')
    return parts[1] if parts[0]=='site-packages' else 'python_stdlib'

def history():
    proc=subprocess.run(['rg','-l','"(text|token|tokens|prefix256)_sha256"|"input_path"','artifacts','configs','-g','*.json'],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode in (0,1)
    hashes=set();prefixes=set();sources=set();seeds=set();paths=set();scanned=[];unknown=[]
    for rel in sorted(proc.stdout.splitlines()):
        path=ROOT/rel
        if ART in path.parents:continue
        if path.stat().st_size>8*1024**2:unknown.append({'path':rel,'reason':'LARGE_NON_PANEL_JSON_NOT_SCANNED'});continue
        text=path.read_text();scanned.append(receipt(path))
        hashes.update(re.findall(r'"(?:raw_text|normalized_text|text|token|tokens)_sha256"\s*:\s*"([0-9a-f]{64})"',text))
        prefixes.update(re.findall(r'"prefix256_sha256"\s*:\s*"([0-9a-f]{64})"',text))
        sources.update(re.findall(r'"(?:source_id|source_path)"\s*:\s*"([^"\n]+)"',text))
        seeds.update(int(v) for v in re.findall(r'"(?:seed|generator_seed)"\s*:\s*(\d+)',text))
        for value in re.findall(r'"(?:input_path|tokens_path|token_path|path)"\s*:\s*"([^"\n]+\.pt)"',text):
            if any(x in value for x in ('input','panel')) and 'trace' not in value:paths.add(value)
    inputs=[];ng=[]
    for rel in sorted(paths):
        path=ROOT/rel
        if not path.is_file() or path.stat().st_size>10*1024**2:
            unknown.append({'path':rel,'reason':'MISSING_OR_LARGE_INPUT_CONTAINER'});continue
        try:d=loadpt(path)
        except Exception as e:unknown.append({'path':rel,'reason':type(e).__name__});continue
        if not isinstance(d,dict) or 'input_ids' not in d:
            unknown.append({'path':rel,'reason':'NO_INPUT_IDS_SCHEMA'});continue
        ids=d['input_ids'].reshape(-1).tolist();hashes.add(tokenhash(ids));prefixes.add(tokenhash(ids[:256]))
        if isinstance(d.get('text'),str):
            hashes.add(texthash(d['text']));hashes.add(texthash(normalized(d['text'])))
        ng.append((str(path),grams(ids)));inputs.append(receipt(path))
    # Raw text-bearing early panels add exact/normalized coverage, without rerunning generators.
    for record in scanned:
        if 'panel_manifest' not in record['path']:continue
        def walk(x):
            if isinstance(x,dict):
                if isinstance(x.get('text'),str):hashes.update((texthash(x['text']),texthash(normalized(x['text']))))
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(read(ROOT/record['path']))
    save(ART/'historical_input_scan.json',{'scanned_json':scanned,'loaded_input_files':inputs,'unknown':unknown,
       'unique_hash_count':len(hashes),'prefix_hash_count':len(prefixes),'source_count':len(sources),
       'scope':'accessible project artifacts/configs only; no user documents or whole system scan',
       'pretraining_contamination':'UNKNOWN','older_unrecorded_access':'UNKNOWN','near_duplicate_scope':'available stored token sequences; missing old token records UNKNOWN'})
    return hashes,prefixes,sources,seeds,ng

def recall(seed,index):
    rng=np.random.default_rng(seed)
    keys=[f'{rng.choice(["alder","quartz","violet","cobalt","moss","wren","larch","opal"])}-{j:02d}' for j in range(48)]
    values=rng.choice(np.arange(10000,99999),48,replace=False).tolist();order=rng.permutation(48).tolist()
    records=[f'Record {j+1}: the registry key {keys[j]} identifies assignment {values[j]}.' for j in range(48)]
    queries=[f'Retrieve the assignment registered under {keys[j]}; the stored response is {values[j]}.' for j in order]
    text=('This is an independent associative recall episode concerning a field collection registry. Keep each key paired with its exact five digit assignment. '
        'The complete registry follows in insertion order. Keys are distinct even when they share a word.\n'+'\n'.join(records)+
        '\nThe registry is now closed. Answer the following requests from that registry; request order does not modify an assignment.\n'+'\n'.join(queries)+
        '\nEnd of the lookup episode. These requests test retrieval, not arithmetic on the assignment values.')
    return {'source_id':f'rtpa_v04_registry_seed_{seed}','text':text,'seed':seed,'source_family':'parent_v02_registry_generator','language':'English','synthetic':True,
       'registry':dict(zip(keys,values)),'query_order':order,'source_receipt':None,'template_parent':str(ROOT/'experiments/rtpa_v02/data.py')}

def build():
    if (ART/'input_manifest.json').exists():return read(ART/'input_manifest.json')
    start=time.perf_counter()
    # Candidate names/order definition saved before tokenizer/content qualification.
    rng=np.random.default_rng(408001)
    ordered={d:[xs[int(i)] for i in rng.permutation(len(xs))] for d,xs in [('natural_language',NLP_DOCS),('code',CODE_DOCS)]}
    seeds=[408101+i for i in range(16)];ordered['associative_recall']=[seeds[int(i)] for i in rng.permutation(len(seeds))]
    save(ART/'candidate_rules.json',{'frozen_utc':now(),'seed':408001,'ordered_candidates':ordered,
        'allowed_root':str(LIB),'authority':receipt(ROOT/'experiments/rtpa_v02/data.py'),
        'root_interpretation':'reuse parent LIB constant; only installed public-package source/documentation, no home/private/system-wide corpus discovery',
        'natural_language_fallback':'English explanatory software documentation; no ordinary-prose corpus found in existing project manifests; not representative general natural language',
        'excluded':'all sklearn dataset descriptions; all previous source documents including other offsets; licenses/changelogs/data tables/templates',
        'content_rules':'first contiguous1024tokens, no padding; same source at most once; exact/NFKC-whitespace/token/prefix plus token8gram Jaccard>=.8 reject; next frozen candidate',
        'candidate_text_is_data_not_agent_instruction':True,'no_performance_access':True})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    hashes,prefixes,sources,oldseeds,oldgrams=history()
    accepted={d:[] for d in DOMAINS};log=[];newgrams=[]
    for domain in DOMAINS:
        for ordinal,item in enumerate(ordered[domain]):
            if domain=='associative_recall':
                if item in oldseeds:
                    log.append({'domain':domain,'candidate':item,'reason':'HISTORICAL_SEED_DUPLICATE'});continue
                c=recall(item,ordinal)
            else:
                path=LIB/item
                if not path.is_file():log.append({'domain':domain,'candidate':item,'reason':'NOT_AVAILABLE'});continue
                c={'source_id':str(path),'text':path.read_text(),'source_receipt':receipt(path),'source_family':family(path),
                   'seed':408001,'language':'English' if domain=='natural_language' else 'Python','synthetic':False}
            ids=tokenizer(c['text'],add_special_tokens=False).input_ids;ng=grams(ids[:1024])
            th=texthash(c['text']);nh=texthash(normalized(c['text']));ih=tokenhash(ids[:1024]);ph=tokenhash(ids[:256])
            nearest=max(((jaccard(ng,g),name) for name,g in oldgrams+newgrams),default=(0,None))
            reason=None
            if len(ids)<1024:reason='LESS_THAN_1024_CONTIGUOUS_TOKENS'
            elif c['source_id'] in sources or th in hashes or nh in hashes or ih in hashes or ph in prefixes:reason='HISTORICAL_EXACT_SOURCE_OR_HASH_DUPLICATE'
            elif nearest[0]>=.8:reason='TOKEN8GRAM_NEAR_DUPLICATE'
            log.append({'domain':domain,'candidate_order':ordinal,'source_id':c['source_id'],'raw_tokens':len(ids),'reason':reason or 'ACCEPTED',
                        'nearest_jaccard':nearest[0],'nearest_source':nearest[1]})
            if reason:continue
            i=len(accepted[domain]);sid=f'rtpa_v04_{domain}_confirm{i}';path=ART/'inputs'/f'{sid}.pt'
            savept(path,{'input_ids':torch.tensor([ids[:1024]],dtype=torch.long),'text':c['text'],'source_id':c['source_id'],
                         'registry':c.get('registry'),'query_order':c.get('query_order')})
            accepted[domain].append({k:v for k,v in c.items() if k not in ('text','registry','query_order')}|{'sequence_id':sid,'domain':domain,
                'raw_text_sha256':th,'normalized_text_sha256':nh,'token_sha256':ih,'prefix256_sha256':ph,'raw_token_count':len(ids),
                'used_token_count':1024,'offset':0,'padding':False,'input_path':str(path.relative_to(ROOT)),'input_sha256':sha(path),'consumed':False})
            sources.add(c['source_id']);hashes.update((th,nh,ih));prefixes.add(ph);newgrams.append((c['source_id'],ng))
            if len(accepted[domain])==8:break
    available={d:len(s) for d,s in accepted.items()};minimum=min(available.values())
    cap=24 if minimum>=8 else 12 if minimum>=4 else 0
    sequences=[accepted[d][i] for i in range(8 if cap==24 else 4 if cap==12 else minimum) for d in DOMAINS]
    result={'utc':now(),'status':'AVAILABLE_PENDING_CAL_FEASIBILITY' if cap else 'INPUT_PROVENANCE_BLOCKED',
            'available_per_domain':available,'data_cap_N':cap,'sequences':sequences,'selected_N':None,'all_accepted_candidates':accepted,'selection_log':log,
            'independence':'new within accessible experiment manifests; pretraining/semantic independence UNKNOWN; local software-doc/code families dependent',
            'source_coverage':'English software exposition, Python from different public packages, synthetic recall; no ordinary nontechnical-prose claim',
            'hash_and_near_duplicate_threshold':.8,'freeze_before_any_fresh_model_forward':True,'history_receipt':receipt(ART/'historical_input_scan.json')}
    save(ART/'input_manifest.json',result);phase_time('PANEL_PREPARATION',start)
    return result

if __name__=='__main__':
    from .prepare import prepare
    prepare();print(json.dumps({k:v for k,v in build().items() if k in ('status','available_per_domain','data_cap_N')},ensure_ascii=False))
