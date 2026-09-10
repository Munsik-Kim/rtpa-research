"""Outcome-blind local documents, fixed candidate order and parent exclusions."""
import types
from .common import *
from experiments.rtpa_v04 import panel as parent_panel
NLP=['site-packages/bpy/5.0/scripts/addons_core/bl_pkg/readme.rst','site-packages/bpy/lib/materialx/libraries/README.md',
     'idlelib/idle_test/README.txt','site-packages/playhouse/README.md','email/architecture.rst','site-packages/jupyter_server/i18n/README.md',
     'site-packages/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/readme.txt','idlelib/extend.txt']
CODE=['site-packages/PIL/ImageFilter.py','site-packages/scipy/interpolate/_interpolate.py','site-packages/sympy/integrals/integrals.py',
      'site-packages/networkx/algorithms/flow/maxflow.py','asyncio/base_events.py','json/decoder.py','site-packages/pandas/core/window/rolling.py','site-packages/urllib3/connectionpool.py']

def build():
    if (ART/'input_manifest.json').exists():
        d=read(ART/'input_manifest.json')
        for row in d['sequences']:verify({'path':row['input_path'],'sha256':row['input_sha256']})
        return d
    budget(True);tick=time.perf_counter();rng=np.random.default_rng(509001)
    order={d:[bank[int(i)] for i in rng.permutation(len(bank))] for d,bank in [('natural_language',NLP),('code',CODE),('associative_recall',list(range(509101,509117)))]}
    save(ART/'candidate_rules.json',{'utc':now(),'seed':509001,'ordered_candidates':order,'quota_per_domain':4,'insufficiency_rule':'retain all eligible up to4/domain; freeze unequal composition before any new-panel model evaluation, do not fill with old input',
       'root':str(LIB),'corpus':'installed public-package technical documentation, Python code, synthetic registry recall','parent_near_duplicate_threshold':.8,
       'same_document_offsets':'excluded','data_content_not_instructions':True,'performance_access':False})
    # Reuse exact parent history scanner with only its output namespace rebound.
    # No parent global or file is mutated. v0.4 now participates in historical exclusion.
    scope=dict(parent_panel.history.__globals__);scope['ART']=ART
    history=types.FunctionType(parent_panel.history.__code__,scope)
    hashes,prefixes,sources,seeds,oldgrams=history()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    accepted={d:[] for d in DOMAINS};log=[];newgrams=[]
    for domain in DOMAINS:
        for ordinal,item in enumerate(order[domain]):
            if domain=='associative_recall':
                if item in seeds:log.append({'domain':domain,'candidate':item,'reason':'HISTORICAL_SEED'});continue
                c=parent_panel.recall(item,ordinal);c['source_id']=f'rtpa_v05_registry_seed_{item}'
            else:
                path=LIB/item
                if not path.is_file():log.append({'domain':domain,'candidate':item,'reason':'NOT_AVAILABLE'});continue
                c={'source_id':str(path),'text':path.read_text(),'seed':509001,'source_family':parent_panel.family(path),'source_receipt':receipt(path),
                    'synthetic':False,'language':'English technical documentation' if domain=='natural_language' else 'Python'}
            ids=tokenizer(c['text'],add_special_tokens=False).input_ids;ng=parent_panel.grams(ids[:1024])
            th=texthash(c['text']);nh=texthash(parent_panel.normalized(c['text']));ih=tokenhash(ids[:1024]);ph=tokenhash(ids[:256])
            nearest=max(((parent_panel.jaccard(ng,g),name) for name,g in oldgrams+newgrams),default=(0,None))
            reason=None
            if len(ids)<1024:reason='LESS_THAN_1024_CONTIGUOUS_TOKENS'
            elif c['source_id'] in sources or any(h in hashes for h in (th,nh,ih)) or ph in prefixes:reason='HISTORICAL_SOURCE_OR_HASH_DUPLICATE'
            elif nearest[0]>=.8:reason='NEAR_DUPLICATE_8GRAM'
            log.append({'domain':domain,'ordinal':ordinal,'source_id':c['source_id'],'raw_token_count':len(ids),'reason':reason or 'ACCEPTED','nearest_8gram_jaccard':nearest[0],'nearest_source':nearest[1]})
            if reason:continue
            sid=f'rtpa_v05_{domain}_confirm{len(accepted[domain])}';dest=ART/'inputs'/f'{sid}.pt'
            savept(dest,{'input_ids':torch.tensor([ids[:1024]],dtype=torch.long),'text':c['text'],'source_id':c['source_id'],'registry':c.get('registry'),'query_order':c.get('query_order')})
            r={k:v for k,v in c.items() if k not in ('text','registry','query_order')}
            r.update(sequence_id=sid,domain=domain,input_path=str(dest.relative_to(ROOT)),input_sha256=sha(dest),raw_text_sha256=th,normalized_text_sha256=nh,token_sha256=ih,prefix256_sha256=ph,
                raw_token_count=len(ids),used_token_count=1024,offset=0,padding=False,bootstrap_unit=sid,consumed_at_freeze=False)
            accepted[domain].append(r);hashes.update((th,nh,ih));prefixes.add(ph);sources.add(c['source_id']);newgrams.append((c['source_id'],ng))
            if len(accepted[domain])==4:break
    seq=[accepted[d][i] for i in range(4) for d in DOMAINS if len(accepted[d])>i];counts={d:len(accepted[d]) for d in DOMAINS}
    d={'status':'FROZEN_BEFORE_FRESH_FORWARD' if seq else 'BLOCKED_NEW_INPUTS','utc':now(),'sequences':seq,'selected_N':len(seq),'domain_counts':counts,'selection_log':log,
      'input_seed':509001,'history_receipt':receipt(ART/'historical_input_scan.json'),'quota_shortfall':{d:4-counts[d] for d in DOMAINS},
      'scope':'new documents within accessible project history, not unseen pretraining data; limited English technical software corpus',
      'source_family_dependence':'shared library families possible; synthetic recall shares generator; CI not population or family-wise safety',
      'resampling':'one whole original document/episode per sequence, no document chunks; preserve domain counts; share all draws'}
    save(ART/'input_manifest.json',d);phase_time('PANEL',tick)
    return d
