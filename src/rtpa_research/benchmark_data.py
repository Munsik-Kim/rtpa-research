"""Public deterministic synthetic documents, not a real-corpus benchmark.

Each seed supplies an independent simulated ledger/program/field narrative.
TRAIN/CAL/TEST namespaces are disjoint; no answer task or difficulty selection.
"""
import hashlib, random

DOMAINS=('natural_language','code','associative_recall')


def document(domain,seed):
    rng=random.Random(seed)
    if domain=='natural_language':
        names=['cedar','willow','quartz','harbor','orchard','ridge','delta','meadow','valley','summit']
        lines=[f'Field notebook {seed}: independent observations from a simulated environmental survey.']
        for i in range(180):
            place=rng.choice(names);a,b,c=rng.sample(range(10,900),3)
            lines.append(f'On visit {i+1}, the {place} station measured {a} units before the inspection and {b} afterward. The observer assigned sample {c} to a separate container. A new observation does not overwrite the earlier record.')
    elif domain=='code':
        lines=[f'# Synthetic configuration audit module, seed {seed}', 'def audit(records):', '    result = {}']
        for i in range(180):
            a,b,c=rng.sample(range(1,9000),3)
            lines+= [f'    # Validate independent channel {i}',f'    channel_{i} = ({a}, {b}, {c})',
                     f'    result["channel_{i}"] = sum(channel_{i}) % {rng.randrange(17,211)}']
        lines+=['    return result']
    else:
        lines=[f'Simulated registry {seed}. Each record has a unique identifier and independent values.']
        keys=rng.sample(range(10000,99999),180)
        for key in keys:
            a,b,c=rng.sample(range(1000,9999),3)
            lines.append(f'Record K{key}: primary {a}; reserve {b}; checksum {c}. This entry is distinct from the other records.')
    return '\n'.join(lines)+'\n'


def panel(tokenizer,split,count_per_domain,length,base_seed):
    rows=[]
    for di,domain in enumerate(DOMAINS):
        for i in range(count_per_domain):
            seed=base_seed+di*10000+i;text=document(domain,seed)
            ids=tokenizer.encode(text,add_special_tokens=False)
            if len(ids)<length:raise RuntimeError('Generator document too short; no repeat padding permitted')
            ids=ids[:length]
            token_sha=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
            rows.append(dict(id=f'{split.lower()}_{domain}_{i}',domain=domain,split=split,seed=seed,
                             text=text,input_ids=ids,text_sha256=hashlib.sha256(text.encode()).hexdigest(),token_sha256=token_sha,
                             source_family='RTPA public synthetic documents v1; not pretraining-independent or representative natural corpus'))
    assert len({x['text_sha256'] for x in rows})==len(rows)
    assert len({x['token_sha256'] for x in rows})==len(rows)
    return rows
