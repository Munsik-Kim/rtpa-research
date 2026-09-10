"""Independent, seeded contexts; scorer has no access to ground truth."""
import random,re
import hashlib,json
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
FAMILIES=('KEY_VALUE_RETRIEVAL','LATEST_WRITE_RETRIEVAL')
LENGTHS={'short':(240,256),'long':(1008,1024)}
SEEDS={'CAL1':603101,'CAL2':603201,'EVAL':603301,'TIMING':603401}
TEMPLATES={
 1:('Read the records below. Each record assigns a four-digit value to a key. If a key occurs again, the last record replaces its earlier value. Answer the question with only the four-digit value.\n\n',
    'Records:\nKFOX = 1648\nKMUD = 5921\nKFOX = 3076\nQuestion: What is the final value of KFOX?\nAnswer: 3076\n\n',
    'Records:\n','\nQuestion: What is the final value of {key}?\nAnswer:'),
 2:('Complete the lookup. For repeated keys, use the LAST assignment in the list, not the first or most frequent one. Write just the four digits after Value.\n\n',
    'Assignments:\nKRED = 4281\nKRED = 5962\nKRED = 4281\nKBLUE = 7813\nKRED = 3057\nLookup key: KRED\nValue: 3057\n\n',
    'Assignments:\n','\nLookup key: {key}\nValue:')}

def encode(tokenizer,text):return tokenizer(text,add_special_tokens=False)['input_ids']
def parse_answer(text):
    # Answer field ends at first LF. Raw text/tokens remain in results.
    field=text.split('\n',1)[0].strip()
    if not field:return {'answer':None,'format_status':'EMPTY'}
    if re.fullmatch(r'[1-9][0-9]{3}',field) is None:return {'answer':None,'format_status':'MALFORMED_OR_MULTIPLE'}
    return {'answer':field,'format_status':'VALID'}
def independent_gt(item):
    state={}
    for line in item['record_text'].splitlines():
        key,value=line.split(' = ')
        assert re.fullmatch(r'K[A-Z]{5}',key) and re.fullmatch(r'[1-9][0-9]{3}',value)
        state[key]=value
    return state[item['query_key']]

def generate(tokenizer,family,length,index,role,template):
    seed=SEEDS[role]+10000*FAMILIES.index(family)+1000*list(LENGTHS).index(length)+index
    rng=random.Random(seed);lo,hi=LENGTHS[length];usedk=set();usedv={'1648','5921','3076','4281','5962','7813','3057'}
    role_letter={'CAL1':'A','CAL2':'B','EVAL':'C','TIMING':'D'}[role]
    value_range={'CAL1':(1000,3000),'CAL2':(3000,5000),'EVAL':(5000,9000),'TIMING':(9000,10000)}[role]
    def key():
        while True:
            k='K'+role_letter+''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4))
            if k not in usedk:usedk.add(k);return k
    def value():
        while True:
            v=str(rng.randrange(*value_range))
            if v not in usedv:usedv.add(v);return v
    q=key();correct=value();first=value();other=key();ov=value()
    # Semantic events are always retained; filler only adds independent keys.
    essential=[(q,correct)] if family==FAMILIES[0] else [(q,first),(q,first),(q,first),(q,correct),(other,ov)]
    fillers=[];positions=[rng.random()*.90 for _ in essential]
    if family==FAMILIES[1]:positions=sorted(positions) # final unrelated key follows queried latest write
    head,example,start,end=TEMPLATES[template]
    for _ in range(180):
        records=list(fillers)
        for (event,pos) in zip(essential,positions):records.insert(min(len(records),int(pos*(len(fillers)+len(essential)))),event)
        # Insertion of equal-position events remains temporal by sorting on original event order.
        if family==FAMILIES[1]:
            # Explicit fixed relative positions, preserving first/repeated/latest order.
            tagged=[(rngpos,0,i,r) for i,(rngpos,r) in enumerate(zip([j/(len(fillers)+1) for j in range(len(fillers))],fillers))]
            tagged += [(positions[i],1,i,r) for i,r in enumerate(essential)]
            records=[r for _,_,_,r in sorted(tagged)]
        rt='\n'.join(k+' = '+v for k,v in records)
        text=head+example+start+rt+end.format(key=q)
        ids=encode(tokenizer,text)
        if lo<=len(ids)<=hi:
            item={'item_id':f'rtpa_v06_{role.lower()}_{family.lower()}_{length}_{index}','role':role,'family':family,'length':length,'cell':family+'/'+length,'seed':seed,'template':template,
                  'text':text,'token_ids':ids,'prompt_tokens':len(ids),'record_text':rt,'events':[list(r) for r in records],'query_key':q,'ground_truth':correct,
                  'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'token_sha256':digest(ids),'source_family':'deterministic synthetic key/value record generator','independent_context_count':1}
            assert independent_gt(item)==correct
            if family==FAMILIES[0]:assert sum(k==q for k,v in records)==1
            else:
                values=[v for k,v in records if k==q];assert values[-1]==correct and values[0]!=correct and values.count(first)>values.count(correct)
            return item
        if len(ids)>hi:raise ValueError('GENERATOR_LENGTH_GAP: no truncation permitted')
        fillers.append((key(),value()))
    raise ValueError('GENERATOR_LENGTH_UNREACHED')

def panel(tokenizer,role,n_per_cell,template,families=FAMILIES):
    rows=[generate(tokenizer,f,l,i,role,template) for f in families for l in LENGTHS for i in range(n_per_cell)]
    assert len({r['text_sha256'] for r in rows})==len(rows)==len({r['token_sha256'] for r in rows})
    return rows
