"""Small reader-facing tables produced only from recomputed evidence."""
import csv,json
from .io import read,save

def csvwrite(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def render(out):
    allocation=[]
    for v in [4,5]:
        allocation += [{k:r.get(k) for k in ['experiment_id','profile','candidate','baseline','window','N','status','mean_KL_candidate','mean_KL_baseline','gain','gain_CI95','mean_NLL_difference','NLL_difference_CI95','wins','ties','losses','n_KL','n_NLL','harmful_KL_mass','beneficial_KL_mass']} for r in read(out/f'v0{v}_allocation.json')['comparisons']]
    csvwrite(out/'allocation.csv',allocation)
    t=read(out/'task.json')
    csvwrite(out/'task_accuracy.csv',[dict(method=m,**r) for m,r in t['methods'].items()])
    csvwrite(out/'task_paired.csv',t['pairs'])
    d=read(out/'diagnostic.json');csvwrite(out/'diagnostic_items.csv',d['items']);csvwrite(out/'focal.csv',d['focal'])
    csvwrite(out/'readout_by_layer.csv',[dict(layer=k,DIAG_better_items=v,total_items=8,window='last64_prompt',baseline='MATCHED_ENERGY8_FROZEN') for k,v in d['DIAG_readout_wins_last64'].items()])
    c=read(out/'cost.json');csvwrite(out/'cost.csv',[dict(experiment=v,status=c[v]['status'],**r) for v in c for r in c[v]['comparisons']])
    csvwrite(out/'numerical_failures.csv',read(out/'v04_allocation.json')['failures'])
