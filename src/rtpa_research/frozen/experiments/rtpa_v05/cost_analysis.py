"""Requested paired latency contrasts from fixed raw blocks; no new timing runs."""
from .common import *
def run():
    if not (ART/'timing_summary.json').exists():return {'status':'NOT_RUN'}
    raw=read(ART/'timing_samples.json')['rows'];labels=('DAMP8','DIAG8','MATCHED_ENERGY8','DAMP8_REPEAT')
    bad=[r for r in raw if not r['finite_last_logits']]
    if bad:
        result={'status':'UNDEFINED_NONFINITE_TIMING','rows':[],'failed_blocks':bad,'raw_source':receipt(ART/'timing_samples.json'),'ratios_not_used_for_cost_target':True}
        save(ART/'paired_latency_comparisons.json',result);return result
    v={label:np.array([next(r['ms_per_token'] for r in raw if r['label']==label and r['round']==b) for b in (1,2,3,4,5)]) for label in labels}
    noise=float(np.median(np.abs(v['DAMP8_REPEAT']/v['DAMP8']-1)))
    rows=[]
    for a,b in [('DIAG8','MATCHED_ENERGY8'),('DIAG8','DAMP8'),('MATCHED_ENERGY8','DAMP8'),('DAMP8_REPEAT','DAMP8')]:
        ratios=v[a]/v[b];median=float(np.median(ratios));rows.append({'candidate':a,'baseline':b,'paired_rounds':[1,2,3,4,5],
           'raw_ratios':ratios.tolist(),'median_paired_ratio':median,'p95_paired_ratio':float(np.quantile(ratios,.95)),
           'additional_time_fraction':median-1,'within_proposed_5pct_increase':median<=1.05,'absolute_difference_below_repeat_variation':abs(median-1)<=noise})
    result={'status':'COMPLETE','rows':rows,'repeat_median_absolute_fractional_variation':noise,'raw_source':receipt(ART/'timing_samples.json'),
        'all_final_block_logits_finite':True,'cache_initialization_scope':'cache object construction outside timer; native first-token lazy allocation remains inside its unchanged token-step path',
        'cache_object_initialization_excluded':True,'native_lazy_cache_allocation_excluded':False,
        'requested_all_initialization_exclusion_fully_met':False,
        'cost_target_scope':'descriptive target on recorded path only; first-token lazy-allocation limitation retained, no serving or zero-overhead claim',
        'scope':'requested paired comparisons only; no kernel or measurement change; small local repeated-block diagnostic, not serving latency'}
    save(ART/'paired_latency_comparisons.json',result);return result
