"""Independent CPU audit of included GDN token scalars; no model/primary aggregator.

This verifies scalar observations, not their derivation from full logits. Missing
expectations remain NOT_READY. Failed windows are unresolved, never subset means.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def strict(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): raise ValueError('Nonfinite JSON exponent')
        for child in value.values() if isinstance(value, dict) else value if isinstance(value, list) else (): finite(child)
    finite(result)
    return result


def audit(root, recomputed_dir=None):
    root = Path(root); data = root/'data/benchmarks/gdn'; expected_path = root/'results/upgrade/reproduction_expected.json'
    required = [data/'protocol.json', data/'test_panel.json', expected_path]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing: return {'status': 'NOT_READY', 'missing': missing, 'new_model_forwards': 0, 'expectations_updated': False}
    protocol, manifest, expected = [strict(p.read_text()) for p in required]; panel = manifest['items']; exp = expected['gdn']['summary']
    errors = []; unresolved = []; checked = 0; rows = {}; method_values = {}; bootstrap_checked = False
    def check(name, observed, wanted):
        nonlocal checked
        checked += 1
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool) and isinstance(observed, (int, float)) and not isinstance(observed, bool):
            good = math.isfinite(observed) and math.isfinite(wanted) and (observed == wanted if isinstance(wanted, int) else math.isclose(observed, wanted, rel_tol=1e-10, abs_tol=1e-12))
        else: good = observed == wanted
        if not good: errors.append({'check': name, 'observed': observed, 'expected': wanted})
    check('split', manifest['split'], 'TEST'); check('panel_count', len(panel), protocol['TEST_documents'])
    check('unique_documents', len({p['id'] for p in panel}), len(panel)); check('bootstrap_seed', protocol['bootstrap_seed'], 611204)
    check('bootstrap_count', protocol['bootstrap_draws'], 2000); check('warmup_tokens', protocol['warmup_tokens'], 16); check('expected_planned_documents', exp['planned_documents'], len(panel))
    for item in panel:
        path = data/'tokens'/f"{item['id']}.jsonl.gz"
        if not path.is_file(): return {'status': 'NOT_READY', 'missing': [str(path.relative_to(root))], 'new_model_forwards': 0, 'expectations_updated': False}
        with gzip.open(path, 'rt') as stream: raw = [strict(line) for line in stream if line.strip()]
        index = {(r['method'], r['token']): r for r in raw}; rows[item['id']] = index
        check(item['id']+'/unique_rows', len(index), len(raw))
        keys = {(m, t) for m in protocol['methods'] for t in range(len(item['input_ids']))}
        check(item['id']+'/planned_token_rows', sorted(index), sorted(keys))
        for r in raw:
            t = r['token']; label = f"{item['id']}/{r['method']}/{t}"
            if not isinstance(t, int) or not 0 <= t < len(item['input_ids']): errors.append({'check': label, 'reason': 'TOKEN_RANGE'}); continue
            check(label+'/document', r['document'], item['id']); check(label+'/domain', r['domain'], item['domain'])
            check(label+'/input', r['input_id'], item['input_ids'][t])
            check(label+'/next_target', r['target_id'], item['input_ids'][t+1] if t+1 < len(item['input_ids']) else None)
            if r['status'] not in ('OK', 'NUMERICAL_FAILURE', 'NOT_RUN'): errors.append({'check': label, 'reason': 'INVALID_STATUS'})
            if r['status'] != 'OK': check(label+'/null_KL', r['KL'], None); check(label+'/null_NLL', r['NLL'], None); check(label+'/reason', bool(r['reason']), True)
            for metric in ('KL', 'NLL'):
                if r[metric] is not None: check(label+'/'+metric+'_finite', isinstance(r[metric], (int, float)) and not isinstance(r[metric], bool) and math.isfinite(r[metric]), True)
            if t == len(item['input_ids'])-1: check(label+'/last_NLL', r['NLL'], None)
    domains = [p['domain'] for p in panel]; rng = np.random.default_rng(611204)
    draws = np.concatenate([rng.choice([i for i, d in enumerate(domains) if d == domain], size=(2000, domains.count(domain))) for domain in sorted(set(domains))], axis=1)
    if recomputed_dir is not None:
        bp = Path(recomputed_dir)/'gdn/bootstrap_draws.json'
        if not bp.is_file(): unresolved.append({'scope': 'stored_bootstrap', 'reason': 'DRAW_FILE_MISSING'})
        else:
            stored = strict(bp.read_text()); check('bootstrap_document_order', stored['document_ids'], [p['id'] for p in panel])
            check('stored_bootstrap_seed', stored['seed'], 611204); check('bootstrap_integer_indices', all(type(i) is int for row in stored['draw_indices'] for i in row), True)
            check('bootstrap_domains', stored['domains'], domains); check('bootstrap_draws', stored['draw_indices'], draws.tolist()); bootstrap_checked = True
    else: unresolved.append({'scope': 'stored_bootstrap', 'reason': 'RECOMPUTED_DIRECTORY_NOT_PROVIDED'})
    windows = [w for w in protocol['windows'] if w in (512, 1024)]; check('selected_windows', windows, protocol['windows'])
    for length in windows:
        for method in protocol['methods']:
            values = []
            for item in panel:
                ii = rows[item['id']]; rr = [ii.get((method, t)) for t in range(length)]
                valid = all(r is not None and r['status'] == 'OK' and r['KL'] is not None and (t >= length-1 or r['NLL'] is not None) for t, r in enumerate(rr))
                if valid:
                    kl = [r['KL'] for r in rr[16:]]; nll = [r['NLL'] for r in rr[16:length-1]]
                    values.append((kl, nll))
                else: values.append(None)
            method_values[method, length] = values; q = next(r for r in exp['quality'] if (r['method'], r['context']) == (method, length))
            check(f'{method}/{length}/planned', q['planned_documents'], len(panel)); check(f'{method}/{length}/complete', q['complete_documents'], sum(v is not None for v in values))
            if any(v is None for v in values):
                check(f'{method}/{length}/KL_unresolved', q['mean_KL'], None); check(f'{method}/{length}/NLL_unresolved', q['mean_NLL'], None)
                unresolved.append({'method': method, 'context': length, 'reason': 'INCOMPLETE_OR_FAILED_FULL_PANEL'}); continue
            for name, j, tokens in [('KL', 0, length-16), ('NLL', 1, length-17)]:
                check(f'{method}/{length}/{name}_denominator', q[name+'_tokens'], len(panel)*tokens)
                check(f'{method}/{length}/mean_{name}', math.fsum(x for value in values for x in value[j])/(len(panel)*tokens), q['mean_'+name])
        for c in [r for r in exp['comparisons'] if r['context'] == length and r['same_intervention_scope']]:
            aa, bb = (method_values[m, length] for m in (c['candidate'], c['baseline'])); label = f"{c['candidate']}/{c['baseline']}/{length}"
            check(label+'/planned', c['planned_documents'], len(panel))
            if any(v is None for v in aa+bb):
                for name in ('gain','gain_CI95','candidate_mean_KL','baseline_mean_KL','mean_KL_difference','delta_NLL','delta_NLL_CI95','wins','ties','losses'): check(label+'/'+name+'_unresolved', c.get(name), None)
                continue
            ak, bk = (np.asarray([math.fsum(v[0]) for v in group]) for group in (aa, bb)); an, bn = (np.asarray([math.fsum(v[1]) for v in group]) for group in (aa, bb))
            am, bm = math.fsum(ak)/(len(panel)*(length-16)), math.fsum(bk)/(len(panel)*(length-16))
            delta = [x-y for a, b in zip(aa, bb) for x, y in zip(a[0], b[0])]
            observed = {'candidate_mean_KL': am, 'baseline_mean_KL': bm, 'gain': None if bm <= 0 else 1-am/bm,
                        'mean_KL_difference': am-bm, 'delta_NLL': math.fsum(an-bn)/(len(panel)*(length-17)),
                        'wins': int((ak < bk).sum()), 'ties': int((ak == bk).sum()), 'losses': int((ak > bk).sum()),
                        'paired_KL_tokens': len(panel)*(length-16), 'paired_NLL_tokens': len(panel)*(length-17),
                        'harmful_KL_mass': math.fsum(max(d, 0) for d in delta), 'beneficial_KL_mass': math.fsum(max(-d, 0) for d in delta)}
            for name, value in observed.items(): check(label+'/'+name, value, c[name])
            if bm <= 0: unresolved.append({'contrast': label, 'reason': 'ZERO_OR_NONPOSITIVE_BASELINE'})
            bd = bk[draws].sum(1); gain_ci = None if bool((bd <= 0).any()) else np.quantile(1-ak[draws].sum(1)/bd, [.025, .975]).tolist()
            nll_ci = np.quantile((an-bn)[draws].sum(1)/(len(panel)*(length-17)), [.025, .975]).tolist()
            for name, ci in [('gain_CI95', gain_ci), ('delta_NLL_CI95', nll_ci)]:
                if ci is None: check(label+'/'+name, c[name], None)
                else:
                    for i in range(2): check(label+'/'+name+f'/{i}', ci[i], c[name][i])
    return {'status': 'FAIL' if errors else 'MATCHED_WITH_UNRESOLVED_WINDOWS' if unresolved else 'PASS',
            'scope': 'Independent included token-scalar audit, not full-logit or GPU replication', 'checks': checked,
            'errors': errors, 'unresolved': unresolved, 'stored_bootstrap_checked': bootstrap_checked,
            'expected_sha256': hashlib.sha256(expected_path.read_bytes()).hexdigest(),
            'relative_tolerance': 1e-10, 'absolute_tolerance': 1e-12, 'expectations_updated': False, 'new_model_forwards': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--recomputed-dir', type=Path); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    root = a.root.resolve(); out = a.out.resolve()
    if any(out.is_relative_to(root/name) for name in ('results', 'data', 'configs', 'src')): p.error('Write the audit outside frozen evidence/source paths')
    try: result = audit(root, a.recomputed_dir)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        result = {'status': 'FAIL', 'reason': type(error).__name__+': '+str(error).replace(str(root), '<public_root>'), 'expectations_updated': False, 'new_model_forwards': 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'; out.write_text(text); print(text, end='')
    return 0 if result['status'] in ('PASS', 'MATCHED_WITH_UNRESOLVED_WINDOWS') else 1


if __name__ == '__main__': raise SystemExit(main())
