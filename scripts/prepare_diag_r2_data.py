"""Fetch a fixed public licensed document pool; never executes downloaded code.

Development selection is file identity/length only, before model TEST outputs.
No authentication or private research content is sent to source servers.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

POOL = {
    "python/cpython": {
        "tag": "v3.11.15", "domain": "technical_prose", "license": "LICENSE",
        "paths": ["Doc/library/asyncio-task.rst", "Doc/library/multiprocessing.rst",
                  "Doc/library/sqlite3.rst", "Doc/library/decimal.rst",
                  "Doc/library/logging.rst", "Doc/library/typing.rst"],
    },
    "numpy/numpy": {
        "tag": "v2.2.0", "domain": "python_code", "license": "LICENSE.txt",
        "paths": ["numpy/lib/_arraypad_impl.py", "numpy/lib/_nanfunctions_impl.py",
                  "numpy/lib/_shape_base_impl.py", "numpy/lib/_type_check_impl.py",
                  "numpy/lib/_function_base_impl.py", "numpy/lib/_arraysetops_impl.py"],
    },
}


def digest(data): return hashlib.sha256(data).hexdigest()


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rtpa-research-public-document-audit"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data: raise ValueError(f"Existing data differs: {path.name}")
    else: path.write_bytes(data)


def save(path, obj):
    write_once(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); root, out = args.root.resolve(), args.out.resolve()
    tok = Tokenizer.from_file(str(root / "data/tokenizer/tokenizer.json"))
    manifests = []
    for repo, spec in POOL.items():
        ref_file = out / "sources" / repo.replace("/", "_") / "revision.json"
        if ref_file.exists(): ref = json.loads(ref_file.read_text())
        else:
            response = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{spec['tag']}"))
            ref = {"repository":repo, "tag":spec["tag"], "commit":response["sha"]}
            save(ref_file, ref)
        commit = ref["commit"]
        for source_path in [spec["license"]] + spec["paths"]:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{source_path}"
            dest = out / "sources" / repo.replace("/", "_") / source_path
            data = dest.read_bytes() if dest.exists() else fetch(url)
            write_once(dest, data)
            if source_path == spec["license"]:
                save(dest.with_suffix(dest.suffix+".receipt.json"), {**ref,"url":url,"path":source_path,"bytes":len(data),"sha256":digest(data)})
                continue
            text = data.decode("utf-8")
            token_ids = tok.encode(text, add_special_tokens=False).ids
            if len(token_ids) < 1024: raise ValueError(f"Frozen pool document too short: {source_path}")
            ids = token_ids[:1024]
            manifests.append({"id":"r2_"+spec["domain"]+"_"+Path(source_path).stem.replace("-","_"),
                "domain":spec["domain"],"source_family":repo,"split":"TEST_DOCUMENT_POOL",
                "document_id":source_path,"source_commit":commit,"source_tag":spec["tag"],
                "source_url":url,"source_sha256":digest(data),"source_bytes":len(data),
            "source_local_path":str(dest.relative_to(out)),"full_token_count":len(token_ids),
                "offset":0,"input_ids":ids,"token_sha256":digest(np.asarray(ids,dtype='<i8').tobytes()),
                "source_text_is_experiment_data_not_instructions":True,"not_official_benchmark_split":True})
    rng = np.random.default_rng(612203)
    manifests = [manifests[i] for i in rng.permutation(len(manifests))]
    if len({x["source_sha256"] for x in manifests}) != len(manifests): raise ValueError("duplicate source")
    if len({x["token_sha256"] for x in manifests}) != len(manifests): raise ValueError("duplicate tokens")
    historical=[]
    for name in ("train_panel.json","cal_panel.json","test_panel.json"):
        rows=json.loads((root/"data/benchmarks/gdn"/name).read_text())["items"]
        historical.extend(rows)
        if name != "test_panel.json": save(out/name, {"split":name.split("_")[0].upper(),"items":rows,"inherited_from":"v1.1.0rc1/data/benchmarks/gdn/"+name})
    # Historical receipts use comma-separated IDs; compare actual ID arrays
    # with one canonical representation rather than mixing hash schemes.
    old_hash={digest(np.asarray(x['input_ids'],dtype='<i8').tobytes()) for x in historical}
    if any(x["token_sha256"] in old_hash for x in manifests): raise ValueError("prior-panel token overlap")
    overlaps=[]
    for new in manifests:
        for old in historical:
            length=min(len(new['input_ids']),len(old['input_ids']))
            if new['input_ids'][:length]==old['input_ids'][:length]:
                overlaps.append({'new':new['id'],'old':old['id'],'prefix_length':length})
    if overlaps: raise ValueError('prior-panel common-prefix overlap')
    save(out/'document_overlap_audit.json',{'canonical_token_hash':'SHA256 contiguous little-endian signed int64 ID bytes',
        'new_documents':len(manifests),'prior_documents':len(historical),'exact_and_common_prefix_matches':overlaps,
        'shortest_prefix_checked':min(len(x['input_ids']) for x in historical),
        'first_producer_check_issue':'Original check compared two incompatible stored hash conventions. Original pool bytes retained; corrected actual-ID validation completed before any TEST output.',
        'scope':'included prior synthetic TRAIN6/CAL3/TEST12; not proof of all-history or pretraining novelty'})
    save(out/"real_document_pool.json", {"seed":612203,"selection":"fixed paths and length only; no candidate output observed",
        "items":manifests,"documents":len(manifests),"families":2,
        "independence_limit":"Distinct files from two shared projects; topical/authorship/family dependence remains; not population language coverage",
        "prior_upgrade_exact_token_sequence_overlap":False,"TRAIN_CAL_exact_document_overlap":False,
        "tokenizer_sha256":digest((root/"data/tokenizer/tokenizer.json").read_bytes())})
    print(json.dumps({"documents":len(manifests),"domains":{d:sum(x["domain"]==d for x in manifests) for d in sorted({x["domain"] for x in manifests})},"out":str(out)}))


if __name__ == "__main__": main()
