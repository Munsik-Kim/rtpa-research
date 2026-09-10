"""Fixed, local authored/synthetic FRESH inputs and local model tracing.

This bank is fixed before DEV or fresh policy results are inspected. There is
no candidate pool selected on model performance and no cyclic token padding.
"""
import hashlib
import re
import subprocess
import time
import numpy as np
import torch
from .run import (ROOT, ART, MODEL, REV, LAYERS, DOMAINS, now, read, save,
                  savept, loadpt, receipt, sha, verify, guard, progress, budget)


def bank():
    natural=[
      """In a ceramics studio beside the old canal, a potter named Sera is preparing a set of bowls for the town library. The librarian wants visitors to touch the finished clay during a workshop about everyday materials. Sera begins by placing three dry samples on a wooden bench. One contains fine sand, another includes crushed fired clay, and the third comes from a newly opened bag. She adds the same measured amount of water to each sample and kneads them separately. Her assistant records how long the surfaces remain workable. The sandy mixture cracks when stretched too quickly, while the smoother mixture clings to the board. They agree that the workshop should show both effects instead of presenting one recipe as suitable for every object. After the morning test, Sera trims the foot of a shallow bowl and marks the underside with a small triangular stamp. She explains that the stamp identifies the batch, not the person who will eventually own the bowl. At midday a carpenter arrives with shelves for the drying room. He checks the brackets and asks how far apart the boards should sit. Tall vases need more clearance, but closely spaced shelves make better use of the room for plates. They settle on an adjustable arrangement and keep two spare boards against the wall. Later, Sera reviews the kiln log from last month. A pale patch on one glaze appeared only near the door, where the temperature rose more slowly. She plans another small firing before committing the library bowls to that glaze. The assistant cleans the scales, labels the remaining test pieces, and leaves a space in the notebook for observations made after firing. Before leaving, they cover the wet clay, inspect the ventilation switch, and count the tools on the bench. Tomorrow they will make handles for the workshop cups and check whether the new shelves have remained level overnight.""",
      """The crew of a community radio station meets in a narrow room above a grocery shop to prepare a weekend program about local footpaths. Tomas has brought a portable recorder, a folded map, and several pages of notes from walkers. The producer asks him to separate route descriptions from personal impressions so that listeners can follow the directions without confusing them with recommendations. One walker describes an attractive shortcut through a field, but the access gate is sometimes locked. The crew decides to mention the uncertainty and include the public road as an alternative. A second recording contains a useful account of an old bridge, although traffic noise obscures the speaker's final sentence. Tomas marks the passage for another interview rather than reconstructing the missing words. In the next room, a volunteer checks the microphones with a short passage read at ordinary speaking volume. She notices that one cable crackles when bent near its connector and replaces it before recording begins. During the afternoon the team builds a running order for the program. A brief weather report comes first, followed by the path descriptions, two interviews, and a closing explanation of where updated access notices are posted. Music separates the sections, but it must not cover important directions. The host practices pronouncing several place names and asks an older resident to confirm a name used only on hand painted signs. When the first complete recording is played back, the crew finds that a pause between two directions sounds like the end of the segment. They shorten the pause and listen again. The change makes the route clearer without altering the speaker's words. Before the evening broadcast, Tomas saves the final audio with a dated filename, retains the original interviews, and writes down which edits were made. The producer checks the program length against the available slot and leaves a short margin for the live announcer. They finish by placing the annotated map in a folder for the next volunteer team."""
    ]
    code=[
      '''from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class RainSample:
    station: str
    hour: int
    millimeters: float

def validate_samples(samples):
    seen = set()
    for sample in samples:
        key = (sample.station, sample.hour)
        if key in seen:
            raise ValueError("duplicate station and hour")
        if sample.millimeters < 0:
            raise ValueError("negative rainfall")
        seen.add(key)

def daily_totals(samples):
    validate_samples(samples)
    totals = defaultdict(float)
    observed_hours = defaultdict(set)
    for sample in samples:
        day = sample.hour // 24
        key = (sample.station, day)
        totals[key] += sample.millimeters
        observed_hours[key].add(sample.hour % 24)
    rows = []
    for key in sorted(totals):
        station, day = key
        hours = observed_hours[key]
        rows.append({
            "station": station,
            "day": day,
            "rainfall": totals[key],
            "observations": len(hours),
            "complete": len(hours) == 24,
            "missing_hours": sorted(set(range(24)) - hours),
        })
    return rows

def wettest_complete_day(rows):
    eligible = [row for row in rows if row["complete"]]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-row["rainfall"], row["station"], row["day"]))

def compare_stations(rows, first, second):
    lookup = {(row["station"], row["day"]): row for row in rows}
    days = sorted({row["day"] for row in rows})
    differences = []
    for day in days:
        left = lookup.get((first, day))
        right = lookup.get((second, day))
        if left is None or right is None:
            continue
        if not left["complete"] or not right["complete"]:
            continue
        differences.append((day, left["rainfall"] - right["rainfall"]))
    return differences
''',
      '''from collections import deque

class BookReturnQueue:
    def __init__(self, shelf_capacity):
        if shelf_capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = shelf_capacity
        self.pending = deque()
        self.shelves = {}
        self.received = set()

    def receive(self, book_id, section):
        if book_id in self.received:
            raise ValueError("book already received")
        self.received.add(book_id)
        self.pending.append((book_id, section))

    def shelve_batch(self, limit):
        if limit < 0:
            raise ValueError("negative batch size")
        shelved = []
        deferred = deque()
        inspected = 0
        while self.pending and inspected < limit:
            book_id, section = self.pending.popleft()
            inspected += 1
            shelf = self.shelves.setdefault(section, [])
            if len(shelf) >= self.capacity:
                deferred.append((book_id, section))
                continue
            shelf.append(book_id)
            shelved.append(book_id)
        deferred.extend(self.pending)
        self.pending = deferred
        return tuple(shelved)

    def remove(self, book_id):
        for section in sorted(self.shelves):
            shelf = self.shelves[section]
            if book_id in shelf:
                shelf.remove(book_id)
                self.received.remove(book_id)
                return section
        raise KeyError(book_id)

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "pending": list(self.pending),
            "shelves": {name: list(self.shelves[name]) for name in sorted(self.shelves)},
        }

def restore_queue(snapshot):
    queue = BookReturnQueue(snapshot["capacity"])
    queue.pending = deque(tuple(row) for row in snapshot["pending"])
    queue.shelves = {name: list(books) for name, books in snapshot["shelves"].items()}
    all_books = [book for book, _ in queue.pending]
    all_books.extend(book for books in queue.shelves.values() for book in books)
    if len(all_books) != len(set(all_books)):
        raise ValueError("snapshot contains duplicate books")
    queue.received = set(all_books)
    return queue
'''
    ]
    assoc=[]
    names=['amber','birch','copper','dune','elm','fern','garnet','hazel','indigo','jade','kelp','linen']
    for episode in range(2):
        rng=np.random.default_rng(20260908+episode)
        values=rng.choice(np.arange(1000,9999),size=len(names),replace=False)
        pairs=list(zip(rng.permutation(names),values))
        facts=' '.join(f'{name} stores {value}.' for name,value in pairs)
        queries=' '.join(f'When the label is {pairs[i][0]}, retrieve {pairs[i][1]}.' for i in rng.permutation(len(pairs)))
        assoc.append('Memorize this independent label registry. Each label has one four digit value. '+facts+
                     ' The registry is closed. Use only the associations above for the following recall checks. '+queries+
                     ' The checks preserve the stored assignments. The order of requests does not change a value. End of recall episode.')
    return dict(zip(DOMAINS,[natural,code,assoc]))


def historical_hashes():
    proc=subprocess.run(['rg','-l','"(text|token|tokens)_sha256"','artifacts','configs','-g','*.json'],cwd=ROOT,capture_output=True,text=True)
    if proc.returncode not in (0,1):raise RuntimeError(proc.stderr)
    texts=set();tokens=set();scanned=[];errors=[]
    for rel in sorted(proc.stdout.splitlines()):
        p=ROOT/rel
        if ART in p.parents:continue
        try:
            # Hash-only provenance scan. Historical numerical values are not
            # consulted, and unrelated old JSON conventions do not alter data.
            for key,value in re.findall(r'"(text|token|tokens)_sha256"\s*:\s*"([0-9a-f]{64})"',p.read_text()):
                (texts if key=='text' else tokens).add(value)
            scanned.append(receipt(p))
        except (ValueError,OSError) as exc:errors.append({'path':rel,'error':str(exc)})
    return texts,tokens,scanned,errors


def build_inputs():
    path=ART/'fresh_input_manifest.json'
    if path.exists():
        data=read(path)
        for r in data['sequences']:verify({'path':r['input_path'],'sha256':r['input_sha256']})
        return data['sequences']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    old_texts,old_tokens,scanned,errors=historical_hashes()
    if errors:raise RuntimeError('BLOCKED_FRESH_HASH_SCAN: '+repr(errors))
    rows=[]
    for domain,texts in bank().items():
        for i,text in enumerate(texts):
            raw=tokenizer(text,add_special_tokens=False).input_ids
            if len(raw)<256:raise RuntimeError('BLOCKED_NEW_INPUTS: insufficient contiguous tokens '+domain)
            ids=torch.tensor([raw[:256]],dtype=torch.long)
            text_sha=hashlib.sha256(text.encode()).hexdigest();token_sha=hashlib.sha256(ids.numpy().astype(np.int64).tobytes()).hexdigest()
            if text_sha in old_texts or token_sha in old_tokens:raise RuntimeError('BLOCKED_FRESH_DUPLICATE')
            if any(r['text_sha256']==text_sha or r['token_sha256']==token_sha for r in rows):raise RuntimeError('BLOCKED_FRESH_INTERNAL_DUPLICATE')
            seq=f'rtpa_{domain}_fresh{i}';p=ART/'fresh_inputs'/f'{seq}.pt'
            savept(p,{'input_ids':ids,'text':text})
            rows.append({'sequence_id':seq,'domain':domain,'seed':20260908+i,'provenance':'local authored '+domain if domain!='associative_recall' else 'seeded synthetic label-value recall',
              'raw_token_count':len(raw),'used_token_count':256,'prefix_repetition_padding':False,'text_sha256':text_sha,'token_sha256':token_sha,
              'input_path':str(p.relative_to(ROOT)),'input_sha256':sha(p)})
    save(path,{'created_utc':now(),'mask_receipt_sha256':sha(ART/'masks.json'),'sequences':rows,
       'historical_scan_files':scanned,'historical_unique_text_hashes':len(old_texts),'historical_unique_token_hashes':len(old_tokens),
       'exact_hash_overlap_count':0,'historical_scan_errors':errors,'population_representativeness':'UNKNOWN; 6 local screening inputs',
       'selection':'two prewritten inputs per domain; raw length and hash only; no model results used'})
    return rows


def trace_inputs(sequences):
    path=ART/'fresh_trace_manifest.json'
    rows=read(path)['traces'] if path.exists() else []
    for r in rows:verify(r)
    if len(rows)==18:return rows
    guard('fresh_trace')
    from rsq_gate.model_adapter import load_model
    from rsq_gate.trace import trace_text_sequence
    tick=time.perf_counter();model,_=load_model(MODEL,'cuda');model.requires_grad_(False)
    for seq in sequences:
        budget()
        if sum(r['sequence_id']==seq['sequence_id'] for r in rows)==3:continue
        payload=loadpt(ROOT/seq['input_path']);records,_=trace_text_sequence(model,payload['input_ids'].to('cuda'),list(LAYERS))
        for layer in LAYERS:
            p=ART/'fresh_traces'/f"{seq['sequence_id']}__L{layer}.pt"
            record={n:records[layer][n] for n in ('query','key','value','g','beta','initial_state')}
            savept(p,{'trace':record,'metadata':{'model_revision':REV,'sequence_id':seq['sequence_id'],'layer':layer,'input_sha256':seq['input_sha256']}})
            rows=[r for r in rows if not (r['sequence_id']==seq['sequence_id'] and r['layer']==layer)]
            rows.append({**receipt(p),'sequence_id':seq['sequence_id'],'layer':layer})
        save(path,{'traces':rows,'trace_count':len(rows),'model_revision':REV,'elapsed_including_load':time.perf_counter()-tick})
        progress('fresh_trace',sequence=seq['sequence_id'],completed=len(rows),total=18)
    del model;torch.cuda.empty_cache()
    return rows
