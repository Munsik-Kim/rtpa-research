"""Independent, model-free audit of the final R2 execution/cost ledger.

Does not calculate TEST quality or invoke Torch, CUDA, a model, or excluded
capture PT files. Missing, running, interrupted and retry-uncovered work is
explicitly incomplete, not PASS. Historical and R2 expected hashes stay intact.
This post-freeze publication audit does not retrofit checks into the runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


def strict(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("DUPLICATE_JSON_KEY:"+key)
            value[key] = item
        return value
    def number(s):
        value = float(s)
        if not math.isfinite(value):
            raise ValueError("NONFINITE_JSON_NUMBER")
        return value
    def reject(s):
        raise ValueError("NONFINITE_JSON_LITERAL:"+s)
    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=reject)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class Audit:
    def __init__(self, root, run):
        self.root, self.run = Path(root), Path(run)
        self.errors, self.incomplete, self.sources = [], [], {}
        self.checks = 0

    def require(self, ok, reason):
        self.checks += 1
        if not ok:
            self.errors.append(reason)

    def read(self, relative, required=True, package=False):
        path = (self.root if package else self.run)/relative
        if not path.is_file():
            if required:
                self.incomplete.append("NOT_RUN_OR_NOT_STORED:"+str(relative))
            return None
        self.sources[("package:" if package else "run:")+str(relative)] = digest(path)
        return strict(path.read_text())

    def hash(self, relative, package=False):
        path = (self.root if package else self.run)/relative
        if not path.is_file():
            self.incomplete.append("DEPENDENCY_NOT_AVAILABLE:"+str(relative))
            return None
        result = digest(path)
        self.sources[("package:" if package else "run:")+str(relative)] = result
        return result


def ledger_summary(ledger, bounded_interruptions=None):
    errors, incomplete = [], []
    attempts = ledger.get("attempts", [])
    byphase = defaultdict(list)
    for index, row in enumerate(attempts):
        if (not isinstance(row, dict) or not isinstance(row.get("phase"), str)
                or not integer(row.get("physical_forwards")) or row["physical_forwards"] < 0
                or not nonnegative(row.get("seconds"))):
            errors.append("INVALID_ATTEMPT:"+str(index))
            continue
        byphase[row["phase"]].append({"index": index, **row})
        if row.get("status") != "COMPLETE" and index not in (bounded_interruptions or {}):
            incomplete.append("ATTEMPT_NOT_COMPLETE:"+str(index)+":"+str(row.get("status")))
    forwards = sum(x["physical_forwards"] for rows in byphase.values() for x in rows)
    seconds = math.fsum(x["seconds"] for rows in byphase.values() for x in rows)
    if ledger.get("physical_forwards") != forwards:
        errors.append("TOTAL_FORWARD_LEDGER_SUM_MISMATCH")
    if not nonnegative(ledger.get("GPU_active_seconds")) or abs(ledger["GPU_active_seconds"]-seconds) > 1e-7+1e-12*seconds:
        errors.append("TOTAL_ACTIVE_SECONDS_LEDGER_SUM_MISMATCH")
    if not attempts:
        incomplete.append("NO_GPU_ATTEMPTS")
    if not ledger.get("first_gpu_start_utc"):
        errors.append("FIRST_GPU_START_NOT_RECORDED")
    limit = ledger.get("limit_seconds")
    if not nonnegative(limit) or limit == 0:
        errors.append("GPU_BUDGET_LIMIT_UNAVAILABLE")
    return {"by_phase": dict(byphase), "summed_physical_forwards": forwards, "summed_active_seconds": seconds,
            "within_recorded_budget": nonnegative(limit) and seconds <= limit,
            "errors": errors, "incomplete": incomplete}


def audit_interruptions(a, ledger, protocol, freeze_hash):
    """Reconcile archived durable counts without inventing missing forwards.

    An explicit post-freeze receipt may bound one externally interrupted timing
    attempt. It cannot supply exact missing calls, remove its cost, certify its
    reported external cause, or turn a partial set into the eight-block result.
    """
    validated = {}; seen_paths = set()
    for index, attempt in enumerate(ledger.get("attempts", [])):
        if attempt.get("status") != "INTERRUPTED_EXTERNAL_PROCESS_TERMINATION":
            continue
        error_start = len(a.errors)
        name = attempt.get("interruption_receipt")
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            a.require(False, "INTERRUPTION_RECEIPT_PATH"); continue
        a.require(name.startswith('interrupted_attempts/') and name not in seen_paths,
                  "INTERRUPTION_RECEIPT_UNIQUE_ARCHIVE_PATH")
        seen_paths.add(name)
        receipt = a.read(name)
        if receipt is None: continue
        archive = Path(name).parent
        files = receipt.get('archived_files', [])
        a.require({r.get('path') for r in files} == {'budget.json','progress.json','timing.json','timing_execution.log'}
                  and len(files) == 4, "INTERRUPTION_ARCHIVE_FILE_SET")
        for record in files:
            rel = record.get('path')
            if not isinstance(rel, str) or Path(rel).name != rel:
                a.require(False, "INTERRUPTION_ARCHIVE_UNSAFE_PATH"); continue
            actual = a.hash(str(archive/rel)); path = a.run/archive/rel
            a.require(actual == record.get('sha256') and path.is_file()
                      and path.stat().st_size == record.get('bytes'), "INTERRUPTION_ARCHIVE_BYTES:"+rel)
        old = a.read(str(archive/'budget.json')); partial = a.read(str(archive/'timing.json'))
        progress = a.read(str(archive/'progress.json'))
        if old is None or partial is None or progress is None: continue
        a.require(receipt.get('phase') == attempt.get('phase') == 'timing'
                  and receipt.get('status') == attempt['status']
                  and receipt.get('original_attempt_index') == index,
                  "INTERRUPTION_PHASE_OR_INDEX")
        a.require(receipt.get('exit_code') == 143 and receipt.get('signal') == 15
                  and integer(receipt.get('pid')) and receipt['pid'] > 0,
                  "INTERRUPTION_RECORDED_TERMINATION_IDENTITY")
        a.require(receipt.get('numerical_source_or_protocol_changed') is False,
                  "INTERRUPTION_SOURCE_CHANGE_NOT_SUPPORTED")
        a.require(len(old.get('attempts', [])) == index+1
                  and old['attempts'][:index] == ledger['attempts'][:index]
                  and old.get('first_gpu_start_utc') == ledger.get('first_gpu_start_utc')
                  and old.get('limit_seconds') == ledger.get('limit_seconds'),
                  "INTERRUPTION_ARCHIVED_LEDGER_PREFIX")
        if len(old.get('attempts', [])) <= index: continue
        previous = old['attempts'][index]
        a.require(previous.get('phase') == 'timing' and previous.get('status') == 'RUNNING',
                  "INTERRUPTION_ORIGINAL_ATTEMPT_STATUS")
        old_ls = ledger_summary(old)
        a.require(not old_ls['errors'], "INTERRUPTION_ORIGINAL_LEDGER_SUM")
        a.require(progress.get('phase') == 'timing' and progress.get('status') == 'RUNNING'
                  and progress.get('pid') == receipt['pid'], "INTERRUPTION_PROGRESS_IDENTITY")
        width = protocol['timing_prefix']+protocol['timing_decode']
        labels = protocol['timing_labels']; nlabels = len(labels)
        warmup = nlabels*protocol['timing_warmup_blocks']*width
        rows = partial.get('rows', []); seen = set()
        a.require(partial.get('status') == 'RUNNING' and 0 < len(rows) < nlabels*protocol['timing_measured_blocks'],
                  "INTERRUPTION_PARTIAL_TIMING_SCOPE")
        for ordinal, row in enumerate(rows):
            key = (row.get('block'), row.get('method'))
            a.require(key not in seen and row.get('method') in labels
                      and row.get('block') == ordinal//nlabels and row.get('order') == ordinal%nlabels,
                      "INTERRUPTION_DURABLE_ROW_ORDER")
            seen.add(key)
            a.require(row.get('prompt_tokens') == protocol['timing_prefix']
                      and row.get('decode_tokens') == protocol['timing_decode']
                      and row.get('source_freeze_sha256') == freeze_hash,
                      "INTERRUPTION_DURABLE_ROW_CONTRACT")
        confirmed = warmup+len(rows)*width
        a.require(receipt.get('completed_measured_rows') == len(rows)
                  and receipt.get('calls_per_label') == width and receipt.get('warmup_calls') == warmup
                  and receipt.get('recorded_physical_forwards') == confirmed
                  and previous.get('physical_forwards') == confirmed and attempt.get('physical_forwards') == confirmed,
                  "INTERRUPTION_CONFIRMED_FORWARD_COUNT")
        if rows:
            a.require(receipt.get('last_completed_block') == rows[-1]['block'] == previous.get('block')
                      and receipt.get('last_completed_method') == rows[-1]['method'] == previous.get('method'),
                      "INTERRUPTION_LAST_DURABLE_ROW")
        a.require(receipt.get('physical_forward_count_exact') is False
                  and attempt.get('physical_forward_count_exact') is False,
                  "INTERRUPTION_MUST_NOT_CLAIM_EXACT_COUNT")
        a.require(receipt.get('additional_unrecorded_calls_lower_bound') == attempt.get('additional_unrecorded_calls_lower_bound') == 0
                  and receipt.get('additional_unrecorded_calls_upper_bound') == attempt.get('additional_unrecorded_calls_upper_bound') == width
                  and receipt.get('attempt_physical_forwards_lower_bound') == confirmed
                  and receipt.get('attempt_physical_forwards_upper_bound') == confirmed+width,
                  "INTERRUPTION_UNKNOWN_FORWARD_BOUNDS")
        last_seconds = receipt.get('last_recorded_seconds')
        a.require(nonnegative(last_seconds) and last_seconds == previous.get('seconds') == attempt.get('last_measured_seconds'),
                  "INTERRUPTION_LAST_RECORDED_SECONDS")
        try:
            last_time = datetime.fromisoformat(receipt['last_budget_write_utc'])
            dead_time = datetime.fromisoformat(receipt['confirmed_dead_utc'])
            lag = (dead_time-last_time).total_seconds()
            charge = math.ceil(last_seconds+lag+1)
            a.require(lag >= 0 and receipt.get('charged_seconds_upper_bound') == charge == attempt.get('seconds')
                      and attempt.get('seconds_are_conservative_upper_charge') is True,
                      "INTERRUPTION_CONSERVATIVE_WALL_CHARGE")
        except (KeyError, ValueError, TypeError, OverflowError):
            a.require(False, "INTERRUPTION_WALL_BOUND_SCHEMA"); charge = None
        if len(a.errors) == error_start:
            validated[index] = {'attempt_index':index,'phase':'timing','receipt':name,
                                'confirmed_forwards':confirmed,'additional_unknown_forwards':[0,width],
                                'attempt_forwards_interval':[confirmed,confirmed+width],
                                'physical_forward_count_exact':False,'charged_seconds_upper_bound':charge,
                                'last_recorded_seconds':last_seconds,'partial_measured_rows':len(rows),
                                'cause_status':'Recorded external termination; exact cause is UNKNOWN and not independently reproduced',
                                'partial_samples_used_as_final_timing':False}
    if seen_paths:
        a.require(len(seen_paths) == 1, "MULTIPLE_TIMING_INTERRUPTION_RECEIPTS_REQUIRE_NEW_REVIEW")
    if validated:
        upper = sum(x['additional_unknown_forwards'][1] for x in validated.values())
        a.require(ledger.get('physical_forwards_are_confirmed_lower_bound') is True
                  and ledger.get('additional_unrecorded_physical_forwards_upper_bound') == upper,
                  "INTERRUPTION_GLOBAL_LOWER_BOUND_SCOPE")
        first = min(validated)
        retries = [r for r in ledger['attempts'][first+1:] if r.get('phase') == 'timing']
        a.require(len(retries) <= 1, "MORE_THAN_ONE_TIMING_RETRY")
    elif ledger.get('additional_unrecorded_physical_forwards_upper_bound', 0):
        a.require(False, "UNVALIDATED_GLOBAL_UNRECORDED_FORWARD_BOUND")
    return validated


def audit_terminated_timing(a, ledger, protocol, freeze_hash, interruptions):
    """Validate the final closed concurrency failure, not timing completion.

    This intentionally supports only the recorded single full retry stopped by
    the existing after-block GPU-concurrency guard. No partial ratios/CI are
    computed and no missing block or failed receipt is fabricated.
    """
    candidates = [(i,r) for i,r in enumerate(ledger.get('attempts', []))
                  if r.get('phase') == 'timing' and r.get('status') == 'FAILED']
    if not candidates: return None
    error_start = len(a.errors)
    if len(candidates) != 1:
        a.require(False,'TERMINATED_TIMING_FAILURE_ATTEMPT_COUNT'); return None
    index, attempt = candidates[0]
    lifecycle = a.read('timing_retry_lifecycle.json')
    partial = a.read('timing.json')
    paths = sorted((a.run/'phase_failures').glob('timing_*.json'))
    failures = []
    for path in paths:
        item = a.read(str(path.relative_to(a.run)))
        if item is not None: failures.append((str(path.relative_to(a.run)),item))
    a.require(len(failures) == 1,'TERMINATED_TIMING_FAILURE_RECEIPT_COUNT')
    if lifecycle is None or partial is None or len(failures) != 1: return None
    failure_path, failure = failures[0]
    a.require(failure.get('phase') == 'timing' and failure.get('method') is None
              and failure.get('reason') == 'TIMING_CONFOUNDED_OTHER_GPU_WORKER'
              and failure.get('exception') == 'RuntimeError'
              and failure.get('status') == 'FAILED_NOT_SILENTLY_RESUMED'
              and failure.get('completed_observations_preserved') is True,
              'TERMINATED_TIMING_FAILURE_REASON')
    a.require(lifecycle.get('status') == 'FAILED_CHILD_EXIT' and lifecycle.get('return_code') == 1
              and lifecycle.get('registered_phase') == 'timing' and lifecycle.get('attempt_number') == 2
              and lifecycle.get('source_freeze_sha256') == freeze_hash
              and lifecycle.get('numerical_source_changed') is False
              and lifecycle.get('checkpoint_or_policy_reselection') is False,
              'TERMINATED_TIMING_LIFECYCLE_IDENTITY')
    a.require(integer(lifecycle.get('GPU_worker_pid')) and lifecycle['GPU_worker_pid'] > 0
              and integer(lifecycle.get('supervisor_pid')) and lifecycle['supervisor_pid'] > 0
              and lifecycle['GPU_worker_pid'] != lifecycle['supervisor_pid'],
              'TERMINATED_TIMING_PROCESS_IDENTITIES')
    timing_attempts = [i for i,r in enumerate(ledger['attempts']) if r.get('phase') == 'timing']
    a.require(len(interruptions) == 1 and len(timing_attempts) == 2 and index == timing_attempts[-1]
              and min(interruptions) == timing_attempts[0], 'TERMINATED_TIMING_SINGLE_AUTHORIZED_RETRY')
    if interruptions:
        archived_timing = str(Path(next(iter(interruptions.values()))['receipt']).parent/'timing.json')
        a.require(lifecycle.get('first_partial_timing') == archived_timing,'TERMINATED_TIMING_PRIOR_ARCHIVE_LINK')
    width = protocol['timing_prefix']+protocol['timing_decode']; labels = protocol['timing_labels']; nlabels = len(labels)
    warmup = nlabels*protocol['timing_warmup_blocks']*width
    rows = partial.get('rows', []); seen = set()
    a.require(partial.get('status') == 'RUNNING' and 0 < len(rows) < nlabels*protocol['timing_measured_blocks'],
              'TERMINATED_TIMING_STALE_PARTIAL_STATUS')
    for ordinal,row in enumerate(rows):
        key = (row.get('block'),row.get('method'))
        a.require(key not in seen and row.get('method') in labels
                  and row.get('block') == ordinal//nlabels and row.get('order') == ordinal%nlabels,
                  'TERMINATED_TIMING_DURABLE_ROW_ORDER')
        seen.add(key)
        a.require(row.get('prompt_tokens') == protocol['timing_prefix']
                  and row.get('decode_tokens') == protocol['timing_decode']
                  and row.get('source_freeze_sha256') == freeze_hash,
                  'TERMINATED_TIMING_DURABLE_ROW_CONTRACT')
        a.require(row.get('isolated_before') is True,'TERMINATED_TIMING_BEFORE_ISOLATION')
        if ordinal < len(rows)-1:
            a.require(row.get('isolated_after') is True,'TERMINATED_TIMING_EARLIER_AFTER_ISOLATION')
    extra_pids = []
    if rows:
        last = rows[-1]; after = last.get('isolation_after', {})
        extra_pids = [r.get('pid') for r in after.get('processes', [])
                      if integer(r.get('pid')) and r['pid'] != lifecycle.get('GPU_worker_pid')]
        a.require(last.get('isolated_after') is False and after.get('status_ok') is True
                  and after.get('no_competing_GPU_worker') is False and bool(extra_pids),
                  'TERMINATED_TIMING_LAST_ROW_CONCURRENCY_EVIDENCE')
    confirmed = warmup+len(rows)*width
    a.require(attempt.get('physical_forwards') == confirmed,'TERMINATED_TIMING_CLOSED_FORWARD_COUNT')
    a.require(nonnegative(attempt.get('seconds')) and nonnegative(lifecycle.get('supervisor_wall_seconds')),
              'TERMINATED_TIMING_CLOCK_FIELDS')
    if len(a.errors) != error_start: return None
    return {'attempt_index':index,'status':'VALIDATED_CLOSED_PARTIAL_TIMING_NOT_COMPLETE',
            'failure_receipt':failure_path,'lifecycle_receipt':'timing_retry_lifecycle.json',
            'physical_forwards':confirmed,'this_attempt_forward_count_exact':True,
            'stored_timing_status':partial['status'],'planned_measured_rows':nlabels*protocol['timing_measured_blocks'],
            'completed_measured_rows':len(rows),'warmup_forwards':warmup,
            'extra_GPU_PIDs_at_failed_boundary':extra_pids,'final_timing_complete':False,
            'latency_ratio_or_CI_from_partial_rows':'NOT_COMPUTED',
            'runner_monotonic_seconds':attempt['seconds'],'supervisor_epoch_wall_seconds':lifecycle['supervisor_wall_seconds'],
            'clock_difference_seconds':attempt['seconds']-lifecycle['supervisor_wall_seconds'],
            'clock_scope':'Different recorded clocks; discrepancy retained, cause UNKNOWN, no ledger adjustment',
            'reported_external_process_state_independently_reproduced':False}


def allocator_summary(events, outputs):
    """Independently reconstruct allocation lifetimes, not address ownership.

    A recycled address can hold an earlier scratch allocation and the final
    returned tensor. Only the latter lifetime is excluded from scratch peaks.
    free_requested does not end a live allocation; free_completed does.
    """
    active = {}; lifetimes = []
    for step, event in enumerate(events):
        action, address = event["action"], event.get("addr")
        if action == "alloc":
            if address in active or not integer(event.get("size")) or event["size"] < 0:
                raise ValueError("INVALID_ALLOCATOR_LIFETIME")
            active[address] = len(lifetimes)
            lifetimes.append({"address": address, "start": step, "end": len(events), "bytes": event["size"]})
        elif action == "free_completed" and address in active:
            lifetimes[active.pop(address)]["end"] = step
    final_outputs = {active[address] for address in outputs if address in active}
    boundaries = defaultdict(lambda: [0, 0])
    for index, value in enumerate(lifetimes):
        size = value["bytes"]; scratch = 0 if index in final_outputs else size
        boundaries[value["start"]][0] += size; boundaries[value["start"]][1] += scratch
        boundaries[value["end"]][0] -= size; boundaries[value["end"]][1] -= scratch
    allocated = temporary = peak = scratch_peak = 0
    for step in sorted(boundaries):
        allocated += boundaries[step][0]; temporary += boundaries[step][1]
        peak = max(peak, allocated); scratch_peak = max(scratch_peak, temporary)
    return {"max_new_allocated_from_trace": peak,
            "max_transient_excluding_returned_storage_from_trace": scratch_peak,
            "allocation_lifetimes": len(lifetimes), "retained_output_allocation_lifetimes": len(final_outputs),
            "address_reuse_corrected": True}


def audit_timing(a, protocol, freeze_hash, terminated=None):
    obj = a.read("timing.json")
    labels = protocol["timing_labels"]; blocks = protocol["timing_measured_blocks"]
    width = protocol["timing_prefix"]+protocol["timing_decode"]
    planned_warmup = len(labels)*protocol["timing_warmup_blocks"]*width
    planned_measured = len(labels)*blocks*width
    result = {"status": "NOT_RUN", "planned_warmup_forwards": planned_warmup,
              "planned_measured_forwards": planned_measured, "planned_total_forwards": planned_warmup+planned_measured,
              "receipt_forwards": None, "warmup_scope": "No individual warmup time samples; reconcile counts from complete total, not measured rows alone"}
    if obj is None:
        return result
    seen = set(); orders = set(); measured = 0; isolation = True
    for row in obj.get("rows", []):
        key = (row.get("method"), row.get("block")); order = (row.get("block"), row.get("order"))
        a.require(key not in seen and order not in orders, "TIMING_DUPLICATE_BLOCK_OR_ORDER")
        seen.add(key); orders.add(order)
        a.require(key[0] in labels and integer(key[1]) and 0 <= key[1] < blocks, "TIMING_LABEL_OR_BLOCK")
        a.require(row.get("prompt_tokens") == protocol["timing_prefix"] and row.get("decode_tokens") == protocol["timing_decode"], "TIMING_WORKLOAD_CHANGED")
        a.require(row.get("source_freeze_sha256") == freeze_hash, "TIMING_SOURCE_FREEZE")
        for name in ("prefill_seconds", "TTFT_seconds", "decode_seconds", "decode_ms_per_token", "tokens_per_second"):
            a.require(nonnegative(row.get(name)) and row[name] > 0, "TIMING_INVALID_VALUE:"+name)
        if nonnegative(row.get("decode_seconds")) and row["decode_seconds"] > 0:
            seconds = row["decode_seconds"]
            a.require(math.isclose(row["decode_ms_per_token"], 1000*seconds/protocol["timing_decode"], rel_tol=1e-12, abs_tol=1e-12), "TIMING_MS_ARITHMETIC")
            a.require(math.isclose(row["tokens_per_second"], protocol["timing_decode"]/seconds, rel_tol=1e-12, abs_tol=1e-12), "TIMING_THROUGHPUT_ARITHMETIC")
        measured += width
        isolation &= row.get("isolated_before") is True and row.get("isolated_after") is True
    expected = {(label, block) for label in labels for block in range(blocks)}
    complete = seen == expected and obj.get("status") == "COMPLETE"
    if not complete:
        a.incomplete.append("TIMING_PLANNED_BLOCKS_OR_FINAL_STATUS_INCOMPLETE")
    total = obj.get("physical_forwards")
    if terminated is not None:
        # The unchanged runner never updates the partial timing.json status
        # after its guard raises. Its finally-closed budget and failure records
        # supply the count only, not missing timing rows or cost estimates.
        a.require(total is None or total == terminated['physical_forwards'], 'TERMINATED_TIMING_CONTRADICTORY_TOTAL')
        total = terminated['physical_forwards']
    if total is not None:
        a.require(integer(total) and total >= measured, "TIMING_TOTAL_FORWARD_COUNT")
        if complete:
            a.require(total == planned_warmup+planned_measured, "TIMING_TOTAL_VS_FROZEN_WARMUP_AND_MEASURED")
    elif complete:
        a.incomplete.append("TIMING_TOTAL_FORWARD_RECEIPT_MISSING")
    result.update(status="COMPLETE_COUNT_RECEIPT" if complete else "INCOMPLETE", receipt_forwards=total,
                  observed_measured_rows=len(obj.get("rows", [])), observed_measured_forwards=measured,
                  inferred_unmeasured_warmup_forwards=total-measured if integer(total) else None,
                  warmup_count_verified_by_total_only=bool(complete and total == planned_warmup+planned_measured),
                  isolation_at_recorded_boundaries=isolation,
                  isolation_limit="Before/after process samples, not continuous or complete Windows visibility",
                  metric_logging_in_timed_region=False if all(r.get("metric_scalar_logging_in_timed_region") is False for r in obj.get("rows", [])) else "UNVERIFIED",
                  mandatory_guard_host_synchronization_retained=True if all(r.get("mandatory_guard_host_sync") is True for r in obj.get("rows", [])) else "UNVERIFIED",
                  lazy_allocation_included=obj.get("forward_lazy_allocation_included", "UNREPORTED"),
                  timing_comparison_or_cost_target_evaluated=False)
    if terminated is not None:
        result.update(status='TERMINATED_PARTIAL_TIMING_NOT_COMPLETE',
                      receipt_count_source='Finally-closed FAILED budget + phase failure + supervisor lifecycle + durable rows',
                      this_attempt_forward_count_exact=True, final_timing_complete=False,
                      original_raw_status_unchanged=obj.get('status'),
                      unmeasured_warmup_reconciled_from_closed_total_only=total-measured==planned_warmup,
                      latency_ratio_or_CI_from_partial_rows='NOT_COMPUTED')
    return result


def audit_memory(a, protocol, freeze_hash):
    measurements = []; phases = {}; pids = []
    heads = len(protocol["method_layers"]["R2_DIAG"])*16
    mixed = 120*(128+4*4)+8*128*2
    for method in protocol["memory_labels"]:
        name = "memory/"+method+".json"; row = a.read(name)
        if row is None:
            measurements.append({"method": method, "status": "NOT_RUN_OR_NOT_STORED"}); continue
        expected_forwards = protocol["memory_prefix"]+protocol["memory_decode"]
        phases["memory_"+method] = row.get("physical_forwards")
        a.require(row.get("method") == method and row.get("source_freeze_sha256") == freeze_hash, "MEMORY_IDENTITY:"+method)
        a.require(row.get("prompt_tokens") == protocol["memory_prefix"] and row.get("decode_tokens") == protocol["memory_decode"], "MEMORY_WORKLOAD:"+method)
        a.require(row.get("physical_forwards") == expected_forwards, "MEMORY_FORWARD_COUNT:"+method)
        a.require(integer(row.get("pid")) and row["pid"] > 0, "MEMORY_PID:"+method)
        if row.get("fresh_process_for_method") is not True:
            a.incomplete.append("FRESH_MEMORY_PROCESS_NOT_RECORDED:"+method)
        pids.append(row.get("pid"))
        per_head = 128*128*2 if method == "NATIVE" else mixed
        a.require(row.get("payload_bytes") == heads*per_head and row.get("payload_bytes_per_head") == per_head, "MEMORY_PAYLOAD_ARITHMETIC:"+method)
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "allocated_after_model_load", "steady_allocated_bytes", "cache_tensor_bytes", "model_parameter_bytes", "static_policy_bytes"):
            a.require(integer(row.get(key)) and row[key] >= 0, "MEMORY_BYTE_FIELD:"+method+":"+key)
        static = row.get("static_policy_bytes_by_type", {})
        a.require(sum(static.values()) == row.get("static_policy_bytes"), "MEMORY_STATIC_SUM:"+method)
        a.require(row["peak_reserved_bytes"] >= row["peak_allocated_bytes"], "MEMORY_RESERVED_BELOW_ALLOCATED:"+method)
        a.require(row["cache_tensor_bytes"] >= row["payload_bytes"], "MEMORY_CACHE_BELOW_PAYLOAD:"+method)
        scratch = row.get("codec_scratch_measurement")
        scratch_results = []
        if method.startswith("R2") or method.startswith("DAMP_R2"):
            if not scratch:
                a.incomplete.append("R2_SCRATCH_RECEIPT_NOT_STORED:"+method)
            else:
                rel = Path(scratch["file"])
                if rel.name != str(rel) or rel.name in {".", ".."}:
                    raise ValueError("UNSAFE_ALLOCATION_RECEIPT_PATH")
                allocation = a.read("memory/"+str(rel))
                a.require(a.hash("memory/"+str(rel)) == scratch["sha256"], "ALLOCATION_RECEIPT_HASH:"+method)
                if allocation is not None:
                    a.require(allocation.get("no_model_forward_in_this_codec_only_trace") is True, "ALLOCATOR_FORWARD_SCOPE:"+method)
                    raw = allocation.get("records", [])
                    a.require(sorted(r["operation"] for r in raw) == ["decode", "encode"], "ALLOCATOR_OPERATION_SET:"+method)
                    a.require(len(scratch.get("records", [])) == len(raw), "ALLOCATOR_SUMMARY_ROWS:"+method)
                    for index, record in enumerate(raw):
                        calculated = allocator_summary(record["trace_entries"], record["output_storage_addresses"])
                        for key, value in calculated.items():
                            a.require(record.get(key) == value, "ALLOCATOR_RAW_SUMMARY:"+method+":"+record["operation"]+":"+key)
                        reduced = {k:v for k,v in record.items() if k not in {"trace_entries", "output_storage_addresses"}}
                        if index < len(scratch.get("records", [])):
                            a.require(reduced == scratch["records"][index], "ALLOCATOR_PARENT_SUMMARY:"+method+":"+record["operation"])
                        expected_output = mixed*16 if record["operation"] == "encode" else 16*128*128*4
                        a.require(record["retained_output_tensor_bytes"] == expected_output, "ALLOCATOR_OUTPUT_BYTES:"+method+":"+record["operation"])
                        a.require(nonnegative(record["peak_increment_bytes"]), "ALLOCATOR_NEGATIVE_PEAK_INCREMENT:"+method)
                        scratch_results.append({"operation": record["operation"], "status": "RAW_LIFETIME_RECONSTRUCTION_CHECKED", **calculated,
                                                "observed_peak_increment_bytes": record["peak_increment_bytes"],
                                                "allocator_alignment_and_preexisting_lifetimes_preclude_equating_every_size_measure": True})
        measurements.append({"method": method, "status": "MEMORY_RECEIPT_CHECKED", "pid": row.get("pid"),
                             "payload_bytes": row.get("payload_bytes"), "per_head_bytes": per_head,
                             "physical_forwards": row.get("physical_forwards"), "scratch": scratch_results,
                             "process_memory_scope": "One recorded process/method and cache; not independently observed process launch",
                             "source": name})
    unique = len(pids) == len(set(pids))
    if not unique:
        a.incomplete.append("MEMORY_PID_REUSE_REQUIRES_PROCESS_LIFECYCLE_REVIEW")
    return {"measurements": measurements, "distinct_recorded_method_PIDs": unique,
            "PID_limit": "Distinct recorded PIDs support process separation; an explicit fresh-process flag alone is not runtime proof",
            "planned_heads": heads, "mixed_bytes_per_head": mixed, "mixed_bits_per_value": 8*mixed/(128*128)}, phases


def completed_evaluation_receipt(receipt, stored_rows, expected_rows):
    """Completed accounting includes preserved scientific numerical failures."""
    return stored_rows == expected_rows and receipt.get("status") in {
        "COMPLETE", "COMPLETE_WITH_RETAINED_FAILURES"}


def audit_installed_api(a, api, protocol, protocol_hash, cal_panel):
    """Check the frozen example's child receipt, not a newly invented schema."""
    records = api.get("records", [])
    a.require({r["method"] for r in records} == {"NATIVE", "R2_DIAG"}
              and len(records) == 2, "INSTALLED_API_METHOD_SET")
    a.require(api.get("policy_sha256") == protocol["policy_sha256"]
              and api.get("protocol_sha256") == protocol_hash, "INSTALLED_API_INPUT_HASHES")
    a.require(api.get("model_revision") == protocol["model_revision"]
              and api.get("profile") == protocol["codec_revision"]
              and api.get("layers") == protocol["method_layers"]["R2_DIAG"],
              "INSTALLED_API_MODEL_PROFILE_LAYERS")
    a.require(api.get("model_downloads") == 0, "INSTALLED_API_OFFLINE_RECEIPT")
    if cal_panel is not None:
        item = cal_panel["items"][0]
        a.require(cal_panel.get("split") == "CAL" and api.get("document") == item["id"]
                  and api.get("input_ids") == item["input_ids"][:32]
                  and len(api.get("input_ids", [])) == 32, "INSTALLED_API_FIXED_CAL_PREFIX")
    count = 0
    for row in records:
        method = row["method"]
        attempted, completed = row.get("attempted_token_forwards"), row.get("finite_completed_token_forwards")
        a.require(row.get("planned_token_forwards") == 32, "INSTALLED_API_FIXED32_PLAN")
        valid_counts = integer(attempted) and integer(completed) and 0 <= completed <= attempted <= 32
        a.require(valid_counts, "INSTALLED_API_ATTEMPT_COUNTS:"+method)
        if integer(attempted): count += attempted
        a.require(row.get("not_a_latency_measurement") is True
                  and nonnegative(row.get("elapsed_seconds_including_finite_checks")),
                  "INSTALLED_API_TIME_SCOPE:"+method)
        if row.get("status") == "COMPLETE_FINITE":
            a.require(attempted == completed == 32 and row.get("failure") is None,
                      "INSTALLED_API_COMPLETE_COUNTS:"+method)
            if method == "R2_DIAG":
                writes = row.get("successful_R2_layer_writes", {})
                a.require(set(map(int, writes)) == set(protocol["method_layers"]["R2_DIAG"])
                          and all(integer(v) and v == 32 for v in writes.values()), "INSTALLED_API_LAYER_WRITES")
                expected_payload = len(protocol["method_layers"]["R2_DIAG"])*16*19328
                a.require(row.get("actual_R2_payload_bytes") == expected_payload, "INSTALLED_API_R2_PAYLOAD")
        elif row.get("status") == "NUMERICAL_FAILURE":
            failure = row.get("failure")
            a.require(valid_counts and attempted == completed+1 and isinstance(failure, dict)
                      and failure.get("token") == completed and bool(failure.get("reason")),
                      "INSTALLED_API_FAILURE_COUNTS:"+method)
            a.incomplete.append("INSTALLED_API_RETAINED_FAILURE:"+method)
        else:
            a.require(False, "INSTALLED_API_UNKNOWN_STATUS:"+method)
        if method == "NATIVE":
            a.require(row.get("successful_R2_layer_writes") is None
                      and row.get("actual_R2_payload_bytes") is None,
                      "INSTALLED_API_NATIVE_COUNTER_SCOPE")
    all_complete = len(records) == 2 and all(r.get("status") == "COMPLETE_FINITE" for r in records)
    expected_status = "CAL_API_SMOKE_COMPLETE" if all_complete else "CAL_API_SMOKE_WITH_RETAINED_FAILURE"
    a.require(api.get("status") == expected_status, "INSTALLED_API_TOP_LEVEL_STATUS")
    a.require(count == api.get("total_attempted_token_forwards"), "INSTALLED_API_TOTAL_FORWARD_SUM")
    return count


def audit_run(root, run, api_receipt=None):
    a = Audit(root, run)
    protocol, ledger, freeze = a.read("protocol.json"), a.read("budget.json"), a.read("freeze.json")
    if protocol is None or ledger is None or freeze is None:
        return {"status": "INCOMPLETE", "errors": a.errors, "incomplete": a.incomplete, "GPU_forwards": 0}
    fhash = a.hash("freeze.json")
    interruptions = audit_interruptions(a, ledger, protocol, fhash)
    terminated = audit_terminated_timing(a, ledger, protocol, fhash, interruptions)
    resolved_closed_attempts = {**interruptions}
    if terminated is not None: resolved_closed_attempts[terminated['attempt_index']] = terminated
    ls = ledger_summary(ledger, resolved_closed_attempts); a.errors.extend(ls["errors"]); a.incomplete.extend(ls["incomplete"])
    a.require(ledger.get("limit_seconds") == 14400, "RECORDED_GPU_BUDGET_CHANGED")
    a.require(ls["within_recorded_budget"], "RECORDED_GPU_BUDGET_EXCEEDED")
    phase_receipts = {}; receipt_scope = {}
    obj = a.read("cuda_codec_fixture.json")
    if obj: phase_receipts["cuda_fixture"] = obj.get("model_forwards")
    obj = a.read("codec_cal_trajectories.json")
    if obj:
        rows = obj.get("rows", [])
        a.require(len({(r["document"],r["method"]) for r in rows}) == len(rows), "CAL_TRAJECTORY_DUPLICATE")
        phase_receipts["pilot"] = sum(r["forwards"] for r in rows)
    train = a.read("train_panel.json"); capture_calls = 0; captures_complete = train is not None
    if train:
        for item in train["items"]:
            if Path(item["id"]).name != item["id"] or item["id"] in {".", ".."}:
                raise ValueError("UNSAFE_TRAIN_DOCUMENT_ID")
            capture = a.read("capture/"+item["id"]+".json")
            if capture is None:
                captures_complete = False; continue
            a.require(capture.get("input_sha256") == item["token_sha256"], "CAPTURE_INPUT_IDENTITY:"+item["id"])
            a.require(capture.get("status") == "COMPLETE", "CAPTURE_RECEIPT_NOT_COMPLETE:"+item["id"])
            a.require(capture.get("physical_forwards") == len(item["input_ids"]), "CAPTURE_FORWARD_COUNT:"+item["id"])
            capture_calls += capture["physical_forwards"]
    if captures_complete: phase_receipts["capture"] = capture_calls
    fit = a.read("calibration_cost.json")
    if fit:
        a.require(sorted(r["layer"] for r in fit["layers"]) == protocol["method_layers"]["R2_DIAG"], "FIT_LAYER_COVERAGE")
        phase_receipts["fit"] = 0
        receipt_scope["fit"] = "Offline source propagation consumes captured operands; no language-model forwards; GPU/time remains in ledger"
    obj = a.read("reference_profile.json")
    if obj:
        phase_receipts["profile"] = obj.get("physical_forwards")
        a.require(obj.get("physical_forwards") == obj.get("tokens", 0)+obj.get("prefill_tokens", 0), "PROFILE_PREFIX_PLUS_OBSERVED_CALLS")
        receipt_scope["profile"] = "Diagnostic profile, not serving latency; large profiler trace is excluded and not required by this count audit"
    obj = a.read("conformance.json")
    if obj:
        phase_receipts["conformance"] = obj.get("physical_forwards")
        listed = sum(r["physical_forwards"] for r in obj.get("rows", []))
        extra = obj["physical_forwards"]-listed
        # Frozen source explicitly runs two (64+2)-token causality branches and
        # two 32-token observer branches in addition to the named trajectories.
        expected_extra = 2*(64+2)+2*32
        a.require(extra == expected_extra, "CONFORMANCE_UNLISTED_SUBCHECK_FORWARD_RECONCILIATION")
        receipt_scope["conformance"] = {"trajectory_receipt_forwards": listed, "additional_bounded_subchecks": extra,
                                       "subcheck_scope": "Reconstructed from frozen source loop counts, not separate per-subcheck receipts"}
    obj = a.read("final_probe.json")
    if obj:
        phase_receipts["final_probe"] = obj.get("physical_forwards")
        a.require(obj.get("physical_forwards") == 2*obj.get("tokens_per_method", -1), "FINAL_PROBE_TWO_METHOD_CALLS")
    panel = a.read("test_panel.json"); eval_calls = 0; complete_docs = 0; raw_status_counts = Counter()
    if panel:
        for item in panel["items"]:
            if Path(item["id"]).name != item["id"] or item["id"] in {".", ".."}:
                raise ValueError("UNSAFE_TEST_DOCUMENT_ID")
            name = "tokens/"+item["id"]+".jsonl.gz"; receipt_name = name[:-3]+".receipt.json"
            receipt = a.read(receipt_name)
            if receipt is None:
                continue
            actual_hash = a.hash(name)
            a.require(actual_hash == receipt.get("sha256") and receipt.get("source_freeze_sha256") == fhash,
                      "TOKEN_RECEIPT_HASH_OR_FREEZE:"+item["id"])
            if actual_hash is None:
                continue
            counts = Counter(); keys = set(); stored = 0
            with gzip.open(a.run/name, "rt") as stream:
                for line in stream:
                    if not line.strip(): continue
                    row = strict(line); stored += 1
                    key = (row["method"], row["token"])
                    a.require(key not in keys, "ACCOUNTING_DUPLICATE_TOKEN_KEY:"+item["id"])
                    keys.add(key); counts[row["status"]] += 1
                    a.require(row["status"] in {"OK", "NUMERICAL_FAILURE", "NOT_RUN"}, "UNKNOWN_TOKEN_STATUS")
                    # Deliberately no access to KL/NLL or scientific effect.
            physical = counts["OK"]+counts["NUMERICAL_FAILURE"]
            a.require(receipt.get("rows") == stored and receipt.get("physical_forwards") == physical,
                      "TOKEN_RECEIPT_ROWS_OR_PHYSICAL:"+item["id"])
            expected_rows = len(item["input_ids"])*len(protocol["methods"])
            if not completed_evaluation_receipt(receipt, stored, expected_rows):
                a.incomplete.append("DOCUMENT_ACCOUNTING_INCOMPLETE:"+item["id"])
            else: complete_docs += 1
            eval_calls += physical; raw_status_counts.update(counts)
        if complete_docs != protocol["TEST_documents"]:
            a.incomplete.append("QUALITY_FORWARD_COVERAGE_INCOMPLETE")
        phase_receipts["evaluate"] = eval_calls
        receipt_scope["evaluate"] = {"planned_documents": protocol["TEST_documents"], "completed_receipts": complete_docs,
                                     "planned_no_failure_calls": sum(len(i["input_ids"]) for i in panel["items"])*len(protocol["methods"]),
                                     "status_counts": dict(raw_status_counts), "metric_values_analyzed": False}
    timing = audit_timing(a, protocol, fhash, terminated)
    if integer(timing["receipt_forwards"]): phase_receipts["timing"] = timing["receipt_forwards"]
    if interruptions:
        archived_calls = sum(r['confirmed_forwards'] for r in interruptions.values())
        phase_receipts['timing'] = phase_receipts.get('timing', 0)+archived_calls
        receipt_scope['timing'] = {'final_attempt_receipt_forwards':timing['receipt_forwards'],
                                  'archived_confirmed_forwards':archived_calls,
                                  'additional_unknown_forwards':[0,sum(r['additional_unknown_forwards'][1] for r in interruptions.values())],
                                  'partial_samples_used_as_final_timing':False,
                                  'physical_forward_count_exact':False}
        if terminated is not None:
            receipt_scope['timing']['final_attempt_completion'] = 'FAILED_CONCURRENCY_GUARD_WITH_CLOSED_COUNT'
    memory, memory_phases = audit_memory(a, protocol, fhash); phase_receipts.update(memory_phases)
    api_name = api_receipt or "installed_model_example.json"
    api = a.read(api_name)
    if api:
        phase_receipts["installed_api_smoke"] = audit_installed_api(
            a, api, protocol, a.hash("protocol.json"), a.read("cal_panel.json"))
    pool = a.read("real_document_pool.json")
    dependencies = []
    if pool:
        for name, expected in (("data/benchmarks/gdn/policy.npz", protocol["legacy_policy_sha256"]),
                               ("data/tokenizer/tokenizer.json", pool["tokenizer_sha256"])):
            actual = a.hash(name, package=True)
            a.require(actual == expected, "INDEPENDENT_HISTORICAL_DEPENDENCY_HASH:"+name)
            dependencies.append({"path": name, "existing_expected_sha256": expected, "actual_sha256": actual,
                                 "match": actual == expected, "authority": "protocol.json" if "policy.npz" in name else "real_document_pool.json"})
    reconciliations = []
    for phase, attempts in ls["by_phase"].items():
        ledger_calls = sum(x["physical_forwards"] for x in attempts)
        observed = phase_receipts.get(phase)
        difference = None if not integer(observed) else ledger_calls-observed
        if observed is None:
            a.incomplete.append("PHASE_RECEIPT_UNRESOLVED:"+phase); status = "UNRESOLVED_NO_RECEIPT"
        elif difference:
            if len(attempts) == 1 and attempts[0].get("status") == "COMPLETE":
                a.errors.append("COMPLETE_PHASE_FORWARD_RECEIPT_MISMATCH:"+phase)
                status = "FAIL_FORWARD_MISMATCH"
            else:
                a.incomplete.append("RETRY_OR_PARTIAL_FORWARD_COVERAGE_GAP:"+phase)
                status = "UNRESOLVED_RETRY_OR_PARTIAL_GAP"
        else:
            bounded = [x for x in attempts if x['index'] in interruptions]
            if bounded and all(x.get('status') == 'COMPLETE' or x['index'] in resolved_closed_attempts for x in attempts):
                status = ('RECONCILED_CLOSED_PARTIAL_TIMING_WITH_BOUNDED_INITIAL_UNKNOWN_CALLS' if terminated is not None and phase == 'timing'
                          else 'RECONCILED_CONFIRMED_LOWER_BOUND_WITH_UNKNOWN_TRAILING_CALLS')
            else:
                status = "RECONCILED" if all(x.get("status") == "COMPLETE" for x in attempts) else "COUNTS_MATCH_BUT_ATTEMPT_INCOMPLETE"
        reconciliations.append({"phase": phase, "attempts": len(attempts), "ledger_forwards": ledger_calls,
                                "receipt_forwards": observed, "uncovered_or_duplicated_forwards": difference, "status": status,
                                "scope": receipt_scope.get(phase, "Direct phase receipt count; not quality or independent runtime replication")})
    for phase in phase_receipts:
        if phase not in ls["by_phase"]:
            a.incomplete.append("RECEIPT_WITHOUT_THIS_BUDGET_PHASE:"+phase)
    for phase in ("evaluate", "timing", "installed_api_smoke"):
        if phase not in ls["by_phase"]:
            a.incomplete.append("REQUIRED_PHASE_NOT_RUN:"+phase)
    documented_incomplete = {'TIMING_PLANNED_BLOCKS_OR_FINAL_STATUS_INCOMPLETE'} if terminated is not None else set()
    bookkeeping_incomplete = [reason for reason in a.incomplete if reason not in documented_incomplete]
    status = ("FAIL" if a.errors else "INCOMPLETE" if bookkeeping_incomplete else
              'INCOMPLETE_TIMING_WITH_RECONCILED_ACCOUNTING' if terminated is not None else
              "INCOMPLETE" if a.incomplete else
              "ACCOUNTING_RECONCILED_WITH_BOUNDED_UNKNOWN" if interruptions else "PASS_ACCOUNTING_RECONCILED")
    unknown_upper = sum(r['additional_unknown_forwards'][1] for r in interruptions.values())
    coverage_complete = not a.errors and not bookkeeping_incomplete
    return {"status": status,
            "scope": "Post-freeze independent CPU bookkeeping audit; no scientific performance or GPU replication claim",
            "checks": a.checks, "errors": a.errors, "incomplete": a.incomplete,
            "bookkeeping_incomplete":bookkeeping_incomplete,
            "bookkeeping_status":('RECONCILED_WITH_BOUNDED_UNKNOWN' if coverage_complete and interruptions else
                                  'RECONCILED_EXACT_COUNTS' if coverage_complete else 'UNRESOLVED'),
            "execution_complete":coverage_complete and not a.incomplete and terminated is None,
            "ledger": {k:v for k,v in ls.items() if k != "by_phase"}, "phase_reconciliation": reconciliations,
            "timing": timing, "memory": memory, "historical_dependency_checks": dependencies,
            "interrupted_attempts":list(interruptions.values()),
            "terminated_partial_timing":terminated,
            "physical_forward_count_exact":status == 'PASS_ACCOUNTING_RECONCILED',
            "confirmed_physical_forwards_lower_bound":ls['summed_physical_forwards'],
            "physical_forwards_interval":([ls['summed_physical_forwards'],ls['summed_physical_forwards']+unknown_upper]
                                           if coverage_complete else None),
            "physical_forwards_interval_reason":('All attempts receipt-covered; archived trailing count bounded, never imputed'
                                                  if coverage_complete else 'Unfinished/unverified phases prevent a final total interval'),
            "budget_charge_seconds":ls['summed_active_seconds'],
            "GPU_wall_charge_exact":False if interruptions else None,
            "GPU_wall_charge_scope":('Includes conservative upper wall charge for external interruption; not exact GPU occupancy'
                                     if interruptions else 'Recorded phase wall time, not independent utilization integral'),
            "runtime_guard_gaps": ["Legacy policy and packaged tokenizer byte checks here are publication checks against independently existing authorities, not additional guards in the frozen GPU runner.",
                                   "The frozen runner checks its evidence-root source files and selected native source; this audit does not prove every external library byte or imported-path identity.",
                                   "Warmup completion is count-reconciled from final timing receipt; no independent per-warmup trace was stored.",
                                   "Distinct memory PIDs and recorded flags are not independently observed OS process-lifecycle evidence."],
            "large_capture_PT_read": False, "profiler_trace_required": False, "TEST_quality_values_analyzed": False,
            "audit_model_forwards": 0, "original_expected_hashes_changed": False, "sources": a.sources,
            "auditor_sha256": digest(Path(__file__))}


class SelfTests(unittest.TestCase):
    def ledger(self):
        return {"first_gpu_start_utc": "synthetic", "limit_seconds": 14400, "physical_forwards": 3, "GPU_active_seconds": .3,
                "attempts": [{"phase":"example", "status":"COMPLETE", "physical_forwards":3, "seconds":.3}]}

    def test_sum_mismatch_is_not_pass(self):
        value = self.ledger(); value["physical_forwards"] = 4
        self.assertIn("TOTAL_FORWARD_LEDGER_SUM_MISMATCH", ledger_summary(value)["errors"])
        value = self.ledger(); value["GPU_active_seconds"] = .4
        self.assertIn("TOTAL_ACTIVE_SECONDS_LEDGER_SUM_MISMATCH", ledger_summary(value)["errors"])

    def test_running_or_failed_attempt_is_incomplete(self):
        for status in ("RUNNING", "FAILED"):
            value = self.ledger(); value["attempts"][0]["status"] = status
            result = ledger_summary(value)
            self.assertTrue(result["incomplete"]); self.assertFalse(result["errors"])

    def test_retry_calls_are_not_collapsed(self):
        value = self.ledger(); value["attempts"].append({"phase":"example", "status":"COMPLETE", "physical_forwards":2, "seconds":.2})
        value.update(physical_forwards=5, GPU_active_seconds=.5)
        result = ledger_summary(value)
        self.assertEqual(result["summed_physical_forwards"],5)
        self.assertEqual(len(result["by_phase"]["example"]),2)

    def interruption_fixture(self, root):
        relative = Path('interrupted_attempts/timing_01'); archive = root/relative
        archive.mkdir(parents=True)
        protocol = {'timing_prefix':16,'timing_decode':2,'timing_labels':['A','B'],
                    'timing_warmup_blocks':1,'timing_measured_blocks':8}
        old = {'first_gpu_start_utc':'synthetic','GPU_active_seconds':10.25,'physical_forwards':90,
               'limit_seconds':14400,'attempts':[{'phase':'timing','status':'RUNNING','physical_forwards':90,
                                                'seconds':10.25,'block':1,'method':'A'}]}
        partial = {'status':'RUNNING','rows':[{'block':i//2,'order':i%2,'method':'A' if i%2 == 0 else 'B',
                                              'prompt_tokens':16,'decode_tokens':2,'source_freeze_sha256':'synthetic'}
                                             for i in range(3)]}
        objects = {'budget.json':old,'timing.json':partial,
                   'progress.json':{'phase':'timing','status':'RUNNING','pid':1}}
        for name,value in objects.items(): (archive/name).write_text(json.dumps(value))
        (archive/'timing_execution.log').write_text('Synthetic interruption fixture, not actual execution\n')
        receipt = {'status':'INTERRUPTED_EXTERNAL_PROCESS_TERMINATION','phase':'timing','original_attempt_index':0,
                   'exit_code':143,'signal':15,'pid':1,'numerical_source_or_protocol_changed':False,
                   'calls_per_label':18,'warmup_calls':36,'completed_measured_rows':3,'recorded_physical_forwards':90,
                   'last_completed_block':1,'last_completed_method':'A','physical_forward_count_exact':False,
                   'additional_unrecorded_calls_lower_bound':0,'additional_unrecorded_calls_upper_bound':18,
                   'attempt_physical_forwards_lower_bound':90,'attempt_physical_forwards_upper_bound':108,
                   'last_recorded_seconds':10.25,'last_budget_write_utc':'2026-09-12T00:00:00+00:00',
                   'confirmed_dead_utc':'2026-09-12T00:00:03+00:00','charged_seconds_upper_bound':15,
                   'archived_files':[{'path':name,'bytes':(archive/name).stat().st_size,'sha256':digest(archive/name)}
                                     for name in ('budget.json','progress.json','timing.json','timing_execution.log')]}
        (archive/'receipt.json').write_text(json.dumps(receipt))
        closed = {**old['attempts'][0],'status':receipt['status'],'seconds':15,
                  'last_measured_seconds':10.25,'seconds_are_conservative_upper_charge':True,
                  'physical_forward_count_exact':False,'additional_unrecorded_calls_lower_bound':0,
                  'additional_unrecorded_calls_upper_bound':18,'interruption_receipt':str(relative/'receipt.json')}
        ledger = {**old,'GPU_active_seconds':45,'physical_forwards':414,
                  'physical_forwards_are_confirmed_lower_bound':True,'additional_unrecorded_physical_forwards_upper_bound':18,
                  'attempts':[closed,{'phase':'timing','status':'COMPLETE','seconds':30,'physical_forwards':324}]}
        return protocol, ledger, receipt, archive

    def test_archived_interruption_and_retry_preserve_unknown_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, ledger, receipt, archive = self.interruption_fixture(root)
            audit = Audit(root,root); bounded = audit_interruptions(audit,ledger,protocol,'synthetic')
            self.assertFalse(audit.errors); self.assertFalse(audit.incomplete)
            self.assertEqual(bounded[0]['attempt_forwards_interval'],[90,108])
            self.assertFalse(bounded[0]['physical_forward_count_exact'])
            value = ledger_summary(ledger,bounded)
            self.assertFalse(value['errors']); self.assertFalse(value['incomplete'])
            self.assertEqual(value['summed_physical_forwards'],414)
            self.assertEqual(value['summed_physical_forwards']+bounded[0]['additional_unknown_forwards'][1],432)
            self.assertEqual(value['summed_active_seconds'],45)

    def test_unknown_calls_cannot_be_declared_exact_or_archive_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, ledger, receipt, archive = self.interruption_fixture(root)
            receipt['physical_forward_count_exact'] = True
            (archive/'receipt.json').write_text(json.dumps(receipt))
            audit = Audit(root,root); bounded = audit_interruptions(audit,ledger,protocol,'synthetic')
            self.assertFalse(bounded); self.assertIn('INTERRUPTION_MUST_NOT_CLAIM_EXACT_COUNT',audit.errors)
            receipt['physical_forward_count_exact'] = False
            (archive/'receipt.json').write_text(json.dumps(receipt))
            (archive/'timing_execution.log').write_text('different original bytes')
            audit = Audit(root,root); bounded = audit_interruptions(audit,ledger,protocol,'synthetic')
            self.assertFalse(bounded); self.assertIn('INTERRUPTION_ARCHIVE_BYTES:timing_execution.log',audit.errors)

    def test_more_than_one_timing_retry_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, ledger, receipt, archive = self.interruption_fixture(root)
            ledger['attempts'].append({'phase':'timing','status':'COMPLETE','seconds':30,'physical_forwards':324})
            ledger.update(GPU_active_seconds=75,physical_forwards=738)
            audit = Audit(root,root); audit_interruptions(audit,ledger,protocol,'synthetic')
            self.assertIn('MORE_THAN_ONE_TIMING_RETRY',audit.errors)

    def test_bounded_exit_flag_never_swallows_failure_or_incomplete(self):
        self.assertEqual(exit_status('ACCOUNTING_RECONCILED_WITH_BOUNDED_UNKNOWN',False),3)
        self.assertEqual(exit_status('ACCOUNTING_RECONCILED_WITH_BOUNDED_UNKNOWN',True),0)
        for allowed in (False,True):
            self.assertEqual(exit_status('FAIL',allowed),1)
            self.assertEqual(exit_status('INCOMPLETE',allowed),2)
            self.assertEqual(exit_status('PASS_ACCOUNTING_RECONCILED',allowed),0)

    def terminated_fixture(self, root):
        protocol,ledger,receipt,archive=self.interruption_fixture(root)
        ledger['attempts'][1].update(status='FAILED',seconds=12,physical_forwards=90)
        ledger.update(GPU_active_seconds=27,physical_forwards=180)
        rows=[{'block':i//2,'order':i%2,'method':'A' if i%2==0 else 'B','prompt_tokens':16,'decode_tokens':2,
               'source_freeze_sha256':'synthetic','isolated_before':True,'isolated_after':i<2,
               'isolation_after':{'status_ok':True,'no_competing_GPU_worker':i<2,
                                  'processes':[{'pid':3}]+([{'pid':4}] if i==2 else [])}}
              for i in range(3)]
        (root/'timing.json').write_text(json.dumps({'status':'RUNNING','rows':rows}))
        lifecycle={'status':'FAILED_CHILD_EXIT','return_code':1,'registered_phase':'timing','attempt_number':2,
                   'source_freeze_sha256':'synthetic','numerical_source_changed':False,'checkpoint_or_policy_reselection':False,
                   'GPU_worker_pid':3,'supervisor_pid':2,'first_partial_timing':'interrupted_attempts/timing_01/timing.json',
                   'supervisor_wall_seconds':10}
        (root/'timing_retry_lifecycle.json').write_text(json.dumps(lifecycle))
        (root/'phase_failures').mkdir()
        failure={'phase':'timing','method':None,'reason':'TIMING_CONFOUNDED_OTHER_GPU_WORKER','exception':'RuntimeError',
                 'status':'FAILED_NOT_SILENTLY_RESUMED','completed_observations_preserved':True}
        (root/'phase_failures/timing_all_fixture.json').write_text(json.dumps(failure))
        return protocol,ledger

    def test_closed_concurrency_partial_reconciles_count_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,ledger=self.terminated_fixture(root);audit=Audit(root,root)
            interrupts=audit_interruptions(audit,ledger,protocol,'synthetic')
            result=audit_terminated_timing(audit,ledger,protocol,'synthetic',interrupts)
            self.assertFalse(audit.errors);self.assertFalse(audit.incomplete)
            self.assertEqual(result['physical_forwards'],90)
            self.assertEqual(result['completed_measured_rows'],3)
            self.assertEqual(result['planned_measured_rows'],16)
            self.assertFalse(result['final_timing_complete'])
            self.assertEqual(result['latency_ratio_or_CI_from_partial_rows'],'NOT_COMPUTED')
            self.assertEqual(result['clock_difference_seconds'],2)
            self.assertEqual(result['extra_GPU_PIDs_at_failed_boundary'],[4])

    def test_partial_failure_requires_real_guard_and_isolation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,ledger=self.terminated_fixture(root);audit=Audit(root,root)
            interrupts=audit_interruptions(audit,ledger,protocol,'synthetic')
            path=root/'timing.json';partial=strict(path.read_text());partial['rows'][-1]['isolated_after']=True
            path.write_text(json.dumps(partial))
            result=audit_terminated_timing(audit,ledger,protocol,'synthetic',interrupts)
            self.assertIsNone(result)
            self.assertIn('TERMINATED_TIMING_LAST_ROW_CONCURRENCY_EVIDENCE',audit.errors)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,ledger=self.terminated_fixture(root);audit=Audit(root,root)
            interrupts=audit_interruptions(audit,ledger,protocol,'synthetic')
            ledger['attempts'][1]['physical_forwards']-=1
            self.assertIsNone(audit_terminated_timing(audit,ledger,protocol,'synthetic',interrupts))
            self.assertIn('TERMINATED_TIMING_CLOSED_FORWARD_COUNT',audit.errors)

    def test_documented_incomplete_flag_is_narrow_and_remains_incomplete(self):
        name='INCOMPLETE_TIMING_WITH_RECONCILED_ACCOUNTING'
        self.assertEqual(exit_status(name,False,False),2)
        self.assertEqual(exit_status(name,True,False),2)
        self.assertEqual(exit_status(name,False,True),2)
        self.assertEqual(exit_status(name,True,True),0)
        self.assertEqual(exit_status('INCOMPLETE',True,True),2)
        self.assertEqual(exit_status('FAIL',True,True),1)

    def test_retained_evaluation_failure_is_completed_accounting(self):
        self.assertTrue(completed_evaluation_receipt({"status":"COMPLETE_WITH_RETAINED_FAILURES"}, 40, 40))
        self.assertFalse(completed_evaluation_receipt({"status":"COMPLETE_WITH_RETAINED_FAILURES"}, 39, 40))
        self.assertFalse(completed_evaluation_receipt({"status":"RUNNING"}, 40, 40))

    def api_fixture(self):
        protocol = {"policy_sha256":"synthetic-policy", "model_revision":"synthetic-model",
                    "codec_revision":"R2_OFFSET", "method_layers":{"R2_DIAG":[0,1]}}
        panel = {"split":"CAL", "items":[{"id":"synthetic-cal", "input_ids":list(range(64))}]}
        api = {"status":"CAL_API_SMOKE_COMPLETE", "policy_sha256":"synthetic-policy",
               "protocol_sha256":"synthetic-protocol", "model_revision":"synthetic-model",
               "profile":"R2_OFFSET", "layers":[0,1], "model_downloads":0,
               "document":"synthetic-cal", "input_ids":list(range(32)), "total_attempted_token_forwards":64,
               "records":[{"method":m, "status":"COMPLETE_FINITE", "planned_token_forwards":32,
                           "attempted_token_forwards":32, "finite_completed_token_forwards":32, "failure":None,
                           "not_a_latency_measurement":True, "elapsed_seconds_including_finite_checks":.1,
                           "successful_R2_layer_writes":{"0":32,"1":32} if m == "R2_DIAG" else None,
                           "actual_R2_payload_bytes":2*16*19328 if m == "R2_DIAG" else None}
                          for m in ("NATIVE","R2_DIAG")]}
        return api, protocol, panel

    def test_installed_api_matches_child_schema_and_rejects_false_completion(self):
        api, protocol, panel = self.api_fixture(); audit = Audit(".", ".")
        self.assertEqual(audit_installed_api(audit, api, protocol, "synthetic-protocol", panel), 64)
        self.assertFalse(audit.errors); self.assertFalse(audit.incomplete)
        api["records"][1]["finite_completed_token_forwards"] = 31
        api["records"][1]["actual_R2_payload_bytes"] -= 1
        api["input_ids"][0] = -1
        audit = Audit(".", "."); audit_installed_api(audit, api, protocol, "synthetic-protocol", panel)
        self.assertIn("INSTALLED_API_COMPLETE_COUNTS:R2_DIAG", audit.errors)
        self.assertIn("INSTALLED_API_R2_PAYLOAD", audit.errors)
        self.assertIn("INSTALLED_API_FIXED_CAL_PREFIX", audit.errors)

    def test_installed_api_failure_counts_preserved_not_complete(self):
        api, protocol, panel = self.api_fixture()
        api.update(status="CAL_API_SMOKE_WITH_RETAINED_FAILURE", total_attempted_token_forwards=37)
        api["records"][1].update(status="NUMERICAL_FAILURE", attempted_token_forwards=5,
                                 finite_completed_token_forwards=4, failure={"token":4,"reason":"synthetic"},
                                 successful_R2_layer_writes={"0":5,"1":4})
        audit = Audit(".", ".")
        self.assertEqual(audit_installed_api(audit, api, protocol, "synthetic-protocol", panel), 37)
        self.assertFalse(audit.errors)
        self.assertIn("INSTALLED_API_RETAINED_FAILURE:R2_DIAG", audit.incomplete)

    def test_output_address_reuse_remains_earlier_scratch(self):
        events = [{"action":"alloc","addr":1,"size":100}, {"action":"alloc","addr":2,"size":50},
                  {"action":"free_completed","addr":1}, {"action":"alloc","addr":1,"size":20},
                  {"action":"free_completed","addr":2}]
        result = allocator_summary(events,[1])
        self.assertEqual(result["max_transient_excluding_returned_storage_from_trace"],150)
        self.assertEqual(result["allocation_lifetimes"],3)

    def test_free_request_does_not_end_allocation(self):
        events = [{"action":"alloc","addr":1,"size":100}, {"action":"free_requested","addr":1},
                  {"action":"alloc","addr":2,"size":20}, {"action":"free_completed","addr":1}]
        self.assertEqual(allocator_summary(events,[2])["max_new_allocated_from_trace"],120)

    def test_timing_warmup_is_separate_and_missing_calls_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"timing_labels":["R2_DIAG"], "timing_measured_blocks":8,
                        "timing_prefix":16, "timing_decode":2, "timing_warmup_blocks":1}
            raw = [{"method":"R2_DIAG", "block":block, "order":0, "prompt_tokens":16, "decode_tokens":2,
                    "prefill_seconds":.016, "TTFT_seconds":.016, "decode_seconds":.002,
                    "decode_ms_per_token":1., "tokens_per_second":1000., "source_freeze_sha256":"synthetic",
                    "isolated_before":True, "isolated_after":True} for block in range(8)]
            value = {"rows":raw, "status":"COMPLETE", "physical_forwards":162}
            (root/"timing.json").write_text(json.dumps(value))
            audit = Audit(root, root); result = audit_timing(audit, protocol, "synthetic")
            self.assertFalse(audit.errors)
            self.assertEqual(result["observed_measured_forwards"],144)
            self.assertEqual(result["inferred_unmeasured_warmup_forwards"],18)
            self.assertTrue(result["warmup_count_verified_by_total_only"])
            value["physical_forwards"] = 144
            (root/"timing.json").write_text(json.dumps(value))
            audit = Audit(root, root); audit_timing(audit, protocol, "synthetic")
            self.assertIn("TIMING_TOTAL_VS_FROZEN_WARMUP_AND_MEASURED",audit.errors)


def exit_status(status, allow_bounded_interruption, allow_documented_timing_incomplete=False):
    if status == 'PASS_ACCOUNTING_RECONCILED': return 0
    if status == 'ACCOUNTING_RECONCILED_WITH_BOUNDED_UNKNOWN':
        return 0 if allow_bounded_interruption else 3
    if status == 'INCOMPLETE_TIMING_WITH_RECONCILED_ACCOUNTING':
        return 0 if allow_bounded_interruption and allow_documented_timing_incomplete else 2
    return 2 if status == 'INCOMPLETE' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, help="Public checkout or installed evidence root")
    p.add_argument("--run", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--api-receipt", default="installed_model_example.json", help="Relative receipt path within run")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--allow-bounded-interruption", action="store_true",
                   help="Accept only the named receipt-validated bounded-unknown state as successful bookkeeping, never exact PASS")
    p.add_argument('--allow-documented-timing-incomplete', action='store_true',
                   help='With --allow-bounded-interruption, accept only validated closed concurrency-failed timing bookkeeping; cost and execution remain incomplete')
    args = p.parse_args(argv)
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests))
        return 0 if result.wasSuccessful() else 1
    if args.root is None or args.run is None or args.out is None:
        p.error("--root, --run and --out are required for an actual audit")
    if args.out.exists():
        raise FileExistsError("Audit receipt already exists; use a new output, never overwrite an earlier audit")
    if Path(args.api_receipt).is_absolute() or ".." in Path(args.api_receipt).parts:
        p.error("--api-receipt must be a safe relative path within the run")
    try:
        result = audit_run(args.root.resolve(), args.run.resolve(), args.api_receipt)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result = {"status":"FAIL_AUDIT_SCHEMA", "reason":type(exc).__name__+":"+str(exc), "audit_model_forwards":0,
                  "original_expected_hashes_changed":False, "TEST_quality_values_analyzed":False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"], "errors":result.get("errors",[]),
                      "incomplete":result.get("incomplete",[]), "reason":result.get("reason")}, indent=2))
    return exit_status(result['status'], args.allow_bounded_interruption, args.allow_documented_timing_incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
