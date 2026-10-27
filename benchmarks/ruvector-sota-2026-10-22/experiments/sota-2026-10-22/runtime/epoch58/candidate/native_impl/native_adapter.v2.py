"""E58 source-bound native adapter; importing this file performs no execution.

One invocation owns exactly one native helper child and one index/corpus.  The
caller owns the enclosing session; this adapter never creates or signals a new
process group.  Partial output and observed counters survive failures.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import time

KIND = 'REAL_NATIVE_TYPED_NLI_COVERAGE_ORDER_DEV'
BINARY_SHA = 'ff559e9fac2e3d29a8b32e890fdb047bcd4bd28f389b4731ad72148b221d5a2e'
NATIVE_KEYS = (
    'native_index_initializations_attempted', 'native_index_initializations_completed',
    'native_insertions_attempted', 'native_insertions_completed',
    'native_queries_attempted', 'native_queries_completed',
    'native_len_calls_attempted', 'native_len_calls_completed',
    'native_hit_source_validations', 'native_binary_loads_attempted',
    'native_binary_loads_completed', 'native_version_calls_attempted',
    'native_version_calls_completed', 'Node_network_operations_attempted')
PROCESS_KEYS = ('native_processes_attempted', 'native_processes_completed')
COUNTER_KEYS = NATIVE_KEYS + PROCESS_KEYS
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 65536
MAX_PROGRESS_BYTES = 4194304


def need(value, code):
    if not value:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'DUPLICATE_JSON_KEY')
        value[key] = item
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode() + b'\n'


def safe(path):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and not any(x.is_symlink() for x in (p, *p.parents)),
         'CANONICAL_NO_SYMLINK_PATH')
    return p


def read_stable(path, maximum=None):
    p = safe(path)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode), 'REGULAR_INPUT')
        need(maximum is None or before.st_size <= maximum, 'INPUT_BYTES_CAP')
        raw = f.read() if maximum is None else f.read(maximum + 1)
        after = os.fstat(f.fileno())
    need(maximum is None or len(raw) <= maximum, 'INPUT_BYTES_CAP')
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'STABLE_INPUT')
    return raw


def checked(ref):
    need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
         type(ref['bytes']) is int and ref['bytes'] >= 0 and
         type(ref['sha256']) is str and re.fullmatch('[a-f0-9]{64}', ref['sha256']), 'TYPED_REF')
    raw = read_stable(ref['path'], ref['bytes'])
    need(len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], 'PINNED_BYTES_SHA')
    return Path(ref['path'])


def reference(path, maximum=None):
    raw = read_stable(path, maximum)
    return {'path': str(safe(path)), 'bytes': len(raw), 'sha256': sha(raw)}


def put(path, value):
    p = safe(path)
    raw = canonical(value)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    d = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(d)
    finally:
        os.close(d)
    return {'path': str(p), 'bytes': len(raw), 'sha256': sha(raw)}


def geometry(config):
    need(type(config) is dict and type(config.get('schema_version')) is int and config['schema_version'] == 1 and
         type(config.get('epoch')) is int and config['epoch'] == 58 and
         config.get('kind') == KIND and config.get('phase') in ('engineering', 'dev'), 'NATIVE_PLAN_KIND')
    n, q = config.get('corpus_document_count'), config.get('query_count')
    need(type(n) is int and type(q) is int and
         ((config['phase'] == 'engineering' and n == 128 and q == 16) or
          (config['phase'] == 'dev' and 1 <= n <= 24 and q == 17)), 'NATIVE_GEOMETRY')
    need(type(config.get('plan_sha256')) is str and re.fullmatch('[a-f0-9]{64}', config['plan_sha256']),
         'MASTER_PLAN_SHA')
    need(type(config.get('native_helper_max_seconds')) in (int, float) and
         0 < config['native_helper_max_seconds'] <= 60, 'HELPER_TIME_CAP')
    need(type(config.get('native_process_timeout_seconds')) in (int, float) and
         2 < config['native_process_timeout_seconds'] <= 85, 'PROCESS_INCLUSIVE_CAP')
    return n, q


def limits(config):
    n, q = geometry(config)
    return dict(zip(NATIVE_KEYS, (1, 1, n, n, q, q, 2, 2, n*q, 1, 1, 1, 1, 0)))


def valid_ids(values, length):
    need(type(values) is list and len(values) == length and
         all(type(v) is str and v and len(v.encode()) <= 256 and
             not any(ord(x) < 32 or ord(x) == 127 for x in v) for v in values) and
         len(set(values)) == length, 'UNIQUE_BOUNDED_IDS')
    return values


def decode(block, rows):
    need(type(block) is dict and set(block) == {'encoding', 'dtype', 'shape', 'bytes', 'sha256', 'data_base64'}
         and block['encoding'] == 'base64' and block['dtype'] == 'float32le' and
         block['shape'] == [rows, 384] and type(block['bytes']) is int and
         block['bytes'] == rows*1536 and type(block['data_base64']) is str, 'F32_SCHEMA')
    raw = base64.b64decode(block['data_base64'], validate=True)
    need(len(raw) == block['bytes'] and base64.b64encode(raw).decode() == block['data_base64'] and
         sha(raw) == block['sha256'], 'F32_HASH_BYTES')
    vectors = [struct.unpack_from('<384f', raw, i*1536) for i in range(rows)]
    for v in vectors:
        need(all(math.isfinite(x) for x in v) and abs(math.sqrt(math.fsum(x*x for x in v))-1) <= 1e-5,
             'FINITE_NORMALIZED_F32')
    return raw, vectors


def validate_bundle(config, bundle):
    n, q = geometry(config)
    need(type(bundle) is dict and set(bundle) == {'schema_version', 'epoch', 'kind', 'phase',
         'plan_sha256', 'corpus', 'queries'} and type(bundle['schema_version']) is int and
         bundle['schema_version'] == 1 and type(bundle['epoch']) is int and
         bundle['epoch'] == 58 and bundle['kind'] == KIND and
         bundle['phase'] == config['phase'] and bundle['plan_sha256'] == config['plan_sha256'], 'BUNDLE_BINDING')
    need(all(type(bundle[k]) is dict and set(bundle[k]) == {'ids', 'vectors'} for k in ('corpus', 'queries')),
         'FEATURES_ONLY_BUNDLE')
    source_ids = valid_ids(bundle['corpus']['ids'], n)
    query_ids = valid_ids(bundle['queries']['ids'], q)
    need(query_ids == config['selected_query_ids'], 'FIXED_QUERY_IDS')
    raw, vectors = decode(bundle['corpus']['vectors'], n)
    _, queries = decode(bundle['queries']['vectors'], q)
    return source_ids, query_ids, raw, vectors, queries


def process_identity(pid):
    text = Path('/proc', str(pid), 'stat').read_text()
    columns = text.rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'startticks': int(columns[19]),
            'pgid': int(columns[2]), 'sid': int(columns[3]), 'state': columns[0]}


def same_process(identity):
    try:
        actual = process_identity(identity['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return None
    if actual['startticks'] != identity['startticks']:
        return None
    need(actual['pgid'] == identity['pgid'] and actual['sid'] == identity['sid'], 'OWNED_PROCESS_GROUP_CHANGED')
    return actual


def physically_closed(identity):
    actual = same_process(identity)
    if actual is None:
        return True
    if actual['state'] not in ('Z', 'X'):
        return False
    p = Path('/proc', str(identity['pid']))
    try:
        return not any(int(v) for v in (p/'statm').read_text().split()) and not list((p/'fd').iterdir())
    except FileNotFoundError:
        return True


def signal_exact(identity, value):
    actual = same_process(identity)
    if actual is not None and actual['state'] not in ('Z', 'X'):
        try:
            os.kill(identity['pid'], value)
        except ProcessLookupError:
            pass


def observed_counters(value, config, *, terminal, enforce_caps=True):
    bound = limits(config)
    need(type(value) is dict and all(type(value.get(k)) is int and 0 <= value[k] and
         (not enforce_caps or value[k] <= bound[k]) for k in NATIVE_KEYS), 'NATIVE_COUNTER_SCHEMA_CAP')
    result = {k: value[k] for k in NATIVE_KEYS}
    for base in ('native_index_initializations', 'native_insertions', 'native_queries',
                 'native_len_calls', 'native_binary_loads', 'native_version_calls'):
        need(result[base+'_completed'] <= result[base+'_attempted'] <= result[base+'_completed']+1,
             'NATIVE_ATTEMPT_COMPLETION_COUNTS')
    if terminal:
        need(all(result[k] == bound[k] for k in NATIVE_KEYS if k != 'native_hit_source_validations'),
             'NATIVE_TERMINAL_COUNTS')
    return result


def validate_result(result, config, bundle, bundle_ref, config_ref, helper_ref):
    source_ids, query_ids, raw, vectors, queries = validate_bundle(config, bundle)
    n, q = geometry(config)
    need(type(result) is dict and result.get('schema_version') == 1 and result.get('epoch') == 58 and
         result.get('kind') == KIND and result.get('phase') == config['phase'] and
         result.get('plan_sha256') == config['plan_sha256'] and result.get('errors') == [] and
         result.get('stage') == 'COMPLETE', 'ACTUAL_NATIVE_SUCCESS_BINDING')
    need(result.get('native_config_sha256') == config_ref['sha256'] and
         result.get('bundle_sha256') == bundle_ref['sha256'] and
         result.get('helper_source_sha256') == helper_ref['sha256'] and
         result.get('native_binary_sha256') == BINARY_SHA, 'ACTUAL_NATIVE_HASH_BINDINGS')
    need(result.get('corpus_document_count') == n and result.get('query_count') == q and
         result.get('top_k') == n and result.get('binary_version') == '2.3.0' and
         result.get('package_version') == '0.1.33' and result.get('platform_package_version') == '0.1.31' and
         result.get('raw_native_scores_are_not_probabilities') is True, 'ACTUAL_NATIVE_VERSIONS_GEOMETRY')
    observed = observed_counters(result, config, terminal=True)
    rows = result.get('queries')
    need(type(rows) is list and [r.get('query_id') for r in rows] == query_ids, 'ACTUAL_NATIVE_QUERY_IDENTITIES')
    source_map = {v: i for i, v in enumerate(source_ids)}
    total_hits = 0; misses = 0
    for query_index, row in enumerate(rows):
        ids = row.get('ids'); hits = row.get('raw_hits')
        need(type(ids) is list and len(ids) <= n and len(set(ids)) == len(ids) and
             all(x in source_map for x in ids), 'NATIVE_HIT_SOURCE_IDENTITIES')
        need(type(hits) is list and len(hits) == len(ids) and
             row.get('raw_ids') == [h.get('id') for h in hits] and
             row.get('raw_scores') == [h.get('score') for h in hits], 'NATIVE_RAW_HITS_RETAINED')
        need(len(set(row['raw_ids'])) == len(hits) and set(row['raw_ids']) == set(ids), 'NATIVE_HIT_UNIQUE')
        for hit in hits:
            need(type(hit) is dict and set(hit) == {'id','score','vector_receipt','metadata_receipt'} and
                 hit['id'] in source_map and type(hit['score']) in (int,float) and math.isfinite(hit['score']),
                 'NATIVE_VALID_RAW_HIT')
            i = source_map[hit['id']]
            need(hit['vector_receipt'] == {'source_vector_index':i,'dimensions':384,'dtype':'float32le',
                 'sha256':sha(raw[i*1536:(i+1)*1536]),'source_equal_before_stripping':True} and
                 hit['metadata_receipt'] == {'present':False}, 'NATIVE_SOURCE_VECTOR_RECEIPT')
        ordered = []
        pos = 0
        while pos < len(hits):
            end = pos + 1
            while end < len(hits) and hits[end]['score'] == hits[pos]['score']:
                end += 1
            ordered.extend(sorted(hits[pos:end], key=lambda x:x['id']))
            pos = end
        need(ids == [h['id'] for h in ordered] and row.get('scores') == [h['score'] for h in ordered] and
             row.get('rank_scores') == list(range(n,n-len(ids),-1)) and row.get('requested_k') == n,
             'NATIVE_ORDER_RETAINED_EXCEPT_EXACT_EQUAL_TIES')
        missing = [x for x in source_ids if x not in ids]
        miss = bool(missing); misses += int(miss); total_hits += len(hits)
        need(row.get('missing_source_ids') == missing and row.get('retrieval_miss') is miss and
             row.get('fallback_required') is miss and row.get('coverage_complete') is (not miss) and
             row.get('native_order_used_except_returned_equal_score_ties') is True and
             row.get('exact_order_never_substituted') is True, 'NATIVE_MISS_FULL_CONTEXT_FALLBACK')
        need(type(row.get('elapsed_seconds')) in (int,float) and math.isfinite(row['elapsed_seconds']) and
             row['elapsed_seconds'] >= 0, 'NATIVE_QUERY_LATENCY')
        exact = row.get('exact_cosine_diagnostic')
        need(type(exact) is list and len(exact) == n and set(x.get('id') for x in exact) == set(source_ids),
             'DIAGNOSTIC_COSINE_FULL_SOURCE')
        qv = queries[query_index]; qn = math.fsum(x*x for x in qv)
        expected = {}
        for sid, vec in zip(source_ids, vectors):
            expected[sid] = math.fsum(x*y for x,y in zip(vec,qv))/math.sqrt(math.fsum(x*x for x in vec)*qn)
        need(all(type(x.get('cosine')) in (int,float) and math.isfinite(x['cosine']) and
                 abs(x['cosine']-expected[x['id']]) <= 1e-10 for x in exact), 'DIAGNOSTIC_COSINE_RECALCULATION')
        need(all(exact[i]['cosine'] >= exact[i+1]['cosine'] for i in range(n-1)) and
             row.get('exact_full_order_agreement') is ([x['id'] for x in exact] == ids), 'DIAGNOSTIC_NEVER_SELECTS_ORDER')
    need(result.get('observedHits') == total_hits == observed['native_hit_source_validations'] and
         result.get('retrieval_miss_count') == misses, 'OBSERVED_NATIVE_HIT_AND_MISS_COUNTS')
    return observed


def merge_counts(target, observed):
    need(all(k in target and type(target[k]) is int and target[k] >= 0 for k in COUNTER_KEYS),
         'CALLER_NATIVE_COUNTER_KEYS')
    need(all(target[k] <= observed[k] for k in NATIVE_KEYS), 'NATIVE_COUNTS_MONOTONIC')
    for key in NATIVE_KEYS:
        target[key] = observed[key]


def run_native(config, bundle, external, progress, deadline, *, counters=None, state=None):
    """Launch exactly once, retain actuals, merge counts, return validated native result.

    external is an existing empty directory; deadline is the caller's absolute
    time.monotonic() bound. counters may contain additional parent counter keys.
    native_processes_completed means a spawned child was reaped and physically
    closed, including a child that returned a failure exit status.
    """
    started = time.monotonic()
    need(type(deadline) in (int,float) and math.isfinite(deadline) and deadline > started+3, 'PARENT_DEADLINE')
    n, q = geometry(config)
    validate_bundle(config, bundle)
    folder = safe(external)
    need(folder.is_dir() and not any(folder.iterdir()), 'EMPTY_EXTERNAL_OUTPUT_DIRECTORY')
    need(os.sched_getaffinity(0) == {0,1}, 'NATIVE_ADAPTER_CPU_AFFINITY_0_1')
    node = checked(config['node']); helper = checked(config['native_helper_source'])
    native_binary = checked(config['native_binary'])
    need(config['native_binary']['sha256'] == BINARY_SHA and native_binary ==
         Path(config['native_root'])/'retrieval/node_modules/ruvector-core-linux-x64-gnu/ruvector.node',
         'NATIVE_BINARY_PATH_SHA')
    closure = {str(checked(r)) for r in config['native_closure_sources']}
    need(len(closure) == len(config['native_closure_sources']) and 4 <= len(closure) <= 32,
         'NATIVE_CLOSURE_REFS')
    for name in ('@ruvector/core/index.js','@ruvector/core/package.json',
                 'ruvector-core-linux-x64-gnu/index.js','ruvector-core-linux-x64-gnu/package.json'):
        need(str(Path(config['native_root'])/'retrieval/node_modules'/name) in closure, 'NATIVE_CLOSURE_COMPLETE')
    if counters is None:
        counters = {k:0 for k in COUNTER_KEYS}
    need(all(type(counters.get(k)) is int and counters[k] == 0 for k in COUNTER_KEYS), 'ONE_NATIVE_PROCESS_COUNT_ZERO')
    if state is None:
        state = {}
    need(type(state) is dict, 'MUTABLE_STATE')
    identities = state.setdefault('process_identities', [])
    need(type(identities) is list, 'PROCESS_IDENTITIES_LIST')
    config_ref = put(folder/'native-config.json', config)
    bundle_ref = put(folder/'native-bundle.json', bundle)
    output = folder/'native-result.json'
    put(folder/'NATIVE_ATTEMPT.json', {'schema_version':1,'epoch':58,'kind':KIND,'phase':config['phase'],
        'plan_sha256':config['plan_sha256'],'config_ref':config_ref,'bundle_ref':bundle_ref,
        'node_ref':config['node'],'helper_ref':config['native_helper_source'],
        'parent_pid':os.getpid(),'parent_pgid':os.getpgrp(),'parent_sid':os.getsid(0),
        'parent_deadline_monotonic_seconds':deadline,'automatic_retry':False})
    stop = min(deadline, started+config['native_process_timeout_seconds'])
    logs = [folder/'native-stdout.log', folder/'native-stderr.log']
    env = {'RAYON_NUM_THREADS':'2','UV_THREADPOOL_SIZE':'2','OMP_NUM_THREADS':'2',
           'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2',
           'NUMEXPR_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    process = None; identity = None; reason = None; child_closed = False
    counters['native_processes_attempted'] += 1
    progress('NATIVE_PROCESS_BEFORE')
    try:
        with open(logs[0], 'xb', buffering=0) as out, open(logs[1], 'xb', buffering=0) as err:
            os.chmod(logs[0],0o600); os.chmod(logs[1],0o600)
            process = subprocess.Popen([str(node),str(helper),config_ref['path'],bundle_ref['path'],str(output)],
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env,cwd=folder,
                start_new_session=False,close_fds=True,restore_signals=True)
            identity = process_identity(process.pid)
            need(identity['pgid'] == os.getpgrp() and identity['sid'] == os.getsid(0), 'CHILD_IN_OWNED_ENCLOSING_SESSION')
            identity = {k:identity[k] for k in ('pid','startticks','pgid','sid')}
            identities.append(identity)
            state['native_process_identity_ref'] = put(folder/'NATIVE_CHILD_STARTED.json',
                {'schema_version':1,'epoch':58,'kind':KIND,'phase':config['phase'],
                 'plan_sha256':config['plan_sha256'],'process_identity':identity,
                 'kernel_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'PID_namespace':os.readlink('/proc/self/ns/pid')})
            progress('NATIVE_PROCESS_STARTED')
            while process.poll() is None:
                need(time.monotonic() < stop-2, 'NATIVE_PROCESS_TIME_BOUND')
                need(all(p.stat().st_size <= MAX_LOG_BYTES for p in logs), 'NATIVE_STDIO_BYTES_CAP')
                for name, cap in (('native-result.json',MAX_RESULT_BYTES),('native-result.json.failure.json',MAX_RESULT_BYTES),
                                  ('native-progress.jsonl',MAX_PROGRESS_BYTES)):
                    p = folder/name
                    need(not p.exists() or p.stat().st_size <= cap, 'NATIVE_OUTPUT_BYTES_CAP')
                time.sleep(min(.02,max(.001,stop-2-time.monotonic())))
            process.wait(timeout=max(.01,stop-time.monotonic()))
            child_closed = physically_closed(identity)
            need(child_closed, 'NATIVE_PROCESS_PHYSICAL_CLOSURE')
            counters['native_processes_completed'] += 1
            progress('NATIVE_PROCESS_CLOSED')
            need(all(p.stat().st_size <= MAX_LOG_BYTES for p in logs), 'NATIVE_STDIO_BYTES_CAP')
            need(process.returncode == 0, 'NATIVE_HELPER_EXIT_FAILURE')
        result_ref = reference(output,MAX_RESULT_BYTES)
        state['raw_result_ref'] = result_ref
        result = parse(read_stable(output,MAX_RESULT_BYTES))
        observed = validate_result(result,config,bundle,bundle_ref,config_ref,config['native_helper_source'])
        bindings = parse(read_stable(checked(result['bindings']),65536))
        need(bindings['helper_pid'] == identity['pid'] and bindings['helper_startticks'] == identity['startticks'] and
             bindings['helper_source_sha256'] == config['native_helper_source']['sha256'] and
             result['helper_pid'] == identity['pid'], 'EXACT_CHILD_RESULT_BINDING')
        pr = result['progress']; progress_ref = reference(folder/'native-progress.jsonl',MAX_PROGRESS_BYTES)
        need(pr['path'] == progress_ref['path'] and pr['bytes'] == progress_ref['bytes'] and
             pr['sha256'] == progress_ref['sha256'], 'DURABLE_NATIVE_PROGRESS_REF')
        progress_rows = [parse(x) for x in read_stable(pr['path'],MAX_PROGRESS_BYTES).splitlines()]
        need(len(progress_rows) == pr['rows'] and 1 <= len(progress_rows) <= 10000 and
             progress_rows[-1]['event'] == 'COMPLETE', 'NATIVE_PROGRESS_ROWS')
        prior = {k:0 for k in NATIVE_KEYS}
        for line in progress_rows:
            value = observed_counters(line['counters'],config,terminal=False)
            need(all(prior[k] <= value[k] for k in NATIVE_KEYS), 'NATIVE_PROGRESS_MONOTONIC')
            prior = value
        need(prior == observed, 'NATIVE_TERMINAL_PROGRESS_COUNTS')
        merge_counts(counters,observed)
        progress('NATIVE_COUNTS_MERGED')
        state['native_result_validated'] = True
        need(time.monotonic() < stop, 'NATIVE_ADAPTER_INCLUSIVE_TIME_BOUND')
        return result
    except BaseException as error:
        reason = {'type':type(error).__name__, 'code':str(error) if re.fullmatch('[A-Z0-9_]{1,120}',str(error)) else 'NATIVE_ADAPTER_FAILURE'}
        raise
    finally:
        if process is not None and identity is not None:
            if process.poll() is None:
                signal_exact(identity,signal.SIGTERM)
                try:
                    process.wait(timeout=max(.01,min(.75,stop-time.monotonic())))
                except subprocess.TimeoutExpired:
                    signal_exact(identity,signal.SIGKILL)
                    try:
                        process.wait(timeout=max(.01,min(.75,stop-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            child_closed = physically_closed(identity)
            if child_closed and counters['native_processes_completed'] == 0:
                counters['native_processes_completed'] = 1
        elif process is not None:
            # A direct child remains unreaped, so its PID cannot be reused.  This
            # branch is a failed identity observation, never accepted execution.
            # Popen's owned-child signal is used only to close that uncertainty.
            state['native_process_identity_uncertain'] = True
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=max(.01,min(.75,stop-time.monotonic())))
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=max(.01,min(.75,stop-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            child_closed = process.poll() is not None
            if child_closed:
                counters['native_processes_completed'] = 1
        for path in (output, Path(str(output)+'.failure.json')):
            if path.exists():
                try:
                    raw_ref = reference(path,MAX_RESULT_BYTES)
                    partial = parse(read_stable(path,MAX_RESULT_BYTES))
                    observed = observed_counters(partial,config,terminal=False,enforce_caps=False)
                    if any(observed[k] > limits(config)[k] for k in NATIVE_KEYS):
                        state['observed_native_counter_cap_violation'] = True
                    merge_counts(counters,observed)
                    state.setdefault('raw_result_ref',raw_ref)
                except Exception as e:
                    state['partial_counter_parse_error'] = type(e).__name__
        progress_path = folder/'native-progress.jsonl'
        if progress_path.exists():
            try:
                state['native_progress_ref'] = reference(progress_path,MAX_PROGRESS_BYTES)
                raw_progress = read_stable(progress_path,MAX_PROGRESS_BYTES)
                last = {k:counters[k] for k in NATIVE_KEYS}
                for line in raw_progress.splitlines(keepends=True):
                    if not line.endswith(b'\n'):
                        state['native_progress_partial_tail'] = True
                        break
                    record = parse(line)
                    observed = observed_counters(record['counters'],config,terminal=False,enforce_caps=False)
                    if any(observed[k] > limits(config)[k] for k in NATIVE_KEYS):
                        state['observed_native_counter_cap_violation'] = True
                    if all(observed[k] >= last[k] for k in NATIVE_KEYS):
                        last = observed
                merge_counts(counters,last)
            except Exception as e:
                state['partial_progress_parse_error'] = type(e).__name__
        # A killed/erroring helper may complete work between its last durable
        # event and death.  Retain the observed lower bound and mark uncertainty;
        # never fill from frozen expected counts or accept this as exact actuals.
        state['native_counter_accounting_certain'] = state.get('native_result_validated') is True
        if process is not None:
            state['native_process_exit_code'] = process.poll()
        state['native_process_physically_closed'] = child_closed
        state['native_adapter_elapsed_seconds'] = time.monotonic()-started
        artifacts = {}
        for path in logs:
            if path.exists() and path.stat().st_size <= MAX_LOG_BYTES:
                artifacts[path.name] = reference(path,MAX_LOG_BYTES)
        state['native_stdio_refs'] = artifacts
        state['native_adapter_error'] = reason
        try:
            state['native_adapter_closure_ref'] = put(folder/'NATIVE_CHILD_CLOSED.json',
                {'schema_version':1,'epoch':58,'kind':KIND,'phase':config['phase'],
                 'process_identity':identity,'physically_closed':child_closed,
                 'exit_code':None if process is None else process.poll(),'error':reason,
                 'adapter_elapsed_seconds':state['native_adapter_elapsed_seconds'],
                 'counters':{k:counters[k] for k in COUNTER_KEYS},'stdio_refs':artifacts,
                 'raw_result_ref':state.get('raw_result_ref')})
            progress('NATIVE_ADAPTER_FINAL')
        except Exception as e:
            state['native_closure_persist_error'] = type(e).__name__
        state['native_adapter_inclusive_elapsed_seconds'] = time.monotonic()-started
        state['native_adapter_within_inclusive_bound'] = time.monotonic() < stop
        if reason is None:
            need(state['native_adapter_within_inclusive_bound'], 'NATIVE_ADAPTER_INCLUSIVE_TIME_BOUND')

