"""E59 fixed typed decision, exact source windows and inert utilities.

Import performs no network, model, native, tokenizer or data access.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import time

KIND='REAL_NATIVE_TYPED_READER_BOUNDED_RESEARCH'
LABELS=('Contradiction','Entailment','NotMentioned')
ARMS=('native_order','source_order','full_reference')
HYPOTHESES=('nda-1','nda-10','nda-11','nda-12','nda-13','nda-15','nda-16','nda-17','nda-18','nda-19','nda-2','nda-20','nda-3','nda-4','nda-5','nda-7','nda-8')
DEV_IDS=tuple('contractnli:eced6528:train:473:'+h for h in HYPOTHESES)
QUERY_PREFIX='Represent this sentence for searching relevant passages: '
NUMERIC_FLAGS=('claim_has_numerals','claim_numeral_not_in_supplied_context')
MODEL_SHA='449024a8711a3e655cad410cb2192158f9897fc2a32b65dbb817922351150db7'
NLI_TOKENIZER_SHA='82139106e603ee4e1d5bc99d056ccbed5a92bc24848b1b5a7137c26e00d0dbf6'
NLI_TOKENIZER_GIT_SHA='8df0092a08e1d459f60ec541e08cd35a16362bfe'
PROJECT_ROLE_SEMANTICS='PROJECT_DEV_DIAGNOSTIC_NO_FITTING_NO_CONFIRMATION'
BGE_SHA='6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4'
BGE_TOKENIZER_SHA='d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
PAIR_BASES=('NLI_tokenizer_initializations','BGE_tokenizer_initializations',
'NLI_model_initializations','BGE_model_initializations','NLI_forward_calls',
'NLI_pair_inferences','BGE_forward_calls','BGE_texts','pair_tokenizations',
'BGE_tokenizations','native_index_initializations','native_insertions','native_queries',
'native_len_calls','native_binary_loads','native_version_calls','native_processes')
COUNTER_KEYS=tuple(k+s for k in PAIR_BASES for s in ('_attempted','_completed'))+(
'native_hit_source_validations','Node_network_operations_attempted','forecasts_completed',
'numerical_flagged_cases','Python_network_operations_attempted','paid_API_calls','GPU_calls','Gold_labels_decoded')


def need(value,code):
    if not value: raise ValueError(code)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def unique(items):
    out={}
    for key,value in items:
        need(key not in out,'DUPLICATE_JSON_KEY');out[key]=value
    return out


def parse(raw):
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def safe(path):
    p=Path(path)
    need(p.is_absolute() and p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)),'CANONICAL_NONSYMLINK_PATH')
    return p


def body(path,cap=268435456):
    p=safe(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'BOUNDED_REGULAR_FILE')
        raw=f.read(cap+1);b=os.fstat(f.fileno())
    need(len(raw)<=cap and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns),'STABLE_FILE_READ')
    return raw


def typed_ref(path):
    raw=body(path);return {'path':str(safe(path)),'bytes':len(raw),'sha256':sha(raw)}


def read_ref(ref,cap=268435456):
    need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and ref['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'EXACT_TYPED_REF')
    raw=body(ref['path'],cap);need(len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],'REF_BYTES_SHA')
    return raw


def load_ref(ref,cap=16777216):return parse(read_ref(ref,cap))


def put(path,value):
    p=safe(path);need(p.parent.is_dir(),'OUTPUT_PARENT_EXISTS')
    raw=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n'
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return typed_ref(p)


def imported(ref,name):
    read_ref(ref);spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def percentile(values,p):
    need(values and 0<=p<=1 and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),'FINITE_LATENCIES')
    x=sorted(values);q=(len(x)-1)*p;lo=math.floor(q);hi=math.ceil(q);return x[lo]+(x[hi]-x[lo])*(q-lo)


def stable_scores(logits):
    need(len(logits)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in logits),'THREE_FINITE_TYPED_LOGITS')
    logits=[float(x) for x in logits];m=max(logits);normal=m+math.log(math.fsum(math.exp(x-m) for x in logits))
    logs=[x-normal for x in logits];probs=[math.exp(x) for x in logs]
    need(all(math.isfinite(x) and 0<x<=1 for x in probs) and abs(math.fsum(probs)-1)<=1e-12,'UNCLIPPED_POSITIVE_DOUBLE_SOFTMAX')
    return {'probabilities':probs,'log_probabilities':logs,'prediction':LABELS[max(range(3),key=lambda i:logits[i])]}


def aggregate(logits):
    """Typed max log odds with neutral baseline0; no fitted coefficients."""
    need(logits,'AT_LEAST_ONE_REAL_WINDOW')
    for row in logits:need(len(row)==3 and all(math.isfinite(float(x)) for x in row),'FINITE_WINDOW_LOGITS')
    c=max(float(x[0])-float(x[2]) for x in logits);e=max(float(x[1])-float(x[2]) for x in logits)
    return stable_scores([c,e,0.0])


def may_exit(logits,flags,covered_all):
    need(type(covered_all) is bool and type(flags) is list and all(x in NUMERIC_FLAGS for x in flags),'EARLY_EXIT_INPUTS')
    scores=aggregate(logits)
    if covered_all:return False
    if flags:return False
    c=max(float(x[0])-float(x[2]) for x in logits);e=max(float(x[1])-float(x[2]) for x in logits)
    return (e>=math.log(9) and e-c>=math.log(3)) or (c>=math.log(9) and c-e>=math.log(3))


_NUMBER=re.compile(r'(?<![\w.])[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?![\w.])')

def numeral_tokens(text):
    # Exact lexical normalization only: commas within numeric tokens are removed.
    return sorted(set(m.group(0).replace(',','').lstrip('+') for m in _NUMBER.finditer(text)))


def numeric_diagnostic(claim,context):
    a=numeral_tokens(claim);b=numeral_tokens(context);flags=[]
    if a:flags.append('claim_has_numerals')
    if set(a)-set(b):flags.append('claim_numeral_not_in_supplied_context')
    return {'flags':flags,'claim_numeral_tokens':a,'context_numeral_tokens':b,'claim_tokens_absent_from_supplied_context':sorted(set(a)-set(b)),'policy':'diagnostic_and_exit_disable_only_never_outcome_override'}


def validate_encoding(encoding,maximum=512):
    ids=list(encoding.ids);mask=list(encoding.attention_mask);types=list(encoding.type_ids)
    need(ids and len(ids)<=maximum and len(mask)==len(types)==len(ids) and all(type(x) is int and x>=0 for x in ids) and all(x==1 for x in mask) and all(x in (0,1) for x in types),'WHOLE_UNTRUNCATED_TOKENIZER_ENCODING')
    return {'input_ids':ids,'attention_mask':mask,'token_type_ids':types}


def whole_pair(tokenizer,premise,hypothesis):
    need(type(premise) is str and premise.strip() and type(hypothesis) is str and hypothesis.strip(),'GENUINE_NONEMPTY_SOURCE_CLAIM')
    return validate_encoding(tokenizer.encode(premise,hypothesis))


def whole_single(tokenizer,text):
    need(type(text) is str and text.strip(),'NONEMPTY_EMBEDDING_TEXT');return validate_encoding(tokenizer.encode(text))


def fits_window(text,claims,nli,bge):
    if not text.strip():return False
    try:
        for claim in claims:whole_pair(nli,text,claim)
        whole_single(bge,text)
        return True
    except ValueError as e:
        if str(e)=='WHOLE_UNTRUNCATED_TOKENIZER_ENCODING':return False
        raise


def make_catalog(bank,nli,bge):
    """Greedy exact contiguous source partition; all17claim capacity, no truncation.

    Each original unit is retained if it fits. Oversized units are split at the
    largest fitting codepoint prefix found by binary search, then greedily merge
    adjacent fragments only when the combined text fits every claim and BGE.
    Decisions use tokens and source offsets only, never labels or model logits.
    """
    need(bank['schema_version']==1 and bank['epoch']==59 and bank['kind']==KIND and bank['role']=='dev','BLIND_DEV_FEATURE_BANK')
    need(bank.get('source_split')=='train' and bank.get('project_role_semantics')==PROJECT_ROLE_SEMANTICS,'ORIGINAL_TRAIN_PROJECT_DEV_ONLY')
    need(len(bank['documents'])==1 and bank['documents'][0]['document_id']==473 and tuple(x['id'] for x in bank['cases'])==DEV_IDS,'ONE_DOC473_ALL17_FIXED_CASES')
    doc=bank['documents'][0];units=doc['evidence'];claims=[x['claim'] for x in bank['cases']]
    need(len(units)>=1 and len(units)<=1024 and all(type(x) is str and x.strip() and len(x.encode())<=65536 for x in claims),'BOUNDED_SOURCE_AND_CLAIMS')
    source=''.join(x['sentence'] for x in units);need(sha(source.encode())==doc['source_document_sha256'] and len(source.encode())<=1048576,'EXACT_SOURCE_DOCUMENT_BODY')
    at=0;fragments=[]
    for u in units:
        need(type(u['start']) is int and type(u['end']) is int and u['start']==at and u['end']>at and source[at:u['end']]==u['sentence'],'EXACT_CONTIGUOUS_ORIGINAL_SOURCE_UNIT');at=u['end']
        pos=u['start']
        while pos<u['end']:
            suffix=source[pos:u['end']]
            if not suffix.strip():
                need(fragments,'LEADING_BLANK_UNIT_REQUIRES_SOURCE_POLICY')
                fragments[-1]['end']=u['end'];fragments[-1]['source_unit_ids']=list(dict.fromkeys(fragments[-1]['source_unit_ids']+[u['unit_id']]))
                pos=u['end'];continue
            if fits_window(suffix,claims,nli,bge):end=u['end']
            else:
                lo=pos+1;hi=u['end'];best=None
                while lo<=hi:
                    middle=(lo+hi)//2
                    if fits_window(source[pos:middle],claims,nli,bge):best=middle;lo=middle+1
                    else:hi=middle-1
                need(best is not None,'NO_VALID_NONEMPTY_SOURCE_FRAGMENT');end=best
            fragments.append({'start':pos,'end':end,'source_unit_ids':[u['unit_id']]});pos=end
    need(at==len(source),'ALL_ORIGINAL_SOURCE_OFFSETS')
    merged=[]
    for f in fragments:
        if merged and fits_window(source[merged[-1]['start']:f['end']],claims,nli,bge):
            merged[-1]['end']=f['end'];merged[-1]['source_unit_ids']=list(dict.fromkeys(merged[-1]['source_unit_ids']+f['source_unit_ids']))
        else:merged.append(dict(f))
    need(1<=len(merged)<=24,'FROZEN_WINDOW_COUNT_CAP24_NO_ADAPTATION')
    windows=[]
    for i,f in enumerate(merged):
        text=source[f['start']:f['end']];need(fits_window(text,claims,nli,bge),'EVERY_CLAIM_WHOLE_WINDOW_CAPACITY')
        windows.append({'window_id':i,'start':f['start'],'end':f['end'],'source_unit_ids':f['source_unit_ids'],'text':text,'text_sha256':sha(text.encode())})
    need(windows[0]['start']==0 and windows[-1]['end']==len(source) and all(a['end']==b['start'] for a,b in zip(windows,windows[1:])),'EXHAUSTIVE_SHARED_SOURCE_WINDOW_CATALOG')
    for claim in claims:whole_single(bge,QUERY_PREFIX+claim)
    return {'schema_version':1,'epoch':59,'kind':KIND,'role':'dev','source_split':'train','project_role_semantics':PROJECT_ROLE_SEMANTICS,'selected_query_ids':list(DEV_IDS),'reserved_ids':list(DEV_IDS),'document_id':473,'source_document_sha256':doc['source_document_sha256'],'frozen_window_count':len(windows),'windows':windows,'cases':bank['cases'],'all17_pair_capacity_checked':True,'BGE_query_prefix':QUERY_PREFIX,'max_tokens':512,'max_windows':24,'no_label_or_model_output_access':True}


def validate_catalog(catalog):
    need(catalog['schema_version']==1 and catalog['epoch']==59 and catalog['kind']==KIND and catalog['role']=='dev' and catalog['document_id']==473,'EXACT_DEV_CATALOG')
    need(catalog.get('source_split')=='train' and catalog.get('project_role_semantics')==PROJECT_ROLE_SEMANTICS,'ORIGINAL_TRAIN_PROJECT_DEV_CATALOG_ONLY')
    W=catalog['frozen_window_count'];windows=catalog['windows']
    need(type(W) is int and 1<=W<=24 and len(windows)==W and [w['window_id'] for w in windows]==list(range(W)),'FROZEN_WINDOWS_IDENTITY')
    need(catalog['selected_query_ids']==list(DEV_IDS) and catalog['reserved_ids']==list(DEV_IDS) and [x['id'] for x in catalog['cases']]==list(DEV_IDS),'EXACT_CASE_CATALOG')
    at=0;source=[]
    for w in windows:
        need(type(w['start']) is int and type(w['end']) is int and w['start']==at and w['end']>at and len(w['text'])==w['end']-at and sha(w['text'].encode())==w['text_sha256'],'WINDOW_BODY_AND_SOURCE_OFFSETS')
        at=w['end'];source.append(w['text'])
    need(sha(''.join(source).encode())==catalog['source_document_sha256'] and catalog['all17_pair_capacity_checked'] is True and catalog['max_tokens']==512 and catalog['max_windows']==24 and catalog['BGE_query_prefix']==QUERY_PREFIX,'EXHAUSTIVE_UNTRUNCATED_CATALOG_RECEIPT')
    return W


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid;parts=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'startticks':int(parts[19]),'pgid':int(parts[2]),'sid':int(parts[3])}


def memory_snapshot():
    p=Path('/sys/fs/cgroup');limit=(p/'memory.max').read_text().strip();current=int((p/'memory.current').read_text());limit=None if limit=='max' else int(limit)
    return {'cgroup_memory_current_bytes':current,'cgroup_memory_max_bytes':limit,'headroom_bytes':None if limit is None else limit-current,'max_RSS_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def resources():
    r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'self_user_seconds':r.ru_utime,'self_system_seconds':r.ru_stime,'self_max_RSS_kib':r.ru_maxrss,'children_user_seconds':c.ru_utime,'children_system_seconds':c.ru_stime,'children_max_RSS_kib':c.ru_maxrss,'minor_page_faults':r.ru_minflt,'major_page_faults':r.ru_majflt,'voluntary_context_switches':r.ru_nvcsw,'involuntary_context_switches':r.ru_nivcsw,'CPU_affinity':sorted(os.sched_getaffinity(0)),**memory_snapshot()}


class Events:
    def __init__(self,path,deadline):
        self.path=safe(path);self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);self.origin=time.monotonic();self.deadline=deadline;self.values={k:0 for k in COUNTER_KEYS};self.rows=0
    def event(self,event,**extra):
        need(time.monotonic()<self.deadline,'WHOLE_WORKER_DEADLINE');self.rows+=1;need(self.rows<=10000,'PROGRESS_ROW_CAP')
        raw=canonical({'event':event,'elapsed_seconds':time.monotonic()-self.origin,'counters':self.values,**extra})+b'\n';need(os.fstat(self.fd).st_size+len(raw)<=8388608,'PROGRESS_BYTE_CAP');offset=0
        while offset<len(raw):
            n=os.write(self.fd,raw[offset:]);need(n>0,'PROGRESS_SHORT_WRITE');offset+=n
        os.fsync(self.fd)
    def attempt(self,base,n=1):
        need(type(n) is int and n>0 and base+'_attempted' in self.values,'KNOWN_COUNTER');self.values[base+'_attempted']+=n;self.event(base+'_BEFORE',amount=n)
    def complete(self,base,n=1):
        need(type(n) is int and n>0 and self.values[base+'_completed']+n<=self.values[base+'_attempted'],'COMPLETION_NOT_INVENTED');self.values[base+'_completed']+=n;self.event(base+'_AFTER',amount=n)
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
