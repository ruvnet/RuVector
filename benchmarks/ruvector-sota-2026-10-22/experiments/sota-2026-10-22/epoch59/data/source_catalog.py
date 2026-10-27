"""Inherited exact-character source catalog and fixed lexical ordering."""
import hashlib
import re
MAX_UNITS=1024


MAX_DOCUMENT_BYTES=1048576


MAX_UNIT_BYTES=262144


MAX_CLAIM_BYTES=65536


MAX_JOINT_TOKENS=512


SEPARATOR='\n\n'


ABBREVIATIONS=frozenset(('mr.','mrs.','ms.','dr.','prof.','inc.','ltd.','corp.',
    'co.','vs.','no.','sec.','art.','e.g.','i.e.','u.s.','u.k.'))


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sentence_catalog(document):
    if type(document)is not str or not document.strip():raise ValueError('SOURCE_DOCUMENT_EMPTY_OR_TYPE')
    if len(document.encode('utf-8'))>MAX_DOCUMENT_BYTES:raise ValueError('SOURCE_DOCUMENT_CAP')
    boundaries=[];length=len(document)
    for match in re.finditer(r'[.!?]',document):
        at=match.start();mark=document[at]
        if mark=='.':
            if at>0 and at+1<length and document[at-1].isdigit()and document[at+1].isdigit():continue
            token=re.search(r'[A-Za-z.]+\.$',document[:at+1])
            word=token.group().lower()if token else ''
            if word in ABBREVIATIONS or re.fullmatch(r'[a-z]\.',word):continue
        end=at+1
        while end<length and document[end]in '\"\'”’)]}':end+=1
        if end<length and not document[end].isspace():continue
        while end<length and document[end].isspace():end+=1
        if end<length:boundaries.append(end)
    boundaries=sorted(set([0,*boundaries,length]));spans=[]
    for start,end in zip(boundaries,boundaries[1:]):
        text=document[start:end]
        if not text:continue
        if len(text.encode('utf-8'))>MAX_UNIT_BYTES:raise ValueError('SOURCE_UNIT_CAP_NO_SPLITTING')
        spans.append({'unit_id':len(spans),'start':start,'end':end,'sentence':text})
    if len(spans)>MAX_UNITS:raise ValueError('SOURCE_UNIT_COUNT_CAP')
    if ''.join(row['sentence']for row in spans)!=document:raise AssertionError('SOURCE_PARTITION')
    return spans


def validate_spans(spans):
    if type(spans)is not list or not 1<=len(spans)<=MAX_UNITS:raise ValueError('SOURCE_UNIT_COUNT_CAP')
    end=0
    for index,row in enumerate(spans):
        if (set(row)!={'unit_id','start','end','sentence'}or type(row['unit_id'])is not int
            or row['unit_id']!=index or type(row['start'])is not int or type(row['end'])is not int
            or row['start']!=end or type(row['sentence'])is not str or not row['sentence']
            or row['end']!=row['start']+len(row['sentence'])):raise ValueError('NONCONTIGUOUS_SOURCE_CATALOG')
        if len(row['sentence'].encode('utf-8'))>MAX_UNIT_BYTES:raise ValueError('SOURCE_UNIT_CAP_NO_SPLITTING')
        end=row['end']
    if len(''.join(row['sentence']for row in spans).encode('utf-8'))>MAX_DOCUMENT_BYTES:raise ValueError('SOURCE_DOCUMENT_CAP')


def lexical_order(spans,claim):
    query=set(re.findall(r'\w+',claim.lower()))
    def key(row):
        terms=set(re.findall(r'\w+',row['sentence'].lower()));union=query|terms
        return (-(len(query&terms)/len(union)if union else 0),row['unit_id'])
    return [row['unit_id']for row in sorted(spans,key=key)]


