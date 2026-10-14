"""Bounded lexical and pinned SWERank CPU retrieval; no repository execution."""
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import time
import types

MODEL_REVISION="745d2a06103a66d3cfa600aa52fc0d3523010daa"
CODE_REVISION="92d97331f1f4b6a366c1f161354b9f3390cc219f"
PREFIX="Represent this query for searching relevant code: "


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text):return re.findall(r"[a-z0-9]+",text.lower().replace("_"," "))


def bm25(query,documents):
    bags=[Counter(tokens(d["text"])) for d in documents]
    lengths=[sum(b.values()) for b in bags];average=sum(lengths)/len(bags)
    df=Counter(t for bag in bags for t in bag)
    result=[]
    for bag,length in zip(bags,lengths):
        score=0.
        for token in set(tokens(query)):
            count=bag[token]
            idf=math.log(1+(len(bags)-df[token]+.5)/(df[token]+.5))
            score+=idf*count*2.5/(count+1.5*(.25+.75*length/average))
        result.append(score)
    return result


def file_ranking(documents,scores):
    if len(documents)!=len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("invalid candidate scores")
    best={}
    for doc,score in zip(documents,scores):best[doc["path"]]=max(best.get(doc["path"],-math.inf),float(score))
    return [dict(path=p,score=s) for p,s in sorted(best.items(),key=lambda x:(-x[1],x[0]))]


def evaluate(ranking,gold):
    paths=[r["path"] for r in ranking]
    if not gold or len(paths)!=len(set(paths)) or not set(gold)<=set(paths):raise ValueError("incomplete candidate universe")
    ranks=[paths.index(p)+1 for p in gold]
    return dict(mrr=1/min(ranks),hit1=int(min(ranks)==1),recall5=sum(r<=5 for r in ranks)/len(ranks),
                recall10=sum(r<=10 for r in ranks)/len(ranks),goldCount=len(gold),goldRanks=ranks)


class Encoder:
    def __init__(self,model_dir,code_dir,expected_hashes,max_tokens=512):
        self.max_tokens=max_tokens
        for path,h in expected_hashes.items():
            if digest(path)!=h:raise ValueError("pinned model/code artifact differs")
        import torch
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        torch.set_num_threads(2)
        self.torch=torch
        # Explicit local import only after provenance verification. Never resolve auto_map remotely.
        package=types.ModuleType("pinned_swerank");package.__path__=[str(Path(code_dir).resolve())]
        if "pinned_swerank" in sys.modules:raise ValueError("encoder code already imported")
        sys.modules["pinned_swerank"]=package
        config_class=importlib.import_module("pinned_swerank.configuration_hf_nomic_bert").NomicBertConfig
        model_class=importlib.import_module("pinned_swerank.modeling_hf_nomic_bert").NomicBertModel
        config=config_class(**json.loads((Path(model_dir)/"config.json").read_text()))
        config.use_flash_attn=False
        self.model=model_class(config,add_pooling_layer=False)
        self.model.load_state_dict(load_file(str(Path(model_dir)/"model.safetensors")),strict=True)
        self.model.eval()
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)

    def encode(self,texts):
        tensors=[];truncated=0;started=time.perf_counter()
        for offset in range(0,len(texts),8):
            group=texts[offset:offset+8]
            lengths=self.tokenizer(group,truncation=False,padding=False)["input_ids"]
            truncated+=sum(len(x)>self.max_tokens for x in lengths)
            inputs=self.tokenizer(group,padding=True,truncation=True,max_length=self.max_tokens,return_tensors="pt")
            with self.torch.inference_mode():
                vectors=self.model(**inputs).last_hidden_state[:,0]
                vectors=self.torch.nn.functional.normalize(vectors,p=2,dim=1)
            if not self.torch.isfinite(vectors).all():raise ValueError("nonfinite embedding")
            tensors.append(vectors)
        return self.torch.cat(tensors),dict(seconds=time.perf_counter()-started,truncated=truncated,count=len(texts))
