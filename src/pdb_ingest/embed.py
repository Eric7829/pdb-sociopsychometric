"""Frozen-encoder segment embeddings, pooled only over encoder_view's pooled character ranges (torch imported lazily)."""
from pathlib import Path
DEFAULT_MODEL='answerdotai/ModernBERT-base'

def windows(n,max_tokens,stride):
    """Overlapping windows over n tokens; each token is owned by exactly one window (the middle of the overlap splits it)."""
    if n<=max_tokens: return [(0,n,0,n)]
    step=max_tokens-stride; out=[]; start=0
    while True:
        end=min(start+max_tokens,n); first=start==0; last=end==n
        own_start=start if first else start+stride//2
        own_end=end if last else end-(stride-stride//2)
        out.append((start,end,own_start,own_end))
        if last: return out
        start+=step

def pooled_tokens(offsets,ranges):
    """Token i is pooled when its character midpoint falls inside a pooled range (special/empty tokens never)."""
    return [b>a and any(r0<=(a+b)/2<r1 for r0,r1 in ranges) for a,b in offsets]

def embed_segments(views,tokenize,encode,*,max_tokens=8192,stride=512):
    """views: [(text, pooled_ranges)]; tokenize(text)->(ids, offsets); encode(ids)->list of token vectors (no specials).
    Returns [(summed vector or None, pooled token count)] per view."""
    out=[]
    for text,ranges in views:
        ids,offsets=tokenize(text); mask=pooled_tokens(offsets,ranges); total=None; count=0
        for start,end,own0,own1 in windows(len(ids),max_tokens,stride):
            vecs=encode(ids[start:end])
            for i in range(own0,own1):
                if mask[i]:
                    v=vecs[i-start]; total=list(v) if total is None else [a+b for a,b in zip(total,v)]; count+=1
        out.append((total,count))
    return out

def profile_vector(segment_results):
    """Mean of all pooled tokens across segments (= segment means weighted by pooled token count)."""
    total=None; count=0
    for vec,n in segment_results:
        if not n: continue
        total=list(vec) if total is None else [a+b for a,b in zip(total,vec)]; count+=n
    return [x/count for x in total] if count else None

def hf_encoder(model_name=DEFAULT_MODEL,device=None):
    """(tokenize, encode, stats) for a Hugging Face encoder; CUDA fp16 when available, per-window CPU fallback on OOM."""
    import torch
    from transformers import AutoModel,AutoTokenizer
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name,torch_dtype=torch.float16 if device=='cuda' else torch.float32).to(device).eval()
    cpu_model=None; stats={'cpu_fallbacks':0,'device':device}
    def tokenize(text):
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        return enc['input_ids'],enc['offset_mapping']
    def run(m,dev,ids):
        x=torch.tensor([[tok.cls_token_id]+list(ids)+[tok.sep_token_id]],device=dev)
        with torch.no_grad(): return m(input_ids=x).last_hidden_state[0,1:-1].float().cpu().tolist()
    def encode(ids):
        nonlocal cpu_model
        try: return run(model,device,ids)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); stats['cpu_fallbacks']+=1
            cpu_model=cpu_model or AutoModel.from_pretrained(model_name,torch_dtype=torch.float32).eval()
            return run(cpu_model,'cpu',ids)
    return tokenize,encode,stats

def cached_vector(cache_dir,profile_num,corpus_sha,compute):
    """Load data/stage2/emb/<view>/<profile>_<sha>.npy if present, else compute() and save; None results are not cached."""
    import numpy as np
    path=Path(cache_dir)/f'{profile_num}_{corpus_sha[:16]}.npy'
    if path.exists(): return np.load(path).tolist()
    vec=compute()
    if vec is not None:
        path.parent.mkdir(parents=True,exist_ok=True); np.save(path,np.array(vec,dtype='float32'))
    return vec
