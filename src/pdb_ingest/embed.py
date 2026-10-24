"""Frozen-encoder segment embeddings, pooled only over encoder_view's pooled character ranges (torch imported lazily)."""
from pathlib import Path
DEFAULT_MODEL='intfloat/multilingual-e5-small'
# Document prefixes the models were trained with; prefix tokens are never pooled.
JINA='jinaai/jina-embeddings-v5-text-small-classification'
PREFIXES={'intfloat/multilingual-e5-small':'passage: ','intfloat/multilingual-e5-base':'passage: ',JINA:'Document: '}
# Last-token models (decoder embedders) summarise a whole input in its final token, so span-masked mean pooling does not apply:
# they embed only the pooled text of each segment, chunked.
POOLING={JINA:'last'}
# Chunk size for last-token models, picked for an 8 GB GPU: attention memory grows ~quadratically without flash attention
# (2048 tokens: 0.72 s, 2.0 GB; 8192 tokens: 11.6 GB, spilled into system RAM, 171 s).
CHUNK_TOKENS={JINA:2048}
# Matryoshka (MRL) models: the leading k dimensions are themselves a trained embedding, so slicing is a principled way to
# shrink features for small samples.
MATRYOSHKA_DIMS={JINA:(32,64,128,256,512,768,1024)}
# Cap PyTorch's share of GPU memory so an oversized input raises OOM (then CPU fallback) instead of Windows silently spilling
# into system RAM and slowing ~200x.
GPU_MEMORY_FRACTION=.85

def matryoshka(v,k):
    """Leading k dimensions, re-normalised to unit length (rows of a 2-D array, or one vector); k=None returns v unchanged."""
    if k is None: return v
    import numpy as np
    a=np.asarray(v,dtype=float)[...,:k]; n=np.linalg.norm(a,axis=-1,keepdims=True); a=a/np.where(n==0,1,n)
    return a.tolist() if isinstance(v,list) else a

def model_slug(name):
    return name.replace('/','--')

def prefixed(tokenize,prefix):
    """Tokenize prefix+text but report offsets relative to text, so prefix tokens fall outside every pooled range."""
    if not prefix: return tokenize
    def run(text):
        ids,offsets=tokenize(prefix+text)
        return ids,[(a-len(prefix),b-len(prefix)) for a,b in offsets]
    return run

def windows(n,max_tokens,stride):
    """Overlapping windows over n tokens; each token is owned by exactly one window (the middle of the overlap splits it)."""
    if n<=max_tokens: return [(0,n,0,n)]
    if not 0<=stride<max_tokens: raise ValueError('stride must be smaller than max_tokens')
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

def embed_segments(views,tokenize,encode,*,max_tokens=8192,stride=512,normalize_segments=False):
    """views: [(text, pooled_ranges)]; tokenize(text)->(ids, offsets); encode(ids)->token vectors (no specials).
    Returns [(summed vector or None, pooled token count)] per view. With normalize_segments, the "sum" is the segment's
    unit-length mean vector times its count, so profile_vector gives a length-weighted mean of unit segment vectors."""
    import numpy as np
    out=[]
    for text,ranges in views:
        ids,offsets=tokenize(text); mask=np.array(pooled_tokens(offsets,ranges),dtype=bool); total=None; count=0
        for start,end,own0,own1 in windows(len(ids),max_tokens,stride):
            keep=mask[own0:own1]
            if not keep.any(): continue
            vecs=np.asarray(encode(ids[start:end]),dtype=np.float64)[own0-start:own1-start][keep]
            total=vecs.sum(0) if total is None else total+vecs.sum(0); count+=int(keep.sum())
        if total is not None and normalize_segments:
            norm=np.linalg.norm(total/count); total=total/count/(norm or 1)*count
        out.append((None if total is None else total.tolist(),count))
    return out

def last_token_segments(views,tokenize,embed_ids,*,max_tokens=8192):
    """Last-token pooling: embed only each segment's pooled text, in chunks of max_tokens; the segment vector is the
    token-weighted mean of unit chunk vectors, renormalised and multiplied by the token count (as in embed_segments)."""
    import numpy as np
    out=[]
    for text,ranges in views:
        ids,_=tokenize(' '.join(text[a:b] for a,b in ranges).strip())
        if not ids: out.append((None,0)); continue
        total=None
        for i in range(0,len(ids),max_tokens):
            chunk=ids[i:i+max_tokens]; v=np.asarray(embed_ids(chunk),dtype=np.float64); v=v/(np.linalg.norm(v) or 1)
            total=v*len(chunk) if total is None else total+v*len(chunk)
        total=total/len(ids); total=total/(np.linalg.norm(total) or 1)*len(ids)
        out.append((total.tolist(),len(ids)))
    return out

def embed_views(views,encoder,*,normalize_segments=True):
    """Segment results for one profile's encoder views, dispatching on the encoder's pooling type."""
    tokenize,encode,stats=encoder; mx=stats.get('max_tokens',8192)
    if stats.get('pooling')=='last': return last_token_segments(views,tokenize,encode,max_tokens=mx)
    return embed_segments(views,tokenize,encode,max_tokens=mx,stride=min(512,mx//4),normalize_segments=normalize_segments)

def profile_vector(segment_results):
    """Mean of all pooled tokens across segments (= segment means weighted by pooled token count)."""
    total=None; count=0
    for vec,n in segment_results:
        if not n: continue
        total=list(vec) if total is None else [a+b for a,b in zip(total,vec)]; count+=n
    return [x/count for x in total] if count else None

def with_cpu_fallback(gpu,cpu,stats,oom_errors):
    """encode(ids) that retries one window on CPU when the GPU path raises an out-of-memory error."""
    def encode(ids):
        try: return gpu(ids)
        except oom_errors:
            stats['cpu_fallbacks']+=1
            return cpu(ids)
    return encode

def hf_encoder(model_name=DEFAULT_MODEL,device=None,prefix=None):
    """(tokenize, encode, stats) for a Hugging Face encoder; CUDA fp16 when available, per-window CPU fallback on OOM."""
    import torch
    from transformers import AutoModel,AutoTokenizer
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda': torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name,torch_dtype=torch.float16 if device=='cuda' else torch.float32).to(device).eval()
    if POOLING.get(model_name)=='last':
        prefix=PREFIXES.get(model_name,'') if prefix is None else prefix
        stats={'cpu_fallbacks':0,'device':device,'model':model_name,'pooling':'last','max_tokens':CHUNK_TOKENS.get(model_name,2048)}
        def tokenize(text): return tok(text,add_special_tokens=False)['input_ids'],[]
        def embed_ids(ids,m=model,dev=device):
            enc=tok(prefix+tok.decode(ids),return_tensors='pt').to(dev)   # the tokenizer adds the model's own special tokens
            with torch.no_grad(): return m(**enc).last_hidden_state[0,-1].float().cpu().tolist()
        cpu_holder={}
        def on_cpu(ids):
            torch.cuda.empty_cache()
            cpu_holder.setdefault('m',AutoModel.from_pretrained(model_name,torch_dtype=torch.float32).eval())
            return embed_ids(ids,cpu_holder['m'],'cpu')
        return tokenize,with_cpu_fallback(embed_ids,on_cpu,stats,(torch.cuda.OutOfMemoryError,)),stats
    # Content tokens per window: the model's position limit minus CLS/SEP.
    cpu_model=None; stats={'cpu_fallbacks':0,'device':device,'model':model_name,'max_tokens':min(tok.model_max_length,8192)-2}
    def raw_tokenize(text):
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        return enc['input_ids'],enc['offset_mapping']
    tokenize=prefixed(raw_tokenize,PREFIXES.get(model_name,'') if prefix is None else prefix)
    def run(m,dev,ids):
        x=torch.tensor([[tok.cls_token_id]+list(ids)+[tok.sep_token_id]],device=dev)
        with torch.no_grad(): return m(input_ids=x).last_hidden_state[0,1:-1].float().cpu().tolist()
    def on_cpu(ids):
        nonlocal cpu_model
        torch.cuda.empty_cache()
        cpu_model=cpu_model or AutoModel.from_pretrained(model_name,torch_dtype=torch.float32).eval()
        return run(cpu_model,'cpu',ids)
    return tokenize,with_cpu_fallback(lambda ids:run(model,device,ids),on_cpu,stats,(torch.cuda.OutOfMemoryError,)),stats

def cached_vector(cache_dir,profile_num,corpus_sha,compute):
    """Load data/stage2/emb/<view>/<profile>_<sha>.npy if present, else compute() and save; None results are not cached."""
    import numpy as np
    path=Path(cache_dir)/f'{profile_num}_{corpus_sha[:16]}.npy'
    if path.exists(): return np.load(path).tolist()
    vec=compute()
    if vec is not None:
        path.parent.mkdir(parents=True,exist_ok=True); np.save(path,np.array(vec,dtype='float32'))
    return vec
