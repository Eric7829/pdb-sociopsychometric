import json
import pytest
from pdb_ingest import distill
from pdb_ingest.store import Store

def seed(store,n,system,counts,eligible=True,complete=True):
    c=store.connection
    if not c.execute('select 1 from profiles where profile_num=?',[n]).fetchone():
        c.execute("insert into profiles(profile_num,profile_slug,complete,snapshot_json) values (?,?,?,'{}')",[n,str(n),complete])
    c.execute('insert into vote_distributions(profile_num,system_name,vote_counts_json,total_system_votes,training_eligible) values (?,?,?,?,?)',
              [n,system,json.dumps(counts),sum(counts.values()),eligible])

def test_class_lists():
    c=distill.class_lists()
    assert len(c['MBTI'])==16 and len(c['ENNEAGRAM'])==18 and len(c['AP'])==24 and '4w5' in c['ENNEAGRAM']

def test_unknown_labels_are_dropped_and_counted(tmp_path):
    with Store(tmp_path/'d.duckdb') as store:
        seed(store,1,'MBTI',{'INFP':8,'INTP':4,'XNFP':3})
        targets,dropped=distill.load_targets(store.connection)
    p,h,n=targets[1]['MBTI']
    assert n==12 and dropped['MBTI']==3 and sum(p)==pytest.approx(1)
    assert p[distill.class_lists()['MBTI'].index('INFP')]==pytest.approx(8/12)
    assert h==pytest.approx(-(2/3)*__import__('math').log2(2/3)-(1/3)*__import__('math').log2(1/3))

def test_profile_without_eligible_heads_is_excluded(tmp_path):
    with Store(tmp_path/'d.duckdb') as store:
        seed(store,1,'MBTI',{'INFP':5},eligible=False)
        seed(store,2,'MBTI',{'INFP':20})
        seed(store,3,'MBTI',{'INFP':20},complete=False)
        seed(store,4,'MBTI',{'INFP':9,'XXXX':5})  # only 9 votes left after dropping -> below 10
        targets,_=distill.load_targets(store.connection)
    assert set(targets)=={2}

from pdb_ingest import embed

def test_windows_own_each_token_once():
    for n,mx,st in ((10,20,4),(100,20,4),(101,16,6),(57,8,3)):
        owned=[i for _,_,a,b in embed.windows(n,mx,st) for i in range(a,b)]
        assert owned==list(range(n))
        assert all(e-s<=mx for s,e,_,_ in embed.windows(n,mx,st))

def test_long_segment_windows_pool_each_token_once():
    words=[f'w{i}' for i in range(50)]; text=' '.join(words)
    offsets=[]; pos=0
    for w in words: offsets.append((pos,pos+len(w))); pos+=len(w)+1
    ids=list(range(50))
    seen=[]
    def encode(chunk): seen.append(len(chunk)); return [[1.0,float(i)] for i in chunk]
    (vec,count),=embed.embed_segments([(text,[(0,len(text))])],lambda t:(ids,offsets),encode,max_tokens=12,stride=4)
    assert count==50 and vec==[50.0,float(sum(ids))] and max(seen)<=12 and len(seen)>1

def test_pooled_ranges_restrict_tokens():
    text='[Q] why [/Q] because books'
    offsets=[(0,3),(4,7),(8,12),(13,20),(21,26)]
    (vec,count),=embed.embed_segments([(text,[(13,26)])],lambda t:(list(range(5)),offsets),lambda ids:[[float(i)] for i in ids])
    assert count==2 and vec==[7.0]
    assert embed.profile_vector([(vec,count),(None,0),([3.0],1)])==[10/3]

def test_cache_keyed_by_corpus_hash(tmp_path):
    pytest.importorskip('numpy')
    calls=[]
    def compute(): calls.append(1); return [1.0,2.0]
    assert embed.cached_vector(tmp_path,7,'a'*64,compute)==[1.0,2.0]
    assert embed.cached_vector(tmp_path,7,'a'*64,compute)==[1.0,2.0] and len(calls)==1
    embed.cached_vector(tmp_path,7,'b'*64,compute); assert len(calls)==2
    assert embed.cached_vector(tmp_path,8,'c'*64,lambda:None) is None and not (tmp_path/f"8_{'c'*16}.npy").exists()

def test_masked_heads_have_zero_grad():
    torch=pytest.importorskip('torch')
    classes=distill.class_lists()
    model=distill.head_model(4,classes,hidden=8)
    x=torch.randn(3,4); _,t=distill.tensors([1,2,3],{n:x[i].tolist() for i,n in enumerate([1,2,3])},
        {n:{'MBTI':([1/16]*16,4.0,20)} for n in (1,2,3)},classes,torch)
    distill.loss(model(x),t).backward()
    assert model.dist['AP'].weight.grad.abs().sum()==0 and model.dist['MBTI'].weight.grad.abs().sum()>0

def test_model_beats_baseline_on_learnable_synthetic():
    torch=pytest.importorskip('torch'); pytest.importorskip('sklearn')
    import random
    rng=random.Random(0); classes={'MBTI':distill.class_lists()['MBTI']}
    X={}; targets={}
    W=[[rng.gauss(0,1.5) for _ in range(6)] for _ in range(16)]
    for n in range(300):
        x=[rng.gauss(0,1) for _ in range(6)]; logits=[sum(a*b for a,b in zip(w,x)) for w in W]
        m=max(logits); e=[__import__('math').exp(l-m) for l in logits]; p=[v/sum(e) for v in e]
        X[n]=x; targets[n]={'MBTI':(p,-sum(v*__import__('math').log2(v) for v in p if v),30)}
    r=distill.cross_validate(X,targets,classes,epochs=400,patience=40,hidden=64)['MBTI']
    assert r['model']['kl']<r['baseline']['kl']*.6 and r['model']['top1']>r['baseline']['top1']

def test_run_distill_end_to_end_with_fakes(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('sklearn'); pytest.importorskip('numpy')
    import random
    rng=random.Random(1); types=['INFP','INTP','ENFP','ISTJ']
    with Store(tmp_path/'d.duckdb') as store:
        for n in range(1,61):
            seed(store,n,'MBTI',{types[n%4]:15,'ENTJ':rng.randint(0,3)})
        views=lambda store_,n,**kw:[(f'profile kind {n%4} words',[(0,10)])]
        tokenize=lambda text:([len(w) for w in text.split()],[(0,7),(8,12),(13,14),(15,20)])
        encode=lambda ids:[[float(i),float(i%3)] for i in ids]
        report=distill.run_distill(store,'main',views_fn=views,encoder=(tokenize,encode,{'device':'cpu','cpu_fallbacks':0}),
                                   out_root=tmp_path/'stage2',folds=3,epochs=50)
    assert report['view']=='main' and report['embedded']==60 and report['targets']==60
    assert set(report['metrics'])=={'MBTI','ENNEAGRAM','AP'} and report['metrics']['ENNEAGRAM']=={'n':0}
    assert (tmp_path/'stage2'/'main'/'report.json').exists() and list((tmp_path/'stage2'/'emb'/'custom'/'main').glob('*.npy'))

def test_validation_split_is_shuffled_and_deterministic():
    train=list(range(100,170))
    val,fit=distill.split_validation(train,seed=0)
    assert len(val)==10 and sorted(val+fit)==train and val!=train[:10]
    assert distill.split_validation(train,seed=0)==(val,fit)

def test_oom_falls_back_to_cpu():
    class OOM(Exception): pass
    stats={'cpu_fallbacks':0}
    def gpu(ids): raise OOM()
    encode=embed.with_cpu_fallback(gpu,lambda ids:[[1.0] for _ in ids],stats,(OOM,))
    assert encode([1,2])==[[1.0],[1.0]] and stats['cpu_fallbacks']==1

def test_cli_audit_and_distill_are_wired(tmp_path):
    from pdb_ingest.cli import main
    import contextlib,io
    with contextlib.redirect_stderr(io.StringIO()):
        for cmd in ('audit','distill'):
            with pytest.raises(SystemExit) as e: main([cmd,'--help'])
            assert e.value.code==0

def test_prefix_tokens_are_never_pooled():
    def tokenize(text):  # whitespace tokenizer with character offsets
        out=[];pos=0
        for w in text.split(' '): out.append((pos,pos+len(w))); pos+=len(w)+1
        return list(range(len(out))),out
    tok=embed.prefixed(tokenize,'passage: ')
    ids,offsets=tok('hello world')
    assert len(ids)==3 and offsets[0][1]<=0 and offsets[1:]==[(0,5),(6,11)]
    (vec,n),=embed.embed_segments([('hello world',[(0,11)])],tok,lambda ids:[[float(i)] for i in ids])
    assert n==2 and vec==[3.0]

def test_segment_normalization_weights_unit_vectors_by_length():
    views=[('a b',[(0,3)]),('c',[(0,1)])]
    tokenize=lambda t:(list(range(len(t.split()))),[(i*2,i*2+1) for i in range(len(t.split()))])
    res=embed.embed_segments(views,tokenize,lambda ids:[[3.0,4.0] for _ in ids],normalize_segments=True)
    assert [n for _,n in res]==[2,1] and embed.profile_vector(res)==pytest.approx([0.6,0.8])

def test_cache_path_includes_model():
    assert embed.model_slug('intfloat/multilingual-e5-small')=='intfloat--multilingual-e5-small'

def test_stack_averages_zscored_predictions_without_fitting():
    import importlib.util,numpy as np
    spec=importlib.util.spec_from_file_location('ctm',embed.Path(__file__).resolve().parents[1]/'tools'/'compare_text_models.py')
    ctm=importlib.util.module_from_spec(spec); spec.loader.exec_module(ctm)
    a=np.array([1.,2.,3.]); b=np.array([10.,30.,20.])
    assert np.allclose(ctm.stack([a,b]),(ctm.zscore(a)+ctm.zscore(b))/2) and np.allclose(ctm.zscore([5,5,5]),0)
