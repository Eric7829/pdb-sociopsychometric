"""Phase 5 audits over crowd vote distributions, text modality and the Stage 3 agent battery.

Plug-in entropies on ~500 profiles and 16-24 classes are biased low, so every dependence statistic is reported against a
label-permutation null (same marginals, dependence destroyed) rather than against zero.
"""
from collections import Counter,defaultdict
import json
import math
import random
import statistics as st

SYSTEMS=('MBTI','ENNEAGRAM','INSTINCT','TRITYPE','AP')
PAIRS=((('MBTI',),'ENNEAGRAM'),(('MBTI',),'ENNEAGRAM_CORE'),(('MBTI',),'AP'),(('MBTI','ENNEAGRAM'),'AP'),(('MBTI',),'INSTINCT'),(('ENNEAGRAM',),'TRITYPE'))
LETTERS={'E-I':('E','I'),'S-N':('S','N'),'T-F':('T','F'),'J-P':('J','P')}
# Crowd letter share vs IPIP domain; sign = expected direction (e.g. more T votes, lower Agreeableness).
CROWD_IPIP={'E':('E-I','E',1),'O':('S-N','N',1),'A':('T-F','T',-1),'C':('J-P','J',1)}

def entropy(counts):
    n=sum(counts.values())
    return -sum(c/n*math.log2(c/n) for c in counts.values() if c) if n else 0.0

def conditional_entropy(xs,ys):
    groups=defaultdict(Counter)
    for x,y in zip(xs,ys): groups[x][y]+=1
    n=len(xs)
    return sum(sum(c.values())/n*entropy(c) for c in groups.values())

def top2(counts):
    ranked=sorted(counts.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    return ranked[0],(ranked[1] if len(ranked)>1 else (None,0))

def loo_pick(counts,top,y):
    """Mode of counts with one y removed, ties broken by the larger key (as max(key=(count,label)) would)."""
    (t,c1),(s,c2)=top
    if y!=t: return t
    if c1-1>c2 or (c1-1==c2 and t>s): return t if c1>1 else s
    return s

def loo_accuracy(xs,ys):
    """Leave-one-out accuracy of predicting y by the most common y among other profiles sharing x (global mode if unseen)."""
    groups=defaultdict(Counter); overall=Counter(ys)
    for x,y in zip(xs,ys): groups[x][y]+=1
    tops={x:top2(g) for x,g in groups.items()}; otop=top2(overall)
    hits=0
    for x,y in zip(xs,ys):
        pick=loo_pick(groups[x],tops[x],y) if sum(groups[x].values())>1 else loo_pick(overall,otop,y)
        hits+=pick==y
    return hits/len(xs)

def cramers_v(xs,ys):
    n=len(xs); cx,cy,cxy=Counter(xs),Counter(ys),Counter(zip(xs,ys))
    chi=sum((cxy[(a,b)]-cx[a]*cy[b]/n)**2/(cx[a]*cy[b]/n) for a in cx for b in cy)
    k=min(len(cx),len(cy))-1
    return math.sqrt(chi/(n*k)) if k>0 else float('nan')

def bundling(xs,ys,*,permutations=1000,seed=0):
    rng=random.Random(seed); h=conditional_entropy(xs,ys); acc=loo_accuracy(xs,ys)
    null_h=[]; null_acc=[]; shuffled=list(xs)
    for _ in range(permutations):
        rng.shuffle(shuffled); null_h.append(conditional_entropy(shuffled,ys)); null_acc.append(loo_accuracy(shuffled,ys))
    null_h.sort(); null_acc.sort()
    return {'n':len(xs),'classes_x':len(set(xs)),'classes_y':len(set(ys)),'H_y':entropy(Counter(ys)),'H_y_given_x':h,
            'null_H_y_given_x_mean':st.fmean(null_h),'null_H_y_given_x_2.5pct':null_h[int(.025*permutations)],
            'information_gain_over_null':st.fmean(null_h)-h,'p_H':(1+sum(v<=h for v in null_h))/(1+permutations),
            'loo_accuracy':acc,'null_loo_accuracy_mean':st.fmean(null_acc),'null_loo_accuracy_97.5pct':null_acc[int(.975*permutations)-1],
            'p_accuracy':(1+sum(v>=acc for v in null_acc))/(1+permutations),'cramers_v':cramers_v(xs,ys),
            'bundled_over_80pct':acc>.8}

def ranks(xs):
    order=sorted(range(len(xs)),key=xs.__getitem__); r=[0.0]*len(xs); i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[order[j+1]]==xs[order[i]]: j+=1
        for k in range(i,j+1): r[order[k]]=(i+j)/2
        i=j+1
    return r

def spearman(xs,ys):
    try: return st.correlation(ranks(xs),ranks(ys))
    except st.StatisticsError: return float('nan')

def residuals(ys,xs):
    """Residuals of a least-squares fit ys ~ xs (used for rank partial correlation)."""
    slope,icept=st.linear_regression(xs,ys)
    return [y-(slope*x+icept) for x,y in zip(xs,ys)]

def partial_spearman(xs,ys,zs):
    rx,ry,rz=ranks(xs),ranks(ys),ranks(zs)
    try: return st.correlation(residuals(rx,rz),residuals(ry,rz))
    except st.StatisticsError: return float('nan')

def permutation_p(stat,xs,ys,*,extra=None,permutations=1000,seed=0):
    rng=random.Random(seed); obs=stat(xs,ys,*([extra] if extra else [])); shuffled=list(xs); more=0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        more+=abs(stat(shuffled,ys,*([extra] if extra else [])))>=abs(obs)
    return obs,(1+more)/(1+permutations)

def jungian_to_mbti(label):
    """Classic Jungian 'IF(N)' -> 'INFP': attitude + dominant (auxiliary). Derived for analysis only; never used for vote subtraction."""
    if len(label)!=5 or label[0] not in 'IE' or label[2]!='(' or label[4]!=')': return None
    att,dom,aux=label[0],label[1],label[3]
    judging={'T','F'}
    if (dom in judging)==(aux in judging): return None
    sn=dom if dom in 'SN' else aux; tf=dom if dom in judging else aux
    jp=('J' if dom in judging else 'P') if att=='E' else ('P' if dom in judging else 'J')
    return att+sn+tf+jp

def load_profiles(c,samples=('top',)):
    """profile_num -> {'dist': {system: adjusted counts}, 'eligible': set, 'self_votes': {system: label}, 'ramble': float|None}.
    samples: crawl_queue sample tags to include (supplemental samples such as type-selected 'es' bias the marginals)."""
    out={}
    marks=','.join('?'*len(samples))
    for n,ramble in c.execute(f'''SELECT p.profile_num,p.ramble_word_ratio FROM profiles p LEFT JOIN crawl_queue q USING(profile_num)
                                  WHERE p.complete AND coalesce(q.sample,'top') IN ({marks})''',list(samples)).fetchall():
        out[n]={'dist':{},'eligible':set(),'n':{},'entropy':{},'self_votes':{},'ramble':ramble}
    for n,system,counts,total,h,eligible,evidence in c.execute('''SELECT profile_num,system_name,vote_counts_json,total_system_votes,
            shannon_entropy,training_eligible,evidence_json FROM vote_distributions''').fetchall():
        if n not in out: continue
        p=out[n]; p['dist'][system]=json.loads(counts); p['n'][system]=total; p['entropy'][system]=h
        if eligible: p['eligible'].add(system)
        labels={e['label'] for e in json.loads(evidence or '[]') if e.get('confirmed')}
        if len(labels)==1: p['self_votes'][system]=next(iter(labels))
    return out

def modal(counts):
    return max(counts,key=lambda k:(counts[k],k)) if counts and sum(counts.values()) else None

def label(p,system):
    if system=='ENNEAGRAM_CORE':
        m=label(p,'ENNEAGRAM'); return m[0] if m else None
    return modal(p['dist'][system]) if system in p['eligible'] else None

def bundling_audit(profiles,permutations=1000):
    out={}
    for given,target in PAIRS:
        rows=[(tuple(label(p,s) for s in given),label(p,target)) for p in profiles.values()]
        rows=[(x,y) for x,y in rows if None not in x and y is not None]
        if len(rows)<20: out['|'.join(given)+'->'+target]={'n':len(rows),'skipped':'fewer than 20 profiles'}; continue
        out['|'.join(given)+'->'+target]=bundling([x for x,_ in rows],[y for _,y in rows],permutations=permutations)
    return out

def modality_audit(profiles,permutations=1000):
    out={}
    for system in ('MBTI','ENNEAGRAM','INSTINCT','TRITYPE','AP'):
        rows=[(p['ramble'],p['entropy'][system],math.log(1+p['n'][system])) for p in profiles.values()
              if system in p['eligible'] and p['ramble'] is not None and p['entropy'].get(system) is not None]
        if len(rows)<20: out[system]={'n':len(rows),'skipped':'fewer than 20 profiles'}; continue
        r,h,logn=map(list,zip(*rows))
        rho,p=permutation_p(spearman,r,h,permutations=permutations)
        prho,pp=permutation_p(partial_spearman,r,h,extra=logn,permutations=permutations)
        out[system]={'n':len(rows),'spearman_ramble_vs_entropy':rho,'p':p,'partial_spearman_given_log_votes':prho,'p_partial':pp,
                     'hypothesis':'rambles -> lower entropy (negative rho)','supported':prho<0 and pp<.05}
    return out

def letter_share(counts,dichotomy,letter):
    i=list(LETTERS).index(dichotomy); total=sum(counts.values())
    return sum(v for k,v in counts.items() if len(k)==4 and k[i]==letter)/total if total else None

def inventory_audit(profiles,c,prompt_version='stage3-v1'):
    rows=c.execute('''SELECT r.profile_num,e.agent_self_mbti,e.form_m_mbti,e.ipip_domains_json,e.ipip_facets_json FROM agent_runs r
                      JOIN psychometric_evaluations e USING(run_id) WHERE r.complete AND r.prompt_version=? ORDER BY r.profile_num,r.seed''',[prompt_version]).fetchall()
    agent={}
    for n,self_mbti,form_m,domains,facets in rows:
        agent.setdefault(n,{'self':self_mbti,'form_m':form_m,'domains':json.loads(domains),'facets':json.loads(facets)})
    if not agent: return {'skipped':f'no complete {prompt_version} runs'}
    def letters(a,b): return sum(x==y for x,y in zip(a,b))/4
    pairs=defaultdict(list)
    for n,a in agent.items():
        p=profiles.get(n)
        if not p: continue
        crowd=label(p,'MBTI')
        sv=p['self_votes'].get('MBTI') or jungian_to_mbti(p['self_votes'].get('JUNGIAN',''))
        if crowd:
            pairs['crowd_vs_form_m'].append(letters(crowd,a['form_m'])); pairs['crowd_vs_agent_self'].append(letters(crowd,a['self']))
        if sv:
            pairs['self_vote_vs_form_m'].append(letters(sv,a['form_m'])); pairs['self_vote_vs_agent_self'].append(letters(sv,a['self']))
            if crowd: pairs['self_vote_vs_crowd'].append(letters(sv,crowd))
        pairs['agent_self_vs_form_m'].append(letters(a['self'],a['form_m']))
    agreement={k:{'n':len(v),'mean_letter_agreement':st.fmean(v),'exact_type':sum(x==1 for x in v)/len(v)} for k,v in pairs.items() if v}
    convergent={}
    for domain,(dichotomy,letter_,sign) in CROWD_IPIP.items():
        rows=[(letter_share(profiles[n]['dist']['MBTI'],dichotomy,letter_),a['domains'][domain]['raw']) for n,a in agent.items()
              if n in profiles and 'MBTI' in profiles[n]['eligible']]
        rows=[r for r in rows if r[0] is not None]
        if len(rows)>=20:
            rho,p=permutation_p(spearman,[x for x,_ in rows],[y for _,y in rows])
            convergent[f'crowd_{letter_}_share_vs_IPIP_{domain}']={'n':len(rows),'spearman':rho,'p':p,'expected_sign':sign,
                                                                    'in_expected_direction':rho*sign>0}
    # Spec hypothesis: is the T (e.g. INTP) minority vote share driven by low Agreeableness facets (A3 Altruism, A5 Modesty)?
    facets={}
    for facet in ('A3','A5'):
        rows=[(letter_share(profiles[n]['dist']['MBTI'],'T-F','T'),a['facets'][facet]['raw']) for n,a in agent.items()
              if n in profiles and 'MBTI' in profiles[n]['eligible']]
        rows=[r for r in rows if r[0] is not None]
        if len(rows)>=20:
            rho,p=permutation_p(spearman,[x for x,_ in rows],[y for _,y in rows])
            facets[f'crowd_T_share_vs_{facet}']={'n':len(rows),'spearman':rho,'p':p,'expected':'negative'}
    return {'prompt_version':prompt_version,'profiles':len(agent),'agreement':agreement,'crowd_vs_ipip':convergent,'agreeableness_facets':facets,
            'caveat':'Agent read the raw corpus, which can contain self-typing jargon; agreement with self-votes is partly echo.'}

def run_audits(c,*,permutations=1000,prompt_version='stage3-v1',samples=('top',)):
    profiles=load_profiles(c,samples)
    return {'profiles':len(profiles),'samples':list(samples),'permutations':permutations,'bundling':bundling_audit(profiles,permutations),
            'modality':modality_audit(profiles,permutations),'inventory':inventory_audit(profiles,c,prompt_version)}
