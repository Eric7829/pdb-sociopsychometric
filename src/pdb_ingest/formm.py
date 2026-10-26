"""Python port of `MBTI Form M/scorer.js` (2PL IRT, Newton-Raphson MLE per dichotomy)."""
from functools import cache
import json
import math
from pathlib import Path
import re

FORM_DIR = Path(__file__).resolve().parents[2]/'MBTI Form M'
DICHOTOMIES = {'E-I':('E','I','I'),'S-N':('S','N','N'),'T-F':('T','F','F'),'J-P':('J','P','P')}

def questions_path():
    """The licensed item text (questions.json, kept local) when present, else the placeholder with the same keys and poles."""
    real=FORM_DIR/'questions.json'
    return real if real.exists() else FORM_DIR/'questions.placeholder.json'

@cache
def load_questions():
    items=json.loads(questions_path().read_text(encoding='utf-8'))['MBTI_Form_M']
    if [q['number'] for q in items]!=list(range(1,len(items)+1)): raise ValueError('Form M items out of order')
    return items

@cache
def load_parameters():
    text=(FORM_DIR/'itemParameterMatrix.js').read_text(encoding='utf-8')
    body=re.search(r'export const itemParameters\s*=\s*(\{.*\});',text,re.S).group(1)
    return {int(k)+1:v for k,v in json.loads(body).items()}

def ci_category(ci):
    return 'Crystal' if ci>=52 else 'Defined' if ci>=32 else 'Variable' if ci>=12 else 'Diffuse'

def theta(items):
    t=0.0
    for _ in range(20):
        first=second=0.0
        for a,b,u in items:
            p=1/(1+math.exp(-a*(t-b)))
            first+=a*(u-p); second-=a*a*p*(1-p)
        if abs(second)<1e-9: break
        new=max(-3,min(3,t-first/second))
        change=abs(new-t); t=new
        if change<1e-4: break
    return t

def score(answers):
    """answers: {item number: 'A'|'B'}; omitted items are excluded, as in scorer.js."""
    questions=load_questions(); params=load_parameters()
    results={}
    for name,(pos,neg,tie) in DICHOTOMIES.items():
        items=[(params[n]['params']['a'],params[n]['params']['b'],questions[n-1]['options'][answers[n]]['scoreKey'])
               for n in params if params[n]['dichotomy']==name and answers.get(n)]
        t=theta(items) if items else 0
        preference=pos if t>0 else neg if t<0 else tie
        # JS Math.round rounds halves up; Python's round() would round them to even.
        ci=1 if t==0 else max(1,math.floor(abs(t)/3*60+.5))
        if (name,preference)==('S-N','S') and ci<=2: preference='N'
        elif (name,preference)==('T-F','T') and ci<=4: preference='F'
        elif (name,preference)==('J-P','J') and ci<=2: preference='P'
        results[name]={'preference':preference,'ci':ci,'ciCategory':ci_category(ci),'theta':round(t,2),'dichotomyName':name}
    return {'type':''.join(results[d]['preference'] for d in DICHOTOMIES),'dichotomyResults':results}
