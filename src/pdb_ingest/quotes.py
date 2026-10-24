"""Quoted text inside creator comments: '>' lines and runs of >=6 words shared with other users' thread comments."""
import re
from .models import Span

QUOTE_MARKS='"“”\''

def ngrams(tokens,n):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def merge_ranges(ranges):
    out=[]
    for a,b in sorted(ranges):
        if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],b))
        else: out.append((a,b))
    return out

def extend_to_marks(text,a,b):
    """Grow (a,b) to enclosing quotation marks on the same line, skipping only spaces/tabs between."""
    i=a
    while i>0 and text[i-1] in ' \t': i-=1
    if i>0 and text[i-1] in QUOTE_MARKS: a=i-1
    j=b
    while j<len(text) and text[j] in ' \t': j+=1
    if j<len(text) and text[j] in QUOTE_MARKS: b=j+1
    return a,b

def quote_ranges(text,others,n=6):
    ranges=[(m.start(),m.end()) for m in re.finditer(r'(?m)^[ \t]*>.*?(?=\r?$)',text)]
    tokens=[(m.group(),m.start(),m.end()) for m in re.finditer(r'\w+',text.lower())]
    shared=set().union(*(ngrams(re.findall(r'\w+',o.lower()),n) for o in others)) if others else set()
    words=[t for t,_,_ in tokens]; marked=[False]*len(tokens)
    for i in range(len(tokens)-n+1):
        if tuple(words[i:i+n]) in shared:
            for k in range(i,i+n): marked[k]=True
    i=0
    while i<len(tokens):
        if marked[i]:
            j=i
            while j+1<len(tokens) and marked[j+1]: j+=1
            ranges.append(extend_to_marks(text,tokens[i][1],tokens[j][2])); i=j+1
        else: i+=1
    return merge_ranges(ranges)

def merge_quotes(spans,quotes,text=None):
    """QUOTE wins over overlapping spans; remaining pieces keep their kind (whitespace-trimmed when text is given)."""
    out=[Span('QUOTE',a,b) for a,b in quotes]
    for s in spans:
        pieces=[(s.start,s.end)]
        for a,b in quotes:
            pieces=[q for x,y in pieces for q in ((x,min(y,a)),(max(x,b),y)) if q[1]>q[0]] if a<s.end and b>s.start else pieces
        for x,y in pieces:
            if text is not None:
                piece=text[x:y]; x+=len(piece)-len(piece.lstrip()); y-=len(piece)-len(piece.rstrip())
            if y>x: out.append(Span(s.kind,x,y))
    return sorted(out,key=lambda s:s.start)
