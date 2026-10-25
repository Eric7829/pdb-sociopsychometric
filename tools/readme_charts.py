"""Static SVG charts for README.md (no plotting dependency). Numbers come from docs/results/2026-10-25-*.md.

  python tools/readme_charts.py      # writes docs/img/*.svg
"""
from pathlib import Path

OUT=Path('docs/img'); SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#8a8984'; GRID='#e6e5e0'
BLUE='#2a78d6'; ORANGE='#eb6834'; GREY='#b9b8b2'; FONT="font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"

def svg(w,h,body,title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT} role='img' aria-label='{title}'>"
            f"<defs><pattern id='hatch' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
            f"<rect width='6' height='6' fill='{SURF}'/><line x1='0' y1='0' x2='0' y2='6' stroke='{ORANGE}' stroke-width='2.5'/></pattern></defs>"
            f"<rect width='{w}' height='{h}' rx='12' fill='{SURF}'/>{body}</svg>")

def text(x,y,s,size=13,fill=INK,anchor='start',weight=400):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"

def hbars(name,title,sub,rows,vmax,unit='%',w=760):
    """rows: (label, value, fill, note). Bars start at the baseline with a 4px rounded end."""
    left=250; right=w-90; top=78; bh=26; gap=16; h=top+len(rows)*(bh+gap)+40
    b=[text(24,34,title,18,INK,weight=650),text(24,56,sub,13,INK2)]
    for t in range(0,int(vmax)+1,10 if vmax<=60 else 20):
        x=left+(right-left)*t/vmax; b.append(f"<line x1='{x:.1f}' y1='{top-8}' x2='{x:.1f}' y2='{h-36}' stroke='{GRID}'/>"); b.append(text(x,h-18,f'{t}{unit}',11,MUTED,'middle'))
    for i,(lab,v,fill,note) in enumerate(rows):
        y=top+i*(bh+gap); bw=(right-left)*v/vmax
        b.append(text(left-12,y+bh/2+5,lab,13,INK,'end'))
        b.append(f"<path d='M{left},{y} h{bw-4:.1f} a4,4 0 0 1 4,4 v{bh-8} a4,4 0 0 1 -4,4 h-{bw-4:.1f} z' fill='{fill}'{' stroke=\"'+ORANGE+'\" stroke-width=\"1.5\"' if fill=='url(#hatch)' else ''}/>")
        b.append(text(left+bw+8,y+bh/2+5,f'{v:g}{unit}',13,INK,weight=650))
        if note: b.append(text(left-12,y+bh/2+20,note,10.5,MUTED,'end'))
    (OUT/name).write_text(svg(w,h,''.join(b),title),encoding='utf-8')

def heatmap():
    cores=[('1','Perfectionist',[70,60,33,34,64]),('2','Helper',[89,31,58,51,73]),('3','Achiever',[77,43,61,23,68]),('4','Individualist',[90,20,28,45,88]),
           ('5','Investigator',[80,39,22,28,73]),('6','Loyalist',[75,31,23,42,82]),('7','Enthusiast',[91,15,61,36,74]),('8','Challenger',[71,34,68,11,55]),('9','Peacemaker',[78,29,30,61,74])]
    dom=['Openness','Conscientious.','Extraversion','Agreeable.','Neuroticism']; left=190; top=92; cw=104; ch=34; w=left+5*cw+30; h=top+9*ch+56
    lo,hi=(234,242,251),(14,62,128)
    def col(v):
        t=v/100; return '#%02x%02x%02x'%tuple(round(a+(b-a)*t) for a,b in zip(lo,hi))
    b=[text(24,34,'Each Enneagram type has a Big Five fingerprint',18,INK,weight=650),
       text(24,56,'Mean Big Five percentile by crowd-voted Enneagram core (727 PDB profiles)',13,INK2)]
    for j,d in enumerate(dom): b.append(text(left+j*cw+cw/2,top-12,d,12,INK2,'middle'))
    for i,(c,nm,vals) in enumerate(cores):
        y=top+i*ch; b.append(text(left-12,y+ch/2+5,f'{c} · {nm}',13,INK,'end'))
        for j,v in enumerate(vals):
            x=left+j*cw; b.append(f"<rect x='{x+1}' y='{y+1}' width='{cw-2}' height='{ch-2}' rx='4' fill='{col(v)}'/>")
            b.append(text(x+cw/2,y+ch/2+5,v,13,'#ffffff' if v>=55 else INK,'middle',650 if v>=85 or v<=15 else 400))
    b.append(text(24,h-16,'Darker = higher percentile. Bold = extreme (≥85 or ≤15).',11,MUTED))
    (OUT/'enneagram-big5.svg').write_text(svg(w,h,''.join(b),'Enneagram Big Five heatmap'),encoding='utf-8')

def flips():
    cats=['E/I flip','S/N flip','T/F flip','J/P flip']; crowd=[30,26,23,20]; model=[29,16,21,34]
    w=760; left=70; top=96; bottom=330; h=390; gw=(w-left-40)/4; bw=38; vmax=40
    b=[text(24,34,'PDB voters think in cognitive functions. Text models think in letters.',18,INK,weight=650),
       text(24,56,'Share of near-miss types differing by one letter',13,INK2),
       f"<rect x='{w-310}' y='66' width='12' height='12' rx='3' fill='{BLUE}'/>",text(w-292,77,"Crowd's runner-up vote",12,INK),
       f"<rect x='{w-150}' y='66' width='12' height='12' rx='3' fill='{ORANGE}'/>",text(w-132,77,'Model errors',12,INK)]
    for t in range(0,vmax+1,10):
        y=bottom-(bottom-top)*t/vmax; b.append(f"<line x1='{left}' y1='{y:.1f}' x2='{w-30}' y2='{y:.1f}' stroke='{GRID}'/>"); b.append(text(left-8,y+4,f'{t}%',11,MUTED,'end'))
    for i,c in enumerate(cats):
        cx=left+i*gw+gw/2
        for k,(v,colr) in enumerate(((crowd[i],BLUE),(model[i],ORANGE))):
            x=cx-bw-1+k*(bw+2); hh=(bottom-top)*v/vmax; y=bottom-hh
            b.append(f"<path d='M{x},{bottom} v-{hh-4:.1f} a4,4 0 0 1 4,-4 h{bw-8} a4,4 0 0 1 4,4 v{hh-4:.1f} z' fill='{colr}'/>")
            b.append(text(x+bw/2,y-6,f'{v}%',12,INK,'middle',650))
        b.append(text(cx,bottom+20,c,13,INK,'middle'))
    b.append(text(24,h-14,'In function theory E/I flips keep both functions (INTP↔ENTP) and J/P flips keep none (INTP↔INTJ).',11,MUTED))
    (OUT/'function-flips.svg').write_text(svg(w,h,''.join(b),'Crowd vs model near-miss structure'),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hbars('mbti-accuracy.svg','Guessing your exact MBTI type from your writing','Exact match with the PDB crowd’s 16-type vote',
          [('Random guess',8,GREY,''),('Small local models',16,BLUE,'TF-IDF + gte-modernbert, 5-fold CV'),
           ('DeepSeek “PDB voter” agent',30,BLUE,'typology talk scrubbed out, 785 profiles'),
           ('Same agent, self-typing left in',48,'url(#hatch)','what most "AI typed me" demos measure')],60)
    hbars('essays-big5.svg','Big Five on 2,467 student essays','Median-split accuracy vs the writers’ own questionnaire scores',
          [('Majority baseline',51.4,GREY,''),('Mairesse et al. 2007',56.7,GREY,'trained on these essays'),
           ('Our model',56.3,BLUE,'never saw a single essay'),('Majumder et al. 2017',58.8,GREY,'trained on these essays, best per trait'),
           ('Our DeepSeek agent',59.8,BLUE,'zero-shot, 100 essays')],70)
    heatmap(); flips()

if __name__=='__main__': main()
