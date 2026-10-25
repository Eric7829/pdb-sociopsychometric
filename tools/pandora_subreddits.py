"""Which PANDORA subreddits are usable test text? Read-only; prints subreddit aggregates only (no users, no comments).

  python tools/pandora_subreddits.py [--min-users 100]

Needs data/external/pandora-full/comments.parquet (the comments CSV converted with DuckDB) and author_profiles.csv.
Per subreddit, over English comments: comments, users, users with an MBTI / Big Five label, words per comment,
first-person rate (I/me/my/mine/myself per word), type-talk share (comments naming an MBTI code, a cognitive function,
MBTI, Myers-Briggs, Enneagram or socionics), and type lift (how much more the most over-represented MBTI type appears
among the subreddit's labelled users than in all labelled users). Kinds:
  typology: type talk >= 5% or a type code / typology word in the name (labels leak into text),
  self-disclosure: first-person rate >= the 75th percentile and type talk < 1%,
  other.
"""
import argparse
import json
from pathlib import Path
import duckdb

ROOT=Path('data/external/pandora-full')
TYPE_TALK=r"(\b[ei][ns][tf][jp]s?\b|\b[ei]?[nsx][tfx][jpx]\b|\b(ni|ne|si|se|ti|te|fi|fe)[- ]?(dom|aux|user|users)?\b.*\b(ni|ne|si|se|ti|te|fi|fe)\b|mbti|myers.?briggs|enneagram|socionics|\b[1-9]w[1-9]\b)"
FIRST=r"\b(i|me|my|mine|myself|im)\b"   # \bi\b also catches i'm, i've, i'd, i'll
NAME=r"(^|_)?([ei][ns][tf][jp]|mbti|enneagram|socionics|personality|typology|jung|mbtimemes|introvert|extrovert)"

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--min-users',type=int,default=100); p.add_argument('--output',type=Path,default=Path('data/stage2/pandora_subreddits.json'))
    args=p.parse_args(); c=duckdb.connect(); c.execute('SET threads=8')
    C=str(ROOT/'comments.parquet').replace('\\','/'); A=str(ROOT/'author_profiles.csv').replace('\\','/')
    c.execute(f"create temp table prof as select author, lower(mbti) mbti, agreeableness is not null big5 from read_csv_auto('{A}')")
    c.execute(f"""create temp table sub as select subreddit, count(*) comments, count(distinct author) users,
        avg(word_count) words, sum(len(regexp_extract_all(lower(body),'{FIRST}')))/greatest(sum(word_count),1) first_person,
        avg(regexp_matches(lower(body),'{TYPE_TALK}')::int) type_talk
        from read_parquet('{C}') where lang='en' group by 1 having count(distinct author)>={args.min_users}""")
    c.execute(f"""create temp table lab as select distinct s.subreddit, s.author, p.mbti, p.big5 from read_parquet('{C}') s join prof p using(author)
        where s.lang='en' and s.subreddit in (select subreddit from sub)""")
    base=dict(c.execute("select mbti, count(*)::double/sum(count(*)) over () from prof where length(mbti)=4 and mbti not like '%x%' group by 1").fetchall())
    lift={}
    for s,t,share in c.execute("select subreddit, mbti, count(*)::double/sum(count(*)) over (partition by subreddit) from lab where length(mbti)=4 and mbti not like '%x%' group by 1,2").fetchall():
        if base.get(t) and share/base[t]>lift.get(s,(0,''))[0]: lift[s]=(share/base[t],t)
    rows=c.execute("""select s.*, count(distinct l.author) filter (where l.mbti is not null) mbti_users, count(distinct l.author) filter (where l.big5) big5_users
        from sub s left join lab l using(subreddit) group by all order by mbti_users desc""").fetchall()
    cols=[d[0] for d in c.description]; rows=[dict(zip(cols,r)) for r in rows]
    import numpy as np, re
    q75=float(np.quantile([r['first_person'] for r in rows],.75))
    for r in rows:
        r['type_lift'],r['lift_type']=(round(lift.get(r['subreddit'],(0,''))[0],2),lift.get(r['subreddit'],(0,''))[1])
        named=bool(re.search(NAME,r['subreddit'].lower()))
        r['kind']='typology' if r['type_talk']>=.05 or named else 'self-disclosure' if r['first_person']>=q75 and r['type_talk']<.01 else 'other'
        for k in ('words','first_person','type_talk'): r[k]=round(float(r[k]),4)
    kinds={k:{'subreddits':sum(r['kind']==k for r in rows),'comments':sum(r['comments'] for r in rows if r['kind']==k)} for k in ('typology','self-disclosure','other')}
    args.output.write_text(json.dumps({'first_person_q75':q75,'kinds':kinds,'subreddits':rows},indent=1),encoding='utf-8')
    print(json.dumps({'n':len(rows),'first_person_q75':round(q75,4),'kinds':kinds}))

if __name__=='__main__': main()
