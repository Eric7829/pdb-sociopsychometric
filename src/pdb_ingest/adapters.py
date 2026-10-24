"""Mappings verified against PDB JSON responses on 2026-10-23."""
import re
from .models import Profile, ProfileRef, Page, Comment, SelfVote, SchemaError
from .text import AP_TYPES

SYSTEMS={'1':'MBTI','2':'ENNEAGRAM','3':'SOCIONICS','4':'ALIGNMENT','5':'INSTINCT','6':'TRITYPE',
         '7':'TEMPERAMENT','8':'AP','9':'BIG5_SLOAN','11':'JUNGIAN','12':'BIG5_E',
         '13':'BIG5_N','14':'BIG5_C','15':'BIG5_A','16':'BIG5_O'}

def parse_profile(payload):
    if hasattr(payload,'payload'): payload=payload.payload
    try:
        name=payload['mbti_profile']
        slug=re.sub(r'[^\w]+','-',name.lower()).strip('-') or str(payload['id'])
        return Profile(int(payload['id']),slug,str(payload['user_id']) if payload.get('user_id') not in (None,0,-1) else None,
                       payload.get('contributor'),payload.get('personality_type'),payload.get('wiki_description'),payload.get('total_vote_counts'))
    except (KeyError,TypeError,ValueError) as e:
        raise SchemaError('Unrecognized profile response') from e

def parse_counts(payload):
    try:
        counts={}
        for system,rows in payload['breakdown_systems'].items():
            key=SYSTEMS.get(str(system),f'PDB_{system}')
            counts[key]={}
            for row in rows:
                label=row['personality_type']; n=row['theCount']
                if not isinstance(label,str) or type(n) is not int or n<0: raise SchemaError('Invalid tally')
                if label in counts[key]: raise SchemaError('Duplicate tally class')
                counts[key][label]=n
        for system in payload.get('systems',[]):
            key=SYSTEMS.get(str(system['id']),f"PDB_{system['id']}")
            total=system.get('system_vote_count')
            if total and key not in counts: raise SchemaError(f'Missing breakdown for {key}')
            if key in counts and total is not None and sum(counts[key].values()) != total:
                raise SchemaError(f'Incomplete breakdown for {key}')
        return counts
    except (KeyError,TypeError,ValueError) as e:
        if isinstance(e,SchemaError): raise
        raise SchemaError('Unrecognized vote breakdown') from e

def badge_system(label):
    if re.fullmatch(r'[IE][NS][TF][JP]',label): return 'MBTI'
    if re.fullmatch(r'[1-9]w[1-9]',label): return 'ENNEAGRAM'
    if label in AP_TYPES: return 'AP'
    if re.fullmatch(r'[1-9]{3}',label): return 'TRITYPE'
    if re.fullmatch(r'(?:S[OPX][1-9]|(?:so|sp|sx)/(?:so|sp|sx)|E[1-9])',label): return 'INSTINCT'
    if re.fullmatch(r'(?:[IE][FTNS](?:\([FTNS]\))?)',label): return 'JUNGIAN'
    if label.startswith(('Melancholic','Phlegmatic','Choleric','Sanguine')): return 'TEMPERAMENT'
    for trait,key in [('Extraversion','E'),('Neuroticism','N'),('Conscientiousness','C'),('Agreeableness','A'),('Openness','O')]:
        if re.fullmatch(trait+r' (?:0|25|50|75|100)%',label): return 'BIG5_'+key
    if label in ('EII','LII','IEI','ILI','ILE','LIE','EIE','IEE','SEE','SLE','LSI','ESI','SEI','SLI','ESE','LSE'): return 'SOCIONICS'
    return 'UNMAPPED_BADGE'

def parse_comment(row):
    try:
        evidence=[]
        badge=row.get('vote_types') or row.get('theVote') or ''
        for label in badge.split(' - '):
            label=label.strip()
            if label and label!='XXXX':
                system=badge_system(label)
                evidence.append(SelfVote(system,label,str(row['id'])+':vote_types',system!='UNMAPPED_BADGE'))
        return Comment(str(row['id']),str(row['user_id']) if row.get('user_id') else None,row.get('username'),
                       row['comment'] if not row.get('is_hidden') and row.get('is_active',True) else '',
                       bool(row.get('is_pinned')),str(row['parent_id']) if row.get('parent_id') else None,
                       evidence,row.get('user_mbti'),row.get('reply_count',0),
                       str(row['reply_to_user_id']) if row.get('reply_to_user_id') else None)
    except (KeyError,TypeError) as e:
        raise SchemaError('Unrecognized comment response') from e

def parse_listing(payload,offset,limit):
    try:
        rows=payload['profiles']
        if not isinstance(rows,list): raise TypeError()
        refs=[ProfileRef(int(r['id']),f"https://www.personality-database.com/profile/{int(r['id'])}") for r in rows]
        terminal=len(rows)<limit
        return Page(refs,None if terminal else str(offset+limit),terminal)
    except (KeyError,TypeError,ValueError) as e:
        raise SchemaError('Unrecognized profile listing') from e
