import argparse
import asyncio
import json
from pathlib import Path
import sys
import httpx
from .annotate import annotate
from .battery import PROMPT_VERSION,PROMPTS,assess
from .browser import browser_session
from .llm import DEFAULT_MODEL,DEFAULT_PROVIDER,OpenRouter,load_api_key
from .capture import capture_session
from .client import PdbClient
from .pipeline import DocumentSource,process_profile,run_crawl,replay
from .store import Store
from .transport import JsonTransport

def main(argv=None):
    parser=argparse.ArgumentParser(description='Capture, crawl and replay PDB community profile JSON.')
    subs=parser.add_subparsers(dest='command',required=True)
    cap=subs.add_parser('capture')
    cap.add_argument('--url',required=True)
    cap.add_argument('--seconds',type=int,default=120)
    for name in ('crawl','profile'):
        sub=subs.add_parser(name)
        if name=='profile': sub.add_argument('profile_num',type=int)
        else:
            sub.add_argument('--limit',type=int,help='Maximum profile attempts this invocation')
            sub.add_argument('--max-profiles',type=int,default=500,help='Scope cap in PDB listing order (default: 500)')
            sub.add_argument('--sort',choices=('hot','top'),default='hot',help='Listing order: site default HOT, or Top (All Time)')
            sub.add_argument('--sample',default='top',help='Queue tag and scope name for a supplemental sample (default: top)')
            sub.add_argument('--type-filter',help="Regex on the listing consensus type, e.g. '^ES[TF][JP]\\b'")
            sub.add_argument('--min-words',type=int,help='Count only complete profiles with at least this many corpus words toward --max-profiles')
        sub.add_argument('--capture-dir',type=Path,default=Path('data/run'))
        sub.add_argument('--max-comments',type=int,help='Fetch reply threads only for the first N top-level comments (pinned first); default: all')
        sub.add_argument('--headless',action='store_true')
    pan=subs.add_parser('panel',help='Local web panel: dataset charts + self-description assessor (127.0.0.1 only)')
    pan.add_argument('--port',type=int,default=8765)
    pan.add_argument('--samples',nargs='+',default=['top'])
    aud=subs.add_parser('audit',help='Phase 5 audits (bundling, modality, inventory) to JSON')
    aud.add_argument('--output',type=Path,default=Path('data/audits/report.json'))
    aud.add_argument('--samples',nargs='+',default=['top'],help="crawl_queue sample tags to include (default: top)")
    aud.add_argument('--permutations',type=int,default=1000)
    dis=subs.add_parser('distill',help='Stage 2 baseline: embed with ModernBERT-base and cross-validate (needs the ml extra)')
    dis.add_argument('--view',choices=('main','unmasked','raw'),default='main')
    dis.add_argument('--samples',nargs='+',default=['top'])
    dis.add_argument('--out',type=Path,default=Path('data/stage2'))
    ann=subs.add_parser('annotate',help='Backfill span annotations and reply context from captures (offline)')
    ann.add_argument('--capture-dir',type=Path,required=True)
    rep=subs.add_parser('replay'); rep.add_argument('directory',type=Path)
    rep.add_argument('--max-comments',type=int,help='Must match the cap used when capturing')
    exp=subs.add_parser('export'); exp.add_argument('--output',type=Path,required=True)
    ass=subs.add_parser('assess',help='Stage 3 agent battery (IPIP-NEO-120, MBTI Form M, self-typing) via OpenRouter')
    ass.add_argument('profile_nums',type=int,nargs='*',help='Default: every complete profile')
    ass.add_argument('--runs',type=int,default=1,help='Complete runs wanted per profile (repeat runs measure agent reliability)')
    ass.add_argument('--limit',type=int,help='Maximum runs this invocation')
    ass.add_argument('--model',default=DEFAULT_MODEL)
    ass.add_argument('--provider',default=DEFAULT_PROVIDER,help=f'OpenRouter upstream pinned without fallbacks (default {DEFAULT_PROVIDER}); "any" lets OpenRouter route')
    ass.add_argument('--key-file',type=Path,default=Path('OPEN_ROUTER_API_KEY.txt'))
    ass.add_argument('--text-view',choices=('raw','scrubbed'),default='raw')
    ass.add_argument('--prompt-version',choices=sorted(PROMPTS),default=PROMPT_VERSION)
    ass.add_argument('--profiles-file',type=Path,help='File of profile numbers (one per line) to assess')
    ass.add_argument('--temperature',type=float,default=.7)
    ass.add_argument('--seed',type=int,default=0)
    ass.add_argument('--min-words',type=int,default=200)
    ass.add_argument('--concurrency',type=int,default=4,help='Profiles assessed in parallel')
    for name,sub in subs.choices.items():
        if name in ('capture','crawl','profile'): sub.add_argument('--session',type=Path,default=Path('data/browser'))
        if name=='capture': sub.add_argument('--output',type=Path,default=Path('data/captures'))
        else: sub.add_argument('--db',type=Path,default=Path('data/pdb.duckdb'))
    args=parser.parse_args(argv)
    if getattr(args,'limit',None) is not None and args.limit<=0: parser.error('--limit must be positive')
    if getattr(args,'max_profiles',1)<=0: parser.error('--max-profiles must be positive')
    if getattr(args,'seconds',1)<=0: parser.error('--seconds must be positive')
    if getattr(args,'profile_num',1)<=0: parser.error('profile_num must be positive')
    if (getattr(args,'max_comments',None) or 1)<=0: parser.error('--max-comments must be positive')
    for name in ('runs','concurrency'):
        if getattr(args,name,1)<=0: parser.error(f'--{name} must be positive')
    if not 0<=getattr(args,'temperature',0)<=2: parser.error('--temperature must be between 0 and 2')
    try:
        if args.command=='capture':
            rows=asyncio.run(capture_session(args.url,args.output,args.session,seconds=args.seconds))
            print(json.dumps({'captures':len(rows)})); return 0 if rows else 1
        if args.command=='panel':
            import duckdb
            from . import embed,panel
            store=Store.__new__(Store); store.connection=duckdb.connect(str(args.db),read_only=True)
            samples=tuple(args.samples); summary=panel.build_summary(store.connection,samples)
            encoder=embed.hf_encoder()
            assessor=panel.Assessor(store,encoder,samples=samples,cache_dir=Path('data/stage2/emb')/embed.model_slug(encoder[2]['model'])/'main')
            panel.serve(summary,assessor,port=args.port); return 0
        with Store(args.db) as store:
            if args.command=='export':
                print(json.dumps({'exported':store.export_jsonl(args.output)})); return 0
            if args.command=='audit':
                from .audits import run_audits
                result=run_audits(store.connection,permutations=args.permutations,samples=tuple(args.samples))
                args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=1),encoding='utf-8')
                print(json.dumps({'output':str(args.output),'profiles':result['profiles']})); return 0
            if args.command=='distill':
                from .distill import run_distill
                report=run_distill(store,args.view,out_root=args.out,samples=tuple(args.samples))
                print(json.dumps({'view':args.view,'embedded':report['embedded'],'metrics':report['metrics']})); return 0
            if args.command=='annotate':
                stats=annotate(store,args.capture_dir)
                print(json.dumps(stats)); return 1 if stats['raw_text_mismatches'] else 0
            if args.command=='assess':
                nums=args.profile_nums+([int(x) for x in args.profiles_file.read_text().split()] if args.profiles_file else [])
                async def run_assess():
                    async with httpx.AsyncClient() as http:
                        llm=OpenRouter(http,load_api_key(args.key_file),args.model,provider=None if args.provider=='any' else args.provider)
                        return await assess(store,llm,nums,runs=args.runs,view=args.text_view,temperature=args.temperature,seed=args.seed,
                                            min_words=args.min_words,concurrency=args.concurrency,limit=args.limit,prompt_version=args.prompt_version)
                stats=asyncio.run(run_assess())
                print(json.dumps(stats)); return 1 if stats['failed'] or stats['stopped'] else 0
            if args.command=='replay':
                stats=replay(args.directory,store,args.max_comments)
            else:
                async def run():
                    async with browser_session(args.session,headless=args.headless) as request:
                        transport=JsonTransport(request,args.capture_dir)
                        client=PdbClient(transport,sort='top' if getattr(args,'sort','hot')=='top' else None)
                        async with httpx.AsyncClient() as http:
                            docs=DocumentSource(http,args.capture_dir)
                            if args.command=='profile':
                                ok=await process_profile(client,store,docs,args.profile_num,args.max_comments)
                                return {'processed':1,'succeeded':int(ok),'failed':int(not ok)}
                            return await run_crawl(client,store,docs,args.limit,max_profiles=args.max_profiles,max_comments=args.max_comments,
                                                   sample=args.sample,type_filter=args.type_filter,min_words=args.min_words)
                stats=asyncio.run(run())
            print(json.dumps(stats))
            return 1 if stats['failed'] or (not stats['processed'] and not stats.get('scope_complete')) else 0
    except KeyboardInterrupt:
        print(json.dumps({'error':'Interrupted; resume with the same database/session.'}),file=sys.stderr); return 1
    except Exception as error:
        print(json.dumps({'error':type(error).__name__,'message':str(error)}),file=sys.stderr); return 1

if __name__=='__main__': raise SystemExit(main())
