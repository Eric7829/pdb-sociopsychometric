from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import duckdb
from .models import (Profile, ProfileRef, Snapshot, Segment, Document, Distribution,
                     CreatorMatch, Comment, SelfVote)

def snapshot_from_dict(data):
    data = dict(data)
    data['profile'] = Profile(**data['profile'])
    data['segments'] = [Segment(**x) for x in data['segments']]
    data['documents'] = [Document(**x) for x in data['documents']]
    distributions = []
    for x in data['distributions']:
        x = dict(x)
        x['evidence'] = [SelfVote(**v) for v in x.get('evidence', [])]
        distributions.append(Distribution(**x))
    data['distributions'] = distributions
    match = dict(data['creator_match'])
    comments = []
    for x in match['comments']:
        x = dict(x)
        x['self_votes'] = [SelfVote(**v) for v in x['self_votes']]
        comments.append(Comment(**x))
    match['comments'] = comments
    data['creator_match'] = CreatorMatch(**match)
    return Snapshot(**data)

class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))
        self.connection.execute(files('pdb_ingest').joinpath('schema.sql').read_text())

    def __enter__(self): return self
    def __exit__(self, *args): self.connection.close()

    def save_snapshot(self, snapshot):
        c = self.connection
        p = snapshot.profile
        body = json.dumps(asdict(snapshot), ensure_ascii=False)
        c.execute('BEGIN')
        try:
            c.execute('INSERT INTO ingestion_attempts VALUES (?,?,?,?)', [p.profile_num,snapshot.captured_at,snapshot.complete,body])
            existing = c.execute('SELECT complete FROM profiles WHERE profile_num=?',[p.profile_num]).fetchone()
            if not snapshot.complete and existing and existing[0]:
                c.execute('COMMIT')
                return
            for table in ('text_segments','vote_distributions','documents','profiles','text_spans','reply_context'):
                c.execute(f'DELETE FROM {table} WHERE profile_num=?',[p.profile_num])
            self_votes = [asdict(v) for d in snapshot.distributions for v in d.evidence]
            c.execute('INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                      [p.profile_num,p.profile_slug,p.creator_id,p.creator_username,p.header_summary,p.bio_quote,
                       json.dumps(self_votes),p.total_votes_all,snapshot.ramble_word_ratio,snapshot.captured_at[:10],
                       snapshot.captured_at,snapshot.complete,body])
            for s in snapshot.segments:
                c.execute('INSERT INTO text_segments(profile_num,source_type,source_ref,modality,original_text,raw_text,scrubbed_text,word_count,ramble_word_count) VALUES (?,?,?,?,?,?,?,?,?)',
                          [p.profile_num,s.source_type,s.source_ref,s.modality,s.original_text,s.raw_text,s.scrubbed_text,s.word_count,s.ramble_word_count])
            for d in snapshot.distributions:
                c.execute('INSERT INTO vote_distributions VALUES (?,?,?,?,?,?,?,?,?,?)',
                          [p.profile_num,d.system_name,d.total_system_votes,json.dumps(d.observed_counts),json.dumps(d.adjusted_counts),json.dumps(d.simplex),d.shannon_entropy,d.self_vote_status,d.training_eligible,json.dumps([asdict(v) for v in d.evidence])])
            for d in snapshot.documents:
                c.execute('INSERT INTO documents VALUES (?,?,?,?,?,?)',[p.profile_num,d.key,json.dumps(d.source_comment_ids),d.status,d.text,d.error])
            c.execute('UPDATE crawl_queue SET complete=? WHERE profile_num=?',[snapshot.complete,p.profile_num])
            c.execute('COMMIT')
        except BaseException:
            c.execute('ROLLBACK')
            raise

    def load_snapshot(self, profile_num):
        row = self.connection.execute('SELECT snapshot_json FROM profiles WHERE profile_num=?',[profile_num]).fetchone()
        return snapshot_from_dict(json.loads(row[0])) if row else None

    def enqueue(self, refs, sample='top'):
        rank=self.connection.execute('SELECT coalesce(max(discovery_rank),0) FROM crawl_queue').fetchone()[0]
        for ref in refs:
            rank+=1
            existing=self.connection.execute('SELECT complete FROM profiles WHERE profile_num=?',[ref.profile_num]).fetchone()
            self.connection.execute('INSERT INTO crawl_queue(profile_num,url,complete,discovery_rank,sample) VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING',[ref.profile_num,ref.url,bool(existing and existing[0]),rank,sample])

    def pending(self):
        return [ProfileRef(*r) for r in self.connection.execute('SELECT profile_num,url FROM crawl_queue WHERE NOT complete AND NOT unavailable ORDER BY discovery_rank NULLS LAST,profile_num').fetchall()]

    def mark_unavailable(self, profile_num):
        self.connection.execute('UPDATE crawl_queue SET unavailable=TRUE WHERE profile_num=?',[profile_num])

    def checkpoint(self, stream, token, complete):
        self.connection.execute('INSERT INTO crawl_checkpoints VALUES (?,?,?) ON CONFLICT(stream) DO UPDATE SET token=excluded.token, complete=excluded.complete',[stream,token,complete])

    def load_checkpoint(self, stream):
        return self.connection.execute('SELECT token,complete FROM crawl_checkpoints WHERE stream=?',[stream]).fetchone() or (None,False)

    def save_agent_run(self, run, usage):
        c = self.connection
        c.execute('BEGIN')
        try:
            c.execute('INSERT INTO agent_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                      [run['run_id'],run['profile_num'],run['model'],run['prompt_version'],run['text_view'],run['temperature'],
                       run['seed'],run['started_at'],run.get('finished_at'),run['complete'],run['corpus_sha256'],
                       run['corpus_words'],usage['cost'],json.dumps(usage),run['error']])
            for x in run['calls']:
                c.execute('INSERT INTO agent_calls VALUES (?,?,?,?,?,?,?,?)',
                          [run['run_id'],x['call_name'],x['attempt'],x['user_prompt'],json.dumps(x['arguments']),
                           x['reasoning'],json.dumps(x['response']),x['error']])
            if run['complete']:
                e = run['evaluation']
                c.execute('INSERT INTO psychometric_evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                          [run['run_id'],run['profile_num'],e['agent_self_mbti'],e['agent_self_enneagram'],
                           e['self_typing_rationale'],e['form_m_mbti']]+[json.dumps(e[k]) for k in
                          ('form_m_continuous','form_m','ipip_domains','ipip_facets','ipip_responses','form_m_responses','response_style')])
            c.execute('COMMIT')
        except BaseException:
            c.execute('ROLLBACK')
            raise

    def load_reflections(self, profile_num, corpus_sha256, model, prompt_version):
        row = self.connection.execute('SELECT reflections_json FROM agent_reflections WHERE profile_num=? AND corpus_sha256=? AND model=? AND prompt_version=?',
                                      [profile_num,corpus_sha256,model,prompt_version]).fetchone()
        return json.loads(row[0]) if row else None

    def save_reflections(self, profile_num, corpus_sha256, model, prompt_version, reflections, result, cost):
        from datetime import datetime, timezone
        self.connection.execute('INSERT INTO agent_reflections VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING',
                                [profile_num,corpus_sha256,model,prompt_version,json.dumps(reflections),result.get('reasoning'),
                                 json.dumps({k:v for k,v in result.items() if k not in ('arguments','reasoning')}),cost,
                                 datetime.now(timezone.utc).isoformat()])

    def completed_runs(self, profile_num, model, prompt_version, text_view):
        return self.connection.execute('SELECT count(*) FROM agent_runs WHERE complete AND profile_num=? AND model=? AND prompt_version=? AND text_view=?',
                                       [profile_num,model,prompt_version,text_view]).fetchone()[0]

    def export_jsonl(self, path):
        path.parent.mkdir(parents=True,exist_ok=True)
        rows = self.connection.execute('SELECT snapshot_json FROM profiles WHERE complete ORDER BY profile_num').fetchall()
        with path.open('w',encoding='utf-8') as out:
            for row in rows:
                data=json.loads(row[0])
                data['raw_corpus']='\n\n'.join(s['raw_text'] for s in data['segments'])
                data['clean_corpus']='\n\n'.join(s['scrubbed_text'] for s in data['segments'])
                out.write(json.dumps(data,ensure_ascii=False)+'\n')
        return len(rows)
