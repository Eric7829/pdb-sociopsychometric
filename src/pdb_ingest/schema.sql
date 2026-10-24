CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
INSERT INTO schema_version VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS profiles (
 profile_num BIGINT PRIMARY KEY, profile_slug TEXT NOT NULL, creator_id TEXT, creator_username TEXT,
 header_summary TEXT, bio_quote TEXT, self_vote_raw TEXT, total_votes_all INTEGER,
 ramble_word_ratio DOUBLE, last_updated_date DATE, captured_at TEXT, complete BOOLEAN,
 snapshot_json JSON NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS segment_ids;
CREATE TABLE IF NOT EXISTS text_segments (
 segment_num BIGINT DEFAULT nextval('segment_ids') PRIMARY KEY, profile_num BIGINT NOT NULL,
 source_type TEXT CHECK(source_type IN ('PINNED_COMMENT','AUTHOR_COMMENT','GOOGLE_DOC')),
 source_ref TEXT, modality TEXT CHECK(modality IN ('RAMBLE','QUESTIONNAIRE','MIXED')),
 original_text TEXT NOT NULL, raw_text TEXT NOT NULL, scrubbed_text TEXT NOT NULL,
 word_count INTEGER, ramble_word_count INTEGER
);
CREATE TABLE IF NOT EXISTS vote_distributions (
 profile_num BIGINT NOT NULL, system_name TEXT, total_system_votes INTEGER,
 observed_counts_json JSON, vote_counts_json JSON, simplex_json JSON, shannon_entropy DOUBLE,
 self_vote_status TEXT, training_eligible BOOLEAN, evidence_json JSON,
 PRIMARY KEY(profile_num,system_name)
);
CREATE TABLE IF NOT EXISTS documents (
 profile_num BIGINT, document_key TEXT, source_comment_ids JSON, status TEXT, text TEXT, error TEXT,
 PRIMARY KEY(profile_num,document_key)
);
CREATE TABLE IF NOT EXISTS ingestion_attempts (
 profile_num BIGINT, captured_at TEXT, complete BOOLEAN, snapshot_json JSON
);
CREATE TABLE IF NOT EXISTS crawl_queue (
 profile_num BIGINT PRIMARY KEY, url TEXT NOT NULL, complete BOOLEAN DEFAULT FALSE, discovery_rank BIGINT
);
ALTER TABLE crawl_queue ADD COLUMN IF NOT EXISTS discovery_rank BIGINT;
ALTER TABLE crawl_queue ADD COLUMN IF NOT EXISTS unavailable BOOLEAN DEFAULT FALSE;
UPDATE crawl_queue SET unavailable=FALSE WHERE unavailable IS NULL;
ALTER TABLE crawl_queue ADD COLUMN IF NOT EXISTS sample TEXT DEFAULT 'top';
UPDATE crawl_queue SET sample='top' WHERE sample IS NULL;
CREATE TABLE IF NOT EXISTS crawl_checkpoints (
 stream TEXT PRIMARY KEY, token TEXT, complete BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
 run_id TEXT PRIMARY KEY, profile_num BIGINT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
 text_view TEXT NOT NULL, temperature DOUBLE, seed BIGINT, started_at TEXT, finished_at TEXT,
 complete BOOLEAN NOT NULL, corpus_sha256 TEXT, corpus_words INTEGER, cost_usd DOUBLE, usage_json JSON, error TEXT
);
CREATE TABLE IF NOT EXISTS agent_calls (
 run_id TEXT, call_name TEXT, attempt INTEGER, user_prompt TEXT, arguments_json JSON, reasoning TEXT,
 response_json JSON, error TEXT, PRIMARY KEY(run_id,call_name,attempt)
);
CREATE TABLE IF NOT EXISTS psychometric_evaluations (
 run_id TEXT PRIMARY KEY, profile_num BIGINT NOT NULL, agent_self_mbti VARCHAR(4), agent_self_enneagram TEXT,
 self_typing_rationale TEXT, form_m_mbti VARCHAR(4), form_m_continuous JSON, form_m_json JSON,
 ipip_domains_json JSON, ipip_facets_json JSON, ipip_responses_json JSON, form_m_responses_json JSON,
 response_style_json JSON
);
CREATE TABLE IF NOT EXISTS agent_reflections (
 profile_num BIGINT, corpus_sha256 TEXT, model TEXT, prompt_version TEXT, reflections_json JSON, reasoning TEXT,
 response_json JSON, cost_usd DOUBLE, created_at TEXT, PRIMARY KEY(profile_num,corpus_sha256,model,prompt_version)
);
CREATE TABLE IF NOT EXISTS text_spans (
 profile_num BIGINT, source_type TEXT, source_ref TEXT, ordinal INTEGER, kind TEXT, start_pos INTEGER, end_pos INTEGER,
 PRIMARY KEY(profile_num,source_type,source_ref,ordinal)
);
CREATE TABLE IF NOT EXISTS reply_context (
 profile_num BIGINT, comment_id TEXT PRIMARY KEY, context_comment_id TEXT, context_user_id TEXT, context_text TEXT
);
