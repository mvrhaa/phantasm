-- Agents: copies of Zero that work on one task in a session of their own and
-- report back to the session that started them. When the parent session is
-- deleted its agents go too; when an agent's own session is deleted (agent_stop)
-- the record stays, so its status and report can still be read.

CREATE TABLE agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  parent_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'working',  -- working | asking | waiting_user | done | failed | stopped
  report TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_agents_parent ON agents(parent_session_id, created_at);
CREATE UNIQUE INDEX idx_agents_session ON agents(session_id);
