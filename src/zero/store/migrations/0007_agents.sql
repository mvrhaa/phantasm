CREATE TABLE agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  parent_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'working',
  report TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_agents_parent ON agents(parent_session_id, created_at);
CREATE UNIQUE INDEX idx_agents_session ON agents(session_id);
