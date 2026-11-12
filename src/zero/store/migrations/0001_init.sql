CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  turn_id TEXT,
  role TEXT NOT NULL,
  content_json TEXT NOT NULL,
  tool_calls_json TEXT,
  tool_call_id TEXT,
  is_error INTEGER NOT NULL DEFAULT 0,
  status TEXT,
  source TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);

CREATE TABLE tool_runs (
  call_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  label TEXT,
  input_json TEXT,
  status TEXT,
  summary TEXT,
  output_preview TEXT,
  duration_ms INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE approval_rules (
  tool TEXT PRIMARY KEY,
  decision TEXT NOT NULL,
  created_at TEXT NOT NULL
);
