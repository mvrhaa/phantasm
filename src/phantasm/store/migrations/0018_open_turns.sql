CREATE TABLE open_turns (
  turn_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  agent_id TEXT,
  started_at TEXT NOT NULL
);
