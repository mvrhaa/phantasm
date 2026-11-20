CREATE TABLE session_memories (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (session_id, text)
);

CREATE INDEX idx_session_memories_session ON session_memories(session_id, created_at);
