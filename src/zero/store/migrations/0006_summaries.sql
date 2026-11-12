CREATE TABLE session_summaries (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  until_message_id TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
