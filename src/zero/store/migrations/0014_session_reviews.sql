CREATE TABLE session_reviews (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  until_message_id TEXT NOT NULL,
  reviewed_at TEXT NOT NULL
);
