CREATE TABLE file_changes (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  turn_id TEXT,
  path TEXT NOT NULL,
  before BLOB,
  after_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX idx_file_changes_session ON file_changes(session_id, created_at);
