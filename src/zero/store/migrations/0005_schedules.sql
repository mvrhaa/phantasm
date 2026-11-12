CREATE TABLE schedules (
  id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  rule_json TEXT NOT NULL,
  rule_text TEXT NOT NULL,
  timezone TEXT NOT NULL,
  next_run_at TEXT,
  last_run_at TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_schedules_due ON schedules(enabled, next_run_at);
