-- Schedules: tasks Zero runs on its own, once or again and again.
-- rule_json is the parsed rule (see zero.scheduler.rules); rule_text is how it
-- was written. next_run_at is NULL once a one-off schedule has run.

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
