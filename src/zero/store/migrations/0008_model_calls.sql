-- Every request to a model: who answered, for what, and how many tokens it took.
-- Free plans limit requests and tokens per day, so failed calls are kept too.
-- No foreign key to sessions: the record of use outlives deleted sessions.

CREATE TABLE model_calls (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT NOT NULL,             -- turn | agent | title | summary
  session_id TEXT,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  ok INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_model_calls_time ON model_calls(provider, created_at);
