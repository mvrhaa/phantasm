-- Tool runs get their own id. Call ids come from the model and may repeat
-- (local models and the fake provider reuse them), so they cannot be the key.
-- SQLite cannot change a primary key in place, so the table is rebuilt.

CREATE TABLE tool_runs_new (
  id TEXT PRIMARY KEY,
  call_id TEXT NOT NULL,
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

INSERT INTO tool_runs_new (
  id, call_id, session_id, turn_id, tool, label, input_json,
  status, summary, output_preview, duration_ms, created_at
)
SELECT
  lower(hex(randomblob(16))), call_id, session_id, turn_id, tool, label, input_json,
  status, summary, output_preview, duration_ms, created_at
FROM tool_runs;

DROP TABLE tool_runs;
ALTER TABLE tool_runs_new RENAME TO tool_runs;
CREATE INDEX idx_tool_runs_turn ON tool_runs(session_id, turn_id);
