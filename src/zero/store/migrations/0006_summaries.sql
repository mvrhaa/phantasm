-- A running summary of the start of a long session, used when the history no
-- longer fits the model's context. until_message_id is the last message it covers.

CREATE TABLE session_summaries (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  until_message_id TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
