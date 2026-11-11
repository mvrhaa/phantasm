-- A todo list per session: the model plans its work and ticks items off.
-- Items are numbered 1, 2, 3 … within their session; the model refers to them
-- by that number.

CREATE TABLE todos (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  number INTEGER NOT NULL,
  text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending | in_progress | done
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (session_id, number)
);
