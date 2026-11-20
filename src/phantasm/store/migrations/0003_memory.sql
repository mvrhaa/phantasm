CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  tags TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  uses INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
  text, tags,
  content='memories', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, text, tags) VALUES (new.rowid, new.text, new.tags);
END;

CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, text, tags)
  VALUES ('delete', old.rowid, old.text, old.tags);
END;

CREATE TRIGGER memories_au AFTER UPDATE OF text, tags ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, text, tags)
  VALUES ('delete', old.rowid, old.text, old.tags);
  INSERT INTO memories_fts (rowid, text, tags) VALUES (new.rowid, new.text, new.tags);
END;
