ALTER TABLE sessions ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat';
UPDATE sessions SET kind = 'scheduled' WHERE id LIKE 'scheduled-%';
UPDATE sessions SET kind = 'agent'
  WHERE id IN (SELECT session_id FROM agents WHERE session_id IS NOT NULL);
