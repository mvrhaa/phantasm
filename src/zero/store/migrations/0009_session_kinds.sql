-- What a session is for, so clients can keep the user's chats apart from the
-- sessions Zero runs on its own: chat | agent | scheduled.

ALTER TABLE sessions ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat';
UPDATE sessions SET kind = 'scheduled' WHERE id LIKE 'scheduled-%';
UPDATE sessions SET kind = 'agent'
  WHERE id IN (SELECT session_id FROM agents WHERE session_id IS NOT NULL);
