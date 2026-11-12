DELETE FROM tool_runs WHERE session_id NOT IN (SELECT id FROM sessions);

CREATE TRIGGER sessions_ad_tool_runs AFTER DELETE ON sessions BEGIN
  DELETE FROM tool_runs WHERE session_id = old.id;
END;
