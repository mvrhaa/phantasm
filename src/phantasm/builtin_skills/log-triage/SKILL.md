---
name: log-triage
description: Look through logs for signs of attacks or compromise — failed logins, sudo use, brute force, scanners, odd requests — and say what happened when. Use when the user asks to check logs or whether someone got in.
argument-hint: [log file or time range]
---
1. Pick the source: a file the user names (read_file; search_files for big ones), a web server's access and error logs, Linux auth logs (`/var/log/auth.log`, `/var/log/secure`), or this Mac's unified log through run_command, filtered and limited, for example:
   - `log show --last 24h --style compact --predicate 'process == "sudo"' | tail -200`
   - `log show --last 24h --style compact --predicate 'eventMessage CONTAINS[c] "authentication failure" OR eventMessage CONTAINS[c] "failed to authenticate"' | tail -200`
   - `log show --last 24h --style compact --predicate 'process == "sshd"' | tail -200`
   Filter and count instead of dumping whole logs.
2. Look for repeated failures from one source (brute force), a success right after many failures, logins at odd hours or from new places, sudo by unexpected users, new users or launch agents, scanners' user agents (sqlmap, nikto, nmap, masscan, zgrab), path traversal (`../`, `%2e%2e`), SQL injection tokens (`' OR 1=1`, `UNION SELECT`), shell metacharacters in parameters, spikes of 401, 403, 404 or 500, and requests for `/.env`, `/.git`, `/wp-admin` and other admin paths.
3. Report a verdict first (nothing notable, suspicious, likely compromise), then a timeline (time · source IP or user · what · how many), then the evidence lines, quoted briefly. Keep facts apart from guesses.
4. Suggest next steps (block an address, rotate a password, turn off a service, collect more logs) as commands or settings for the user to apply; do not apply them.

Mask secrets that show up in logs (tokens, cookies, passwords) instead of repeating them.
