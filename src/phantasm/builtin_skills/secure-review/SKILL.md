---
name: secure-review
description: Review code for security bugs — injection, broken access control, secrets in the code, unsafe file and shell use, weak crypto, vulnerable dependencies — and show the fixes. Use when the user asks for a security review or audit of code.
argument-hint: [folder or files]
---
1. Map the code first: its entry points (HTTP handlers, command-line arguments, message handlers, files and network input), where that data goes, and where the trust boundaries are. Use list_dir, search_files and read_file; do not read everything.
2. Follow input to the places it is used and check:
   - injection: SQL built from strings, shell commands with user input (`shell=True`, `os.system`, backticks), template injection, file names that allow `../`, unsafe deserialization (`pickle`, `yaml.load`), `eval`;
   - access: routes without checks, object ids taken from the client without an ownership check, tokens without expiry, passwords stored without a slow hash;
   - secrets: search for keys and tokens (`AKIA[0-9A-Z]{16}`, `-----BEGIN .*PRIVATE KEY`, `api[_-]?key`, `secret`, `password\s*=`) and committed `.env` files;
   - crypto: MD5 or SHA-1 for passwords, ECB mode, fixed keys or IVs, `random` where `secrets` or `os.urandom` belongs;
   - web: `Access-Control-Allow-Origin: *` with credentials, state-changing forms without CSRF protection, cookies without HttpOnly, Secure or SameSite, server-side requests to user-given URLs, output that is not escaped;
   - dependencies: old or vulnerable versions (load the cve skill for the suspicious ones).
3. Report the findings by severity (critical, high, medium, low): file:line · what is wrong · how it could be abused, in one sentence · the fix, with the corrected code. Say so when a finding is a guess.
4. Do not run exploits against anything; at most suggest a local test the user can run.
