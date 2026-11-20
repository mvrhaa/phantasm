---
name: ctf
description: Help with CTF challenges and security practice labs — first look, encodings, crypto, web, forensics, reversing — step by step, hints first. Use when the user is solving a CTF task or a lab.
argument-hint: <task text or file>
---
1. Ask for the category, the files or the address and the flag format when they are missing. Work only on the challenge's own files and the target the challenge gives; when the target is unclear, ask before touching any host.
2. Hints first: give the next step as a hint and ask whether to go on; give the whole solution when the user asks for it.
3. By category:
   - first look: `file`, `strings -n 6`, `xxd | head -40`, `exiftool` and `binwalk` when installed; try the usual encodings (base64, base32, hex, rot13 and other Caesar shifts, URL encoding, Morse) with python3 one-liners;
   - crypto: name the scheme first (classical ciphers, XOR with a short key, RSA with a small e or shared factors, weak randomness), then compute with python3;
   - web: read the page source, robots.txt, cookies and JWTs (decode them, check `alg`), and the parameters for SQL injection, template injection and file inclusion, only on the challenge's target;
   - forensics: packet captures with `tshark` when installed, images (least significant bits, metadata, data appended after the end), disk and memory images with the user's tools;
   - reversing: `otool -tV` or `objdump -d`, strings, and comparisons against constants; suggest Ghidra for bigger binaries.
4. Explain why each step works, so the user learns the technique, and keep a short list of what was tried.
5. Install nothing without asking; prefer the tools already on the Mac (python3, openssl, xxd, file, strings).
