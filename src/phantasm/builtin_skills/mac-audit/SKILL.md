---
name: mac-audit
description: Check this Mac's security settings without changing anything — FileVault, SIP, Gatekeeper, firewall, updates, screen lock, open ports, sharing, launch agents — and rate what to fix. Use when the user asks how secure their Mac is.
---
1. Run every check in one run_command, so the user approves once:

```
echo "== macOS"; sw_vers
echo "== SIP"; csrutil status
echo "== FileVault"; fdesetup status
echo "== Gatekeeper"; spctl --status
echo "== Firewall"; /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate --getstealthmode
echo "== Updates"; defaults read /Library/Preferences/com.apple.SoftwareUpdate 2>/dev/null | grep -E "Automatic|Critical|Config"
echo "== Screen lock"; sysadminctl -screenLock status 2>&1
echo "== Listening ports"; lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk 'NR==1 || !seen[$1 FS $9]++'
echo "== Remote access"; launchctl print-disabled system 2>/dev/null | grep -E "com.openssh.sshd|screensharing"
echo "== Launch agents"; ls ~/Library/LaunchAgents /Library/LaunchAgents /Library/LaunchDaemons 2>/dev/null
```

   A check that needs admin rights may fail: say "needs admin" and go on. Never use sudo and never change a setting.
2. Report a table: check · state · risk (ok, low, medium, high) · what to do (the exact System Settings path or command for the user to run). Rate FileVault off, SIP disabled or Gatekeeper off as high; the firewall off, automatic security updates off, a screen lock later than five minutes, and Remote Login or Screen Sharing on without a reason as medium; stealth mode off as low. For each service listening on all interfaces (`*:port`), name the process and whether it is expected.
3. Launch agents and daemons: name the unfamiliar ones and what each probably is (read its plist with read_file to see the program it runs). Flag anything that runs from /tmp, Downloads or a hidden folder as worth a look, but do not call software malware without evidence.
4. End with the three fixes that matter most.
