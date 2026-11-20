---
name: trust-check
description: Check whether a downloaded file, an app or a website can be trusted — checksum, code signature, notarization, quarantine origin, TLS certificate. Use when the user asks if a download, an app or a site is legit or safe.
argument-hint: <file, app or host>
---
For a file or an app, run the checks in one run_command:

```
shasum -a 256 "<path>"
codesign -dv --verbose=4 "<path>" 2>&1 | grep -E "Authority|TeamIdentifier|Identifier|Timestamp|Signature"
spctl --assess --verbose=4 --type execute "<path>" 2>&1
xattr -l "<path>"
mdls -name kMDItemWhereFroms "<path>"
```

(For a .pkg use `pkgutil --check-signature "<path>"`; for a .dmg `spctl -a -t open --context context:primary-signature -v "<path>"`.)
- When the user has the publisher's checksum, compare it exactly and say whether it matches.
- Say who signed it (the Developer ID name and Team ID), whether Apple notarized it (spctl says `source=Notarized Developer ID`), whether it came from the internet (`com.apple.quarantine`) and from which address.

For a website:

```
echo | openssl s_client -connect <host>:443 -servername <host> 2>/dev/null | openssl x509 -noout -subject -issuer -dates -ext subjectAltName
```

- Check that the name matches, the dates are current and the issuer is a known authority; point out lookalike domains (swapped or extra letters, other alphabets, extra words).

Give the verdict first (trusted, careful, do not run) with its reasons. Never open or run the file to test it.
