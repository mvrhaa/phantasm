---
name: cve
description: Find known vulnerabilities (CVEs) of a program and version and say whether the user is affected and what to update. Use when the user asks if software is vulnerable, about a CVE, or what to patch.
argument-hint: <program> [version]
---
1. Pin down the product and its exact version. Without a version from the user, find the installed one with one read-only command, for example `brew list --versions <name>`, `<name> --version`, `pip show <name>`, `npm ls -g <name>`, or for an app `defaults read /Applications/<App>.app/Contents/Info CFBundleShortVersionString`.
2. Ask NVD with web_fetch:
   - by keyword: `https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=<product>&resultsPerPage=50` (spaces as `%20`);
   - one CVE: `https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-YYYY-NNNNN`.
   Without an API key NVD allows about five requests in 30 seconds, so make few of them.
3. From each result read `cve.id`, the English description, the severity (`metrics.cvssMetricV31[0].cvssData.baseScore` and `baseSeverity`; V30 or V2 when V31 is missing), `published`, and the affected ranges in `configurations[].nodes[].cpeMatch[]` (`criteria`, `versionStartIncluding`, `versionEndExcluding` and the like).
4. Decide for the user's version: affected, not affected, or unclear. Count only CVEs whose ranges include it; when NVD gives no range, say it is unclear.
5. Answer with the verdict first (affected or not, the worst severity), then a short table: CVE · severity (score) · what an attacker gets · fixed in. Then the update command that matches how it was installed (`brew upgrade <name>`, `pip install -U <name>`, the vendor's updater). Link each CVE as https://nvd.nist.gov/vuln/detail/<id>.

Never invent a CVE, a score or a version: everything comes from NVD's answer. If NVD fails or finds nothing, say so and point to the vendor's security advisories. Do not try exploits.
