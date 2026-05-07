# Security policy

## Reporting a vulnerability

If you believe you've found a security issue in Augur, please report it privately rather than opening a public issue or PR.

**How to report:** open a [private security advisory](https://github.com/willgitdata/augur/security/advisories/new) on this repository. GitHub will notify the maintainers and keep the report confidential until it's resolved.

Please include:

- A description of the issue and its impact
- Steps to reproduce, or a minimal proof-of-concept
- Any relevant version, configuration, or environment details
- (Optional) a suggested fix

## What to expect

- Acknowledgement within 7 days
- A working timeline within 30 days, including a target fix date or a clear "won't fix" with reasoning
- Credit in the release notes when the fix ships, unless you'd rather stay anonymous

## Scope

In scope:

- Input validation in `@augur/core` and `@augur/server` (e.g. crafted search payloads that crash or exfiltrate)
- Authentication / authorization gaps in `@augur/server` (the optional API-key path)
- Trace store leaks (e.g. cross-tenant trace bleed if someone embeds Augur multi-tenant)
- Dependency vulnerabilities surfaced by `pnpm audit`

Out of scope:

- Vulnerabilities in user-supplied adapter implementations (your adapter, your security model)
- Issues that require physical access to the host running Augur
- Brute-forcing the in-memory adapter on a non-production deployment

## Supported versions

Until 1.0, only the latest minor version receives security updates. After 1.0 we'll publish an explicit support window.
