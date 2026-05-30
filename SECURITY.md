# Security Policy

## Supported Versions

RuleKiln is currently in early alpha.

Security fixes will target the latest public version of the project.

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, report security issues privately using one of the following:

- GitHub private vulnerability reporting

## What to Include

Please include:

- Description of the issue
- Steps to reproduce
- Potential impact
- Affected version or commit
- Relevant logs or configuration, with secrets removed

## Secret Handling

Never include real API keys, provider tokens, database passwords, or private credentials in issues, PRs, logs, or examples.

RuleKiln users should store provider credentials in environment variables or secret-management systems.

## Scope

Security issues may include:

- Secret leakage
- Unsafe artifact exposure
- Authentication or authorization flaws
- Path traversal
- Remote code execution
- Unsafe file upload handling
- Prompt or case data exposure
- Provider credential misuse

## Out of Scope

The following are generally out of scope:

- Vulnerabilities in third-party model providers
- Issues requiring already-compromised credentials
- Social engineering
- Denial-of-service reports without a practical exploit path

## Disclosure

Please give maintainers reasonable time to investigate and fix confirmed vulnerabilities before public disclosure.