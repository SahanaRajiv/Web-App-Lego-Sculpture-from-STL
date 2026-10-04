---
name: GitHub push authentication
description: Distinguishing Git transport credentials from the GitHub API integration.
---

Treat Git CLI authentication and the GitHub API integration as separate when diagnosing a failed push. Do not reauthorize a healthy API connection solely because Git transport rejects its saved credential.

**Why:** Git push reported an invalid username or token while the connected GitHub API returned success and confirmed write permission.

**How to apply:** Verify API access through the existing connection. If it is healthy but Git push authentication fails, guide the user to Replit account settings → Git Providers to disconnect and reconnect GitHub. Official documentation distinguishes this from Connected Services: reconnecting there alone does not repair Git pane authentication. Never request or expose a token.