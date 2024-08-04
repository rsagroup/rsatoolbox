#!/usr/bin/env python3
"""Helper for processing the Ruff log output
"""
from gha import count_log_issues, set_github_summary_api_envs

THRESHOLD = 0  # How many errors are allowed
CONTEXT = 'Ruff'

n = count_log_issues('ruff.log')
accept = n <= THRESHOLD

summary_issues = """
### Ruff: {} issues found :construction:

See individual errors below under *Annotations*, or 
see them inline at the *Files Changed* tab of the Pull Request.

You can also install ruff (`pip install ruff`) on your machine
and then run `ruff check` locally.
"""

summary_ok = """
### Ruff: No issues found :sparkles:
"""

summary = summary_ok if accept else summary_issues.format(n)
description = 'No issues found' if accept else f'Found {n} issues'
set_github_summary_api_envs(accept, description, summary, CONTEXT)
