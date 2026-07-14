#!/usr/bin/env python3
"""Helper for processing the Pyright log output
"""
from gha import count_log_issues, set_github_summary_api_envs

THRESHOLD = 0  # How many errors are allowed
CONTEXT = 'Pyright'

old_n = count_log_issues('pyright_main.log')
new_n = count_log_issues('pyright_pr.log')
diff_n = new_n - old_n
accept = diff_n <= THRESHOLD

summary_issues = """
### Pyright: {} new issues found :construction:

Try to annotate your new code with type annotations. 
You can check for typing issues with the python plugin for your IDE,
or on the terminal by installing pyright (`pip install pyright`) 
on your machine and then run `pyright` locally.
"""

summary_ok = """
### Pyright: No new issues found :fireworks:

{} fewer issues compared to the main branch!
"""

summary = summary_ok.format(diff_n) if accept else summary_issues.format(diff_n)
description = f'Found {abs(diff_n)} fewer issues compared to main'
if diff_n > 0:
    description = f'Found {diff_n} more issues compared to main'
set_github_summary_api_envs(accept, description, summary, CONTEXT)

