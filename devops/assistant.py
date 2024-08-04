#!/usr/bin/env python3
"""Helper functions for our continuous integration suite on Github Actions"""

import os

THRESHOLD = 0  # How many errors are allowed

fpath = 'ruff.log'
with open(fpath) as fhandle:
    lines = fhandle.readlines()
lines.reverse()
for line in lines:
    if line.startswith('Found '):
        n = int(line.split()[1])


summary_issues = """
### Ruff: {} issues found :construction:

See individual errors below under *Annotations*, or 
see them inline at the *Files Changed* tab of the Pull Request.

You can also run install ruff and then run `ruff check` on your machine.
"""

summary_ok = """
### Ruff: No issues found :yellow_heart:
"""

summary = summary_issues.format(n) if n > THRESHOLD else summary_ok

if 'GITHUB_STEP_SUMMARY' in os.environ:
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fhandle:
        fhandle.write(summary)
