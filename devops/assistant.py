#!/usr/bin/env python3
"""Helper for our continuous integration suite on Github Actions
"""

import os, sys

THRESHOLD = 0  # How many errors are allowed
CONTEXT = 'Ruff'
ENV_VAR_NAME = 'DEVOPS_ASST_API_ARGS'

assert len(sys.argv) == (1+3), 'Expected three input arguments'
run_id = sys.argv[1] # github.run_id
job_id = sys.argv[2] # github.job
sha = sys.argv[3]    # github.event.pull_request.head.sha

fpath = 'ruff.log'
with open(fpath) as fhandle:
    lines = fhandle.readlines()
lines.reverse()
for line in lines:
    if line.startswith('Found '):
        n = int(line.split()[1])
accepted = n <= THRESHOLD

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

summary = summary_ok if accepted else summary_issues.format(n)

target_url = f'https://github.com/rsagroup/rsatoolbox/actions/runs/{ run_id }/attempts/1#summary-{ job_id }'
state = 'success' if accepted else 'failure' # error, failure, pending, or success
description = 'no issues found' if accepted else f'found {n} issues'
api_url = f'/repos/rsagroup/rsatoolbox/statuses/{ sha }'
cmd = (f'--method POST {api_url} -f "state={state}" ' +
       f'-f "target_url={target_url}" -f "description={description}" -f "context={CONTEXT}"')

if 'GITHUB_STEP_SUMMARY' in os.environ:
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fhandle:
        fhandle.write(summary)

if 'GITHUB_ENV' in os.environ:
    with open(os.environ['GITHUB_ENV'], 'a') as fhandle:
        fhandle.write(f'{ENV_VAR_NAME}={cmd}')
