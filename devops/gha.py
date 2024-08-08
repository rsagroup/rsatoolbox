#!/usr/bin/env python3
"""Helper functions for Github Actions
"""
from typing import Tuple
import sys
import os
ENV_VAR_NAME = 'DEVOPS_ASST_API_ARGS'


def parse_github_args() -> Tuple[str, str, str]:
    assert len(sys.argv) == (1+3), 'Expected three input arguments'
    run_id = sys.argv[1] # github.run_id
    job_id = sys.argv[2] # github.job
    sha = sys.argv[3]    # github.event.pull_request.head.sha
    return run_id, job_id, sha


def count_log_issues(fpath: str) -> int:
    with open(fpath) as fhandle:
        lines = fhandle.readlines()
    if len(lines) < 2:
        return 0
    lines.reverse()
    for line in lines:
        if line.startswith('Found '):
            return int(line.split()[1])
        elif 'pyright' in fpath:
            return int(line.split()[0])
    else:
        raise ValueError('[devops.gha] Could not find summary line')


def set_github_summary_api_envs(accept: bool, description: str, summary: str, context: str) -> None:
    run_id, job_id, sha = parse_github_args()
    target_url = (f'https://github.com/rsagroup/rsatoolbox/actions/runs/{ run_id }' 
                  f'/attempts/1#summary-{ job_id }')
    state = 'success' if accept else 'failure' # error, failure, pending, or success
    api_url = f'/repos/rsagroup/rsatoolbox/statuses/{ sha }'
    cmd = (f'--method POST {api_url} -f "state={state}" ' +
           f'-f "target_url={target_url}" -f "description={description}" -f "context={context}"')

    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fhandle:
            fhandle.write(summary)

    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as fhandle:
            fhandle.write(f'{ENV_VAR_NAME}={cmd}')
