#!/usr/bin/env python3
"""Helper functions for our continuous integration suite on Github Actions 
"""
import os
fpath = 'ruff.log'
with open(fpath) as fhandle:
    print(fhandle.readlines()[0])


if "GITHUB_STEP_SUMMARY" in os.environ :
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fhandle :
        fhandle.write('hello from python')