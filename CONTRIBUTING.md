This is some guidance for RSAtoolbox contributors, but our approach evolves over time. Don't hesitate to contact @ilogue (Jasper) or @HeikoSchuett with any questions.


Your cycle
==========

1. If you identify something that has to be fixed or done: create an issue. Discussions
or questions about the theory or best practices should be filed as Github Discussions.
2. If you want to start coding or documenting something, the first step is to check if anyone else is working on this in the list of Pull Requests. If not, create a branch on your local machine, commit something small such as a note or empty file, and push the commit to the same branch on GitHub. Then you can then open a Pull Request. This is for two reasons: you're communicating to the team that you're working on this (so we're not doing things twice), and it gives you and the others an easy way to track your progress.
3. Commit regularly (typically every 10-30 minutes) and give your commits useful messages. "Changes to the data package" does not say anything about what you've done, "Added new feature model" does. If your commit is too large this makes it harder to write a short message.
4. Write unit-tests to cover your new code. This is easier when you recently wrote the code. Tip: try writing tests before you implement the code. The test should assert at least the most important outcomes of the functionality (typically the value returned).
5. Add Python type annotations where practical.
6. When you're done with the feature, ask for reviews from two team members or ask the maintainers for help.


How-to
======

1. `pip install -r requirements.txt` install rsatoolbox dependencies (and repeat when you make changes)
2. `pip install -r tests/requirements.txt` install test dependencies (and repeat when you make changes)
3. `pip uninstall rsatoolbox` uninstall rsatoolbox in your environment
4. `rm dist/*` remove any previously built packages if necessary
5. `python -m build` compile a new rsatoolbox package with your latest changes
6. `pip install dist/*` install the new package
7. `pytest` run the unit tests on the installed version of rsatoolbox
8. run linting tools such as `flake8`, `vulture` to discover any style issues


Rules
=====

1. Only through Pull Requests can you submit changes or additions to the code.
2. Every Pull Request has to be reviewed by two team members.
3. New code should have useful unit tests.
4. Code should pass the `pylint` style check.
5. Functions, classes, methods should have a `Google-style docstring`.
6. Larger new features should come with narrative documentation and an example.
7. When you're ready for your Pull Request to be reviewed, in the top right corner you can suggest two reviewers,
or alternatively, ping @ilogue or @HeikoSchuett and we will assign reviewers.
8. Consider how to handle NaNs in the user input. If your code can't handle them, you can throw an exception.


Deployment
==========


- when a PR is merged into the branch main, it is build as a pre-release (or "development") package and uploaded to pypi. The latest pre-release version can be installed using `pip install --pre rsatoolbox`
- when a release tag is added to the branch main, the package is instead marked as a released (or "stable") version.


Naming scheme
=============

**Classes**

- CamelCase
- ends in noun

*example: FancyModel*


**Functions and methods**

- lowercase with underscores
- starts with verb

*example: rdm.ranktransform(), transform_rank(rdm), calculate_gram_matrix, load_fmri_data*


**Variables**

- lowercase and underscore
- typically nouns or concepts

*example: contrast_matrix*
