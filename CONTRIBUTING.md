Your cycle
==========

1. If you identify something that has to be fixed or done: create an issue. Issues can be associated with projects.
2. If you want to start coding or documenting something, the first step is to check if anyone else is working on this in the list of Pull Requests. If not, create a branch on your local machine, commit something small such as a note or empty file, and push the commit to the same branch on GitHub. Then you can then open a Pull Request. This is for two reasons: you're communicating to the team that you're working on this (so we're not doing things twice), and it gives you and the others an easy way to track your progress.
3. Commit regularly (typically every 10-30 minutes) and give your commits useful messages. "Changes to the data package" does not say anything about what you've done, "Added new feature model" does. If your commit is too large this makes it harder to write a short message.
4. When you're done with the feature, ask for reviews from two team members or ask the maintainers for help.
5. When the reviewers have approved the Pull Request, they will merge it into the master branch. At this point you want to checkout the master branch again and pull so that you have your latest changes, and can open a new branch for a new feature. 


Rules
=====

1. Only through Pull Requests can you submit changes or additions to the code.
2. Every Pull Request has to be reviewed by two team members.
3. New code should be covered 100% by unit tests.
4. Code should pass the `pylint` style check.
5. Functions, classes, methods should have a `Google-style docstring`.
6. Larger new features should come with narrative documentation and an example.
7. When you're ready for your Pull Request to be reviewed, in the top right corner you can suggest two reviewers,
or alternatively, ping @ilogue or @HeikoSchuett and we will assign reviewers.


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