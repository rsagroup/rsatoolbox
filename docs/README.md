To build and check the documentation:


1. `pip install docs/requirements.txt`
2. `cd docs/`
3. `make html`

When adding, removing or renaming a module, create / adapt a stub for autodoc.
To build the docs, your system must also have pandoc installed, (`choco install pandoc` or `apt install pandoc` or `brew install pandoc`)
Have a look at this RST cheatsheet: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html