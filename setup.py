from setuptools import setup, Extension
import setuptools_scm  # noqa: F401
from Cython.Build import build_ext ## missing dev time req
from os.path import isfile

test_requires = []
if isfile('tests/requirements.txt'):
    with open('tests/requirements.txt') as reqfile:
        test_requires = reqfile.read().splitlines()

setup(
    tests_require=test_requires,
    test_suite='tests',
    url = "https://github.com/rsagroup/rsatoolbox",
    zip_safe = False,
    ext_modules = [
        Extension(
            "rsatoolbox.cengine.similarity",
            ["src/rsatoolbox/cengine/similarity.pyx"])],
    cmdclass={'build_ext': build_ext}
)
