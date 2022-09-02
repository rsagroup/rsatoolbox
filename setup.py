from setuptools import setup, find_packages, Extension
from Cython.Build import build_ext
import numpy

test_requires = []
with open('tests/requirements.txt') as reqfile:
    test_requires = reqfile.read().splitlines()

setup(
    tests_require=test_requires,
    test_suite='tests',
    url = "https://github.com/rsagroup/rsatoolbox",
    zip_safe = False,
    extensions = [Extension("*", ["*.pyx"])],
    cmdclass={'build_ext': build_ext}
)
