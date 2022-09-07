from setuptools import setup, Extension
from Cython.Build import build_ext

test_requires = []
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
