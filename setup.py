"""Setup.py now only remains as a build script for the cython extensions.
Using setup.py for other things is now deprecated:
    setup.py test -> pytest
    setup.py develop -> pip install -e
"""
from setuptools import setup, Extension
import setuptools_scm  # noqa # pylint: disable=unused-import
from Cython.Build import build_ext


setup(
    ext_modules = [
        Extension(
            "rsatoolbox.cengine.similarity",
            ["src/rsatoolbox/cengine/similarity.pyx"])],
    cmdclass={'build_ext': build_ext}
)
