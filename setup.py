"""Setup.py now only remains as a build script for the cython extensions.
Using setup.py for other things is now deprecated and may have unexpected results: 
    setup.py test -> pytest
    setup.py develop -> pip install -e
"""
from setuptools import setup, Extension
import setuptools_scm  # noqa: F401
from Cython.Build import build_ext ## missing dev time req


setup(
    ext_modules = [
        Extension(
            "rsatoolbox.cengine.similarity",
            ["src/rsatoolbox/cengine/similarity.pyx"])],
    cmdclass={'build_ext': build_ext}
)
