from setuptools import setup, find_packages

requires = []
with open('requirements.txt') as reqfile:
    requires = reqfile.read().splitlines()


with open('README.md', encoding='utf-8') as readmefile:
    long_description = readmefile.read()


setup(
    name='pyrsa',
    version='3.0.0',
    description='Python Representational Similarity Analysis (RSA)',
    url='https://github.com/rsagroup/pyrsa',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
      "Programming Language :: Python",
      "License :: OSI Approved :: MIT License",
      "Development Status :: 1 - Planning",
      "Topic :: Scientific/Engineering",
      "Intended Audience :: Science/Research",
    ],
    author='pyrsa authors',
    author_email='',
    keywords='neuroscience ',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requires,
    tests_require=requires+['parameterized',],
    test_suite="tests",
    entry_points = {
        'console_scripts': ['rsa3=pyrsa.io.commandline:main'],
    }
)
