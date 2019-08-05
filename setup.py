from os import path
import re
from setuptools import setup, find_packages
import sys
import versioneer

# This used to be necessary before I removed sckit-sparse from the base
# dependencies.
#import numpy

# try:
#     import numpy
# except ImportError:
#     error = """
# paragami requires ``numpy`` to be installed before installation
# due to https://github.com/scikit-sparse/scikit-sparse/issues/55.
# """
#     sys.exit(error)

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
if sys.version_info < (3, 5):
    error = """
paragami does not support Python {0}.{2}.
Python 3.5 and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(3, 5)
    sys.exit(error)

import pip
pip_version_match = re.search(r'^[0-9]*', pip.__version__)
if pip_version_match:
    if int(pip_version_match.group(0)) < 19:
        sys.exit('Install requires pip version 19 or greater.  ' +
                 'Run pip install --upgrade pip.')
else:
    sys.exit('There was an error getting the pip version number.')


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

# Parse requirements.txt, ignoring any commented-out or git lines.
with open(path.join(here, 'requirements.txt')) as requirements_file:
    requirements_lines = requirements_file.read().splitlines()

requirements = [line for line in requirements_lines if not line.startswith('#')]

setup(
    name='paragami',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python pacakge to flatten and fold parameter data structures.",
    long_description=readme,
    author="Ryan Giordano",
    author_email='rgiordan@gmail.com',
    url='https://github.com/rgiordan/paragami',
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'paragami': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            ]
        },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
