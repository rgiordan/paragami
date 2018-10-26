# from setuptools import setup
#
# setup(name='paragami',
#       version='0.1.0',
#       description='Helper functions for sensitivity analysis',
#       long_description=long_description,
#       url='https://github.com/rgiordan/paragami',
#       author='Ryan Giordano',
#       author_email='rgiordan@gmail.com',
#       license='Apache 2.0',
#       packages=['paragami'],
#
#       python_requires='>=3',
#       classifiers = [
#         'License :: OSI Approved :: Apache Software License',
#         'Intended Audience :: Science/Research',
#         'Intended Audience :: Developers',
#         'Development Status :: 2 - Pre-Alpha',
#         'Natural Language :: English',
#         'Programming Language :: Python :: 3',
#         'Topic :: Scientific/Engineering :: Mathematics'
#       ],
#
#       install_requires = [
#         'autograd',
#         'numpy',
#         'scipy',
#         'json_tricks'
#       ]
# )

from os import path
from setuptools import setup, find_packages
import sys
import versioneer


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

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


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
    license='Apache 2.0',
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
