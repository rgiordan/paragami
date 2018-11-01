from os import path
from setuptools import setup, find_packages
import sys

here = path.abspath(path.dirname(__file__))

# Parse requirements.txt, ignoring any commented-out or git lines.
with open(path.join(here, 'requirements.txt')) as requirements_file:
    requirements_lines = requirements_file.read().splitlines()

requirements = [line for line in requirements_lines
                if not (line.startswith('#') or line.startswith('git'))]

git_requirements = [line for line in requirements_lines
                     if line.startswith('git')]

for git_req in git_requirements:
    loc = git_requirements[0].find('egg=') + 4
    requirements += [ git_requirements[0][loc:] ]

print(requirements)
