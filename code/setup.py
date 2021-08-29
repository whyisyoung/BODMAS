#!/usr/bin/env python

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="UTF-8") as f:
    readme = f.read()

version = "0.0.1"
requires = open("requirements.txt", "r").read().strip().split()
package_data = {}
setup(
    name="bodmas",
    version=version,
    description="Blue Hexagon PE Malware BEnchmark for Research",
    long_description=readme,
    packages=["bodmas"],
    package_data=package_data,
    install_requires=requires,
    author_email="liminy2@illinois.edu")
