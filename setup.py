#!/usr/bin/env python

from setuptools import setup, find_packages

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="dots",
    version="0.1",
    description="Dimensionality of the Tangent Space",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/LRudL/dots",
    author="LRudL",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    scripts=[] # hmm is this right?
)