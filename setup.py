#!/usr/bin/env python3
from setuptools import setup

setup(
    use_scm_version={"write_to": "genomatnn/_version.py"},
    entry_points={"console_scripts": ["genomatnn=genomatnn.cli:main"]},
    dependency_links=[
        "git+https://github.com/grahamgower/stdpopsim.git@selection#egg=stdpopsim-99",
    ],
)
