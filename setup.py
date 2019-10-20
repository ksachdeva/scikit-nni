#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'scikit-learn', 'nni', 'pymongo', 'absl-py', 'pyyaml',]

setup_requirements = []

test_requirements = []

setup(
    author="Kapil Sachdeva",
    author_email='not@anemail.com',
    python_requires='>=3.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Hyper parameters search for scikit-learn components using Microsoft NNI",
    entry_points={
        'console_scripts': [
            'sknni=sknni.cli:cli',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sknni,scikit-nni',
    name='scikit-nni',
    packages=find_packages(include=['sknni', 'sknni.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ksachdeva/scikit-nni',
    version='0.1.1',
    zip_safe=False,
)
