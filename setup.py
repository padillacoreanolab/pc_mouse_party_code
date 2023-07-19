#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Padilla-Coreano Lab",
    author_email='padillacoreanolab@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This package contains all the functions that do not require high performance computing that the lab uses repeatedly.",
    entry_points={
        'console_scripts': [
            'pc_mouse_party=pc_mouse_party.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pc_mouse_party',
    name='pc_mouse_party',
    packages=find_packages(include=['pc_mouse_party', 'pc_mouse_party.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/PadillaCoreanoLabGeneral/pc_mouse_party',
    version='0.0.1a',
    zip_safe=False,
)
