"""
`clearml-serving` - Model-Serving Orchestration and Repository Solution
https://github.com/clearml/clearml-serving
"""

import os.path
# Always prefer setuptools over distutils
from setuptools import setup, find_packages


def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


here = os.path.dirname(__file__)
# Get the long description from the README file
long_description = read_text(os.path.join(here, 'README.md'))


def read_version_string(version_file):
    for line in read_text(version_file).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = read_version_string("clearml_serving/version.py")

requirements = read_text(os.path.join(here, 'requirements.txt')).splitlines()

setup(
    name='clearml-serving',
    version=version,
    description='clearml-serving - Model-Serving Orchestration and Repository Solution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # The project's main homepage.
    url='https://github.com/clearml/clearml-serving.git',
    author='ClearML',
    author_email='support@clear.ml',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Version Control',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords='clearml mlops devops trains development machine deep learning version control machine-learning '
             'machinelearning deeplearning deep-learning model-serving',
    packages=find_packages(exclude=['contrib', 'docs', 'data', 'examples', 'tests']),
    install_requires=requirements,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
          'clearml-serving = clearml_serving.__main__:main',
        ],
    },
)
