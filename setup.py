from os import path
from setuptools import setup, find_packages
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
    name='car-price-prediction',
    description="Python package for predicting car price",
    long_description=readme,
    author="Thomas Reid",
    author_email='thomasred013@hotmail.com',
    url='https://gitlab.com/Uk-amor/RMT/rmii-inputs',
    entry_points={
        'console_scripts': [
            'predictPrice = predictCarPrice:predict'
        ],
    },
    install_requires=requirements
)
