import os
import json
from setuptools import setup, find_packages

with open(os.path.join('python','numpyqi','_package.json'), 'r', encoding='utf-8') as fid:
    _package = json.load(fid)
__version__ = _package['version']


setup(
    name='numpyqi',
    version=__version__,
    packages=find_packages('python'),
    package_dir={'':'python'},
    description='quantum information toolbox implemented in numpy',
    install_requires=[
        'numpy',
        'opt_einsum',
        'scipy',
        'tqdm',
    ], #torch
)
