from setuptools import setup, find_packages

setup(
    name='numpyqi',
    version='0.0.0',
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
