# Installation

Try without installing anything: `application/get-started` in colab

<a target="_blank" href="https://colab.research.google.com/github/numqi/numqi/blob/main/docs/application/get_started/quantum_state.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## pip installation

The following command should be okay for `win/mac/linux`

```bash
pip install numqi
```

test whether succuessfully installed (run it in `python/ipython` REPL)

```Python
import numqi
```

For academic user, personally recommend to use `mosek` [link](https://docs.mosek.com/latest/pythonapi/index.html) over the default convex solver `SCS`. `mosek` seems to much faster on this package. However, `mosek` is not free for commercial use.

```bash
pip install Mosek

# replace <xxx> with YOUR conda environment name
# apply mosek academic license https://www.mosek.com/products/academic-licenses/
conda install -n <xxx> -c MOSEK MOSEK
```

For macOS user, you might need to install `openblas` first and then install `scs` as below.

```bash
# macOS m1/m2 user (Apple silicon M series), see https://www.cvxgrp.org/scs/install/python.html
brew install openblas
OPENBLAS="$(brew --prefix openblas)" pip install scs
```

## Conda environment

conda can create isolated Python environment to install package. If you have any problems install `numqi` using the above `pip install` command, please try following conda commands [miniconda-documentation](https://docs.conda.io/en/latest/miniconda.html)

```bash
# for macOS user, metal is the environment name
conda create -y -n metal
conda install -y -n metal -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow scipy tqdm opt_einsum cvxpy scs pytest-xdist pytest-cov seaborn pytorch sympy galois mkdocs ipywidgets mkdocs-material mkdocs-jupyter pymdown-extensions mkdocstrings twine platformdirs
# conda install -y -n metal -c MOSEK MOSEK
conda activate metal
## scs-macos issue, see https://www.cvxgrp.org/scs/install/python.html
# brew install openblas
# OPENBLAS="$(brew --prefix openblas)" pip install scs
pip install numqi

# for linux users with nvidia-GPU, cuda128 is the environment name
conda create -y -n cuda129
conda install -y -n cuda129 -c conda-forge cuda-version=12.9 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow scipy tqdm opt_einsum cvxpy scs pytest-xdist pytest-cov "pytorch=*=*cuda129_generic*" sympy mkdocs ipywidgets mkdocs-material mkdocs-jupyter pymdown-extensions mkdocstrings twine platformdirs numba=0.61 numpy=2.2
# conda install -y -n metal -c MOSEK MOSEK
conda activate cuda129
pip install galois
pip install numqi
# OMP_NUM_THREADS=1 pytest -n 8 --durations=10 --cov=python/numqi

# windows-user
conda create -y -n cuda129
conda install -y -n cuda129 -c conda-forge cuda-version=12.9 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow scipy=1.16 tqdm opt_einsum cvxpy scs pytest-xdist pytest-cov pytorch sympy mkdocs ipywidgets mkdocs-material mkdocs-jupyter pymdown-extensions mkdocstrings twine platformdirs numba=0.61 numpy=2.2
# conda install -y -n metal -c MOSEK MOSEK
conda activate cuda129
pip install galois
pip install numqi
# $env:MKL_NUM_THREADS="1"
# $env:OMP_NUM_THREADS="1"
# pytest -n 8 --durations=10 --cov=python/numqi

# (deprecated) for win/linux users without naivdia-GPU, nocuda is the environment name
conda create -y -n nocuda
conda install -y -n nocuda -c conda-forge pytorch ipython pytest matplotlib scipy tqdm cvxpy
conda activate nocuda
pip install numqi
```

## Guide for contributors


Quick start for contributors: open this project in GitHub Codespaces and then `pip install -e ".[dev]"`

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/numqi/numqi)

### local development environment

It is recommended to install `numqi` in a virtual environment.  You may use `conda/miniconda/mamba/micromamba` to create a virtual environment.

```bash
conda create -n env-numqi python
conda activate env-numqi
```

Then you should install `numqi` as developer

```bash
git clone git@github.com:numqi/numqi.git
cd numqi
pip install -e ".[dev]"
```

### Unittest

You can now run the unittest

```bash
# "OMP_NUM_THREADS=1" usually runs faster (some unnecessay threads are disabled)
OMP_NUM_THREADS=1 pytest --durations=10 --cov=python/numqi

# if you have a multi-core CPU, you can run the unittest in parallel (take about 120 seconds on my laptop)
OMP_NUM_THREADS=1 pytest -n 8 --durations=10 --cov=python/numqi
```

### Documentation

You can now build the documentation locally.

```bash
mkdocs serve
```
then browse the website `127.0.0.1:8000`

1. **WARNING**: second indentaion must be 4 spaces, not 3 spaces (necessary for `mkdoc`)
2. api style: [griffe/usage](https://mkdocstrings.github.io/griffe/docstrings/)
3. toolchain
    * [github/mkdocstrings](https://github.com/mkdocstrings/mkdocstrings)
    * [github/mkdocstrings/python](https://github.com/mkdocstrings/python)
    * [github/mkdocstrings/griffe](https://github.com/mkdocstrings/griffe)
    * [github/best-of-mkdocs](https://github.com/mkdocs/best-of-mkdocs)
4. special module name, not for users
   * `._xxx.py`: internal functions, not for users
   * `._internal.py`: private to submodules. E.g., `numqi.A._internal` should only be imported in `numqi.A.xxx`
   * `._public.py`: library public functions. E.g., `numqi.A._public` can be imported by `numqi.B`
5. strip the jupyter notebook output before commit
