# installation

(TODO, when the repo `numqi` is public available) The following command should be okay for `win/mac/linux`

```bash
pip install numqi
# TODO upload the package to pypi
```

Since `numqi` is still not public available right now, please download the source code and install it manually.

```bash
# install numqi first
git clone git@github.com:husisy/numqi.git
# TODO add numqi dependency
cd numqi
pip install .
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
# macOS m1/m2 user (Apple silicon M series)
# also see https://www.cvxgrp.org/scs/install/python.html
brew install openblas
OPENBLAS="$(brew --prefix openblas)" pip install scs
```

a complete fresh new conda environments [miniconda-documentation](https://docs.conda.io/en/latest/miniconda.html)

> conda can create isolated Python environment to install package. If you have any problems install `numqi` using the above `pip install` command, please try following conda commands

```bash
# for linux users without naivdia-GPU (this should also works for windows user)
# nocuda is the environment name (you can change it what you like)
conda create -y -n nocuda
conda install -y -n nocuda -c conda-forge pytorch ipython pytest matplotlib scipy tqdm cvxpy
conda activate nocuda
pip install numqi

# for linux users with nvidia-GPU (this should also works for windows user)
# cuda118 is the environment name (you can change it what you like)
conda create -y -n cuda118
conda install -y -n cuda118 -c conda-forge pytorch ipython pytest matplotlib scipy tqdm cvxpy
conda activate cuda118
pip install numqi

# for mac user
# metal is the environment name (you can change it what you like)
conda create -y -n metal
conda install -y -n metal -c conda-forge pytorch ipython pytest matplotlib scipy requests tqdm cvxpy
conda activate metal
brew install openblas
OPENBLAS="$(brew --prefix openblas)" pip install scs
pip install numqi
```

## for developer

Personally i use `conda/miniconda/mamba/micromamba` to create a virtual environment. You can use any familiar tools `poetry/rye/etc.` to create a virtual environment.

```bash
conda create -n env-numqi python
conda activate env-numqi
```

install `numqi`

```bash
git clone git@github.com:husisy/numqi.git
cd numqi
pip install -e ".[dev]"
# pip install -e .
```

run the unittest

```bash
# take about for 300 seconds on my laptop, "OMP_NUM_THREADS=1" usually runs faster (some unnecessay threads are disabled)
OMP_NUM_THREADS=1 pytest --cov=python/numqi

# if you have a multi-core CPU, you can run the unittest in parallel (take about 70 seconds on my laptop)
OMP_NUM_THREADS=1 pytest -n 8 --durations=10 --cov=python/numqi
```

build and Serve the documentation locally, then brower the website `127.0.0.1:8000`

```bash
mkdocs serve
```

1. **WARNING**: second indentaion must be 4 spaces, not 3 spaces (necessary for `mkdoc`)
2. api style: [griffe/usage](https://mkdocstrings.github.io/griffe/docstrings/)
3. toolchain
    * [github/mkdocstrings](https://github.com/mkdocstrings/mkdocstrings)
    * [github/mkdocstrings/python](https://github.com/mkdocstrings/python)
    * [github/mkdocstrings/griffe](https://github.com/mkdocstrings/griffe)
    * [github/best-of-mkdocs](https://github.com/mkdocs/best-of-mkdocs)
4. special module name, not for users
   * `._xxx.py`: internal functions, not for users
   * `._internal.py`: private to submodules. E.g., `numqi.A._internal` can only be imported in `numqi.A.xxx`
   * `._lib_public.py`: library public functions. E.g., `numqi.A._lib_public` can be imported by `numqi.B`

strip the jupyter notebook output before commit (only setup once), add following lines to `.git/config` or global `.gitconfig`

```text
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

**VSCODE-USER**: if you are using `vscode`, please click the button `clear all ouputs` manually before commit.
