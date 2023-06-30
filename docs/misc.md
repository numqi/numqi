# misc

## TODO

1. [ ] check `functools.lru_cache`, all type of input should be simple-typed, e.g. `int(1)` not `np.array([1])[0]`
2. [ ] make it a conda-forge package [link](https://conda-forge.org/docs/maintainer/adding_pkgs.html#the-staging-process)
3. [ ] github CI
4. [ ] coverage
5. documentation
   * [ ] numerical range

## license

```text
MIT License

Copyright (c) 2023 husisy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## contributing

Personally i use `conda/miniconda/mamba/micromamba` to create a virtual environment. You can use any familiar tools `poetry/rye/etc.` to create a virtual environment.

```bash
conda create -n env-numqi python
conda activate env-numqi
```

Then install `numqi` in it using `pip`

```bash
git clone git@github.com:husisy/numqi.git
cd numqi
pip install -e ".[dev]"
```

run the unittest

```bash
pytest --cov=python/numqi
```

build the documentation

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

## Q-and-A

> how this package name `numqi` comes?

Initially, this package is named `numpyqi`, later it's shortened to be `numqi` (pronounce: num-py-q-i). The reason is that `numpy` is the most popular python package for numerical computation, and `numqi` is based on `numpy`. The name `numqi` is also a play on the word `numpy` and quantum information.

1. short, no more than 7 letters
2. keyword: quantum information, numpy, python, optimization
3. NEP41: should not include the whole word "numpy"
4. example (good and bad)
    * `numpyqi`: bad
    * `numqi`: pronounced as "num py q i", emphasize that it's based on numpy and focuses on quantum information field
    * `numqy`: bad, confused with `numpy`

> why `233` appears so frequently?

`233` is a prime number! Internet slang that essentially means “LOL.”
