# misc

## TODO

1. [ ] check `functools.lru_cache`, all type of input should be simple-typed, e.g. `int(1)` not `np.array([1])[0]`
2. [ ] make it a conda-forge package [link](https://conda-forge.org/docs/maintainer/adding_pkgs.html#the-staging-process)

## contributing

to write documentations, see

1. api style: [griffe/usage](https://mkdocstrings.github.io/griffe/docstrings/)
2. toolchain
   * [github/mkdocstrings](https://github.com/mkdocstrings/mkdocstrings)
   * [github/mkdocstrings/python](https://github.com/mkdocstrings/python)
   * [github/mkdocstrings/griffe](https://github.com/mkdocstrings/griffe)

```bash
pip install 'mkdocstrings[crystal,python]'
```
