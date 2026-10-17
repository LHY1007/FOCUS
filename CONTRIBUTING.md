# Contributing to FOCUS

Report bugs and propose changes through [GitHub issues](https://github.com/LHY1007/FOCUS/issues)
and pull requests. Include the commit, operating system, Python/PyTorch/CUDA versions,
command, input shapes and full error message. Share a small public or simulated
example; do not upload private patient data or credentials.

Create a branch from `main`, make the smallest change needed, and update the
relevant documentation. Use four-space Python indentation and preserve model and
checkpoint interfaces unless the change explicitly documents incompatibility.
The current model generates five gene channels per group; RGB H&E and encoder
feature dimensions have separate meanings.

Install development tools with `python -m pip install -e '.[dev]'`. For code or
packaging changes, run the checks relevant to the affected path:

```bash
python -m compileall -q guided_diffusion transform demos
python -m build
python -m twine check dist/*
```

For inference changes, run `python demos/run.py` on a CUDA GPU. The bundled
example contains one registered five-gene input pair. Check that it produces
`demos/output/3_10.npz` with finite `(5,256,256)` float32 predictions and a
five-panel PNG rendered from that NPZ using the documented `[0,1]` display
range. Use an explicitly shortened `--steps N` run only to check execution;
validation of the default sampler requires the full 1,000-step run. Keep
`demos/output/` out of source packages. CPU CI checks package construction,
installation and imports.

Optional local whitespace checks can be enabled with `sh install-hooks.sh`;
this changes only the current checkout's Git hook path. The package is not
published to PyPI and automatic publishing is not configured. Contributions
are distributed under the MIT License.
