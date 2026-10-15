# Contributing to FOCUS

Report bugs and propose changes through [GitHub issues](https://github.com/LHY1007/FOCUS/issues)
and pull requests. Include the commit, operating system, Python/PyTorch/CUDA versions, command,
input shapes and full error message. Share a small public or simulated example; do not upload
private patient data or credentials.

Create a branch from `main`, make the smallest change needed, and update the relevant instructions.
Use four-space Python indentation and preserve the model and checkpoint interfaces unless the
change explicitly documents incompatibility. The current generation target is one gene channel;
RGB H&E and encoder-feature channels have separate meanings.

Install development tools with `python -m pip install -e '.[dev]'`. Before opening a pull request:

```bash
git diff --check
python -m compileall -q guided_diffusion transform demos
python -m build
python -m twine check dist/*
```

For inference changes, run `python demos/run.py` on a CUDA GPU and check that all ten input pairs
produce ten PNGs and ten NPZs, with finite `(256, 256)` float32 `prediction` arrays. The PNG must be
the grayscale rendering of the paired NPZ using the documented `[-1, 1]` display range. Keep
`demos/output/` empty in submitted source packages; do not commit weights or generated results.
CPU CI checks packaging and source integrity, not GPU inference or biological accuracy.

Optional local whitespace checks can be enabled with `sh install-hooks.sh`; this only changes
the current checkout's Git hook path. This repository has not been published to PyPI, and no
automatic publishing is configured. Contributions are distributed under the MIT License.
