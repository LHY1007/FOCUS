# Software submission

The table lists the files and measurements for the Nature software submission checklist.

| Requirement | Location or status |
| --- | --- |
| Source code | [Repository](https://github.com/LHY1007/FOCUS) and versioned release archives |
| Small dataset | Ten H&E PNG and single-gene ST NPZ pairs in `demos/input/` |
| Dependencies and operating systems | `requirements.txt`, the extras in `setup.cfg`, `FOCUS.yaml` and the README |
| Tested versions and hardware | README: tested systems |
| Installation instructions | README: installation |
| Desktop installation time | 14 min 31 s on Windows 10, including downloads and one download retry; see README |
| Demo instructions and outputs | `python demos/run.py`; ten PNGs and ten float32 NPZs |
| Desktop demo time | Not measured; the CB4 run took 11.45 seconds |
| Use with other data | README: data preparation and usage; `docs/usage.md` |
| Reproduction instructions (optional) | [Current commands and paper settings](reproduction.md); full figure reproduction has not been validated |
| License | MIT License; SPDX `MIT`; no numbered version |
| Version identifier | [Release v0.2.0](https://github.com/LHY1007/FOCUS/releases/tag/v0.2.0); no DOI assigned |
| Independent trial | Awaiting a report through the [first-run form](https://github.com/LHY1007/FOCUS/issues/new?template=first-run.yml) |

Preprint version 3 describes the algorithm in Methods sections 1.1–1.6 and 3 (PDF pages 29–39).
The form's author names and manuscript locations should match the submitted version.

License text:

> FOCUS source code is distributed under the MIT License (SPDX: MIT). Redistribution must retain
> the copyright and license notice. Data, pretrained models and third-party components have
> their own licenses and access conditions.

## Reference policies and examples

- [Nature code and software submission checklist](https://www.nature.com/documents/nr-software-policy.pdf)
- [Nature guidelines for authors submitting code and software](https://www.nature.com/documents/GuidelinesCodePublication.pdf)

The checklist gives these examples of software packages:

- [MGC](https://github.com/neurodata/mgcpy-old), now archived.
- [LOL](https://github.com/neurodata/lol), supervised dimensionality reduction.
