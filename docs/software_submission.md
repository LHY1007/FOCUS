# Software-submission evidence

This page records the repository evidence for the Nature code/software submission items.
It is not a signed declaration and does not replace the journal's editable form.

| Item | Evidence / status |
| --- | --- |
| Source code | This versioned public repository and its source archive |
| Small dataset | Ten paired real H&E PNG and single-gene ST NPZ inputs in `demos/input/` |
| Dependencies, OS and tested versions | Root `requirements.txt`, `FOCUS.yaml`, and README's tested-system section |
| Non-standard hardware | Current implementation requires an NVIDIA CUDA GPU; tested on RTX 6000 Ada 48 GiB |
| Installation instructions | README and pip packaging; the original full conda environment was not clean-installed in the CB4 deployment |
| Normal-desktop installation time | Not measured; must remain unclaimed |
| Demo instructions and expected output | `python demos/run.py`; ten PNGs and ten float32 NPZ predictions |
| Normal-desktop demo time | Not measured; CB4 timing is reported separately |
| Use on your own data | Root README, `docs/usage.md` and preprocessing instructions |
| All manuscript quantitative results | Not reproduced by this execution demo; optional full reproduction remains unverified |
| License | MIT License; SPDX `MIT`; no numbered license version |
| Public source | https://github.com/LHY1007/FOCUS |
| DOI / archival identifier | No DOI has been minted in this task; use an immutable release/commit and arrange a DOI archive before claiming one |
| Independent colleague trial | Not performed in this task |

Suggested license text for the form:

> FOCUS source code is distributed under the MIT License (SPDX: MIT), with no additional
> restrictions on use of the source code. The copyright and license notice must be retained.
> Public datasets, pretrained weights and third-party components remain subject to their own
> licenses and access conditions.

Repository URL for the form: `https://github.com/LHY1007/FOCUS`.
The corresponding-author name and manuscript pseudocode location must be confirmed by the authors.
Do not tick items whose timing or manuscript evidence has not been established.

## Reference policies and examples

- [Nature code and software submission checklist](https://www.nature.com/documents/nr-software-policy.pdf)
- [Nature guidelines for authors submitting code and software](https://www.nature.com/documents/GuidelinesCodePublication.pdf)

The checklist lists MGC and LOL as examples of well-structured software packages:

- [MGC](https://github.com/neurodata/mgcpy-old): the original example now redirects to this archived repository.
- [LOL](https://github.com/neurodata/lol): supervised dimensionality reduction.

The reference layout includes legacy CI and publishing files. FOCUS keeps credential-free templates
for those files, and uses GitHub Actions for package integrity. No PyPI upload, Netlify deployment,
Travis activation or Docker runtime verification is implied merely by including configuration files.
