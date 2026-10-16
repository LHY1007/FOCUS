# Software submission

| Requirement | Location or status |
| --- | --- |
| Source code | [Repository](https://github.com/LHY1007/FOCUS) |
| Small execution dataset | Ten registered H&E PNG and five-gene condition NPZ pairs in `demos/input/` |
| Dependencies and tested environments | `requirements.txt`, `setup.cfg`, `FOCUS.yaml`, README and `docs/environments/` |
| Installation | README installation section |
| Demo and outputs | `python demos/run.py`; ten five-panel PNGs and ten float32 five-gene NPZs; `demos/checkpoint_provenance.json` |
| Demo runtime | 788.90 s total for ten five-gene pairs, 1,000 DDIM steps each, one CB4 RTX 6000 Ada GPU |
| New data | [Input preparation and pretrained feature export](usage.md) |
| Reproduction | [Equations, training settings and verification scope](reproduction.md) |
| License | MIT, SPDX `MIT` |
| Version | Package `0.2.0`; this manuscript repair will be identified by its commit |
| Independent trial | [First-run form](https://github.com/LHY1007/FOCUS/issues/new?template=first-run.yml) |

The demo checkpoint was newly trained for 100 updates from random initialization. Its proxy masks, adult WMB reference and zero scGPT/GigaPath feature arrays are disclosed in [usage.md](usage.md). Its five-gene and sample prompt used the pinned frozen BioBERT v1.1 model. No paper figure accuracy is attributed to this checkpoint.

## Reference policies

- [Nature code and software submission checklist](https://www.nature.com/documents/nr-software-policy.pdf)
- [Nature guidelines for authors submitting code and software](https://www.nature.com/documents/GuidelinesCodePublication.pdf)
