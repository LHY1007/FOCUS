# Software submission

| Requirement | Location or status |
| --- | --- |
| Source code | [Repository](https://github.com/LHY1007/FOCUS) |
| Small execution dataset | One registered observed H&E/ST PNG/NPZ pair with real multimodal conditions in `demos/input/` |
| Dependencies and tested environments | `requirements.txt`, `setup.cfg`, `FOCUS.yaml`, README and `docs/environments/` |
| Installation | README installation section |
| Demo and outputs | `python demos/run.py`; one five-panel PNG and one float32 five-gene NPZ; `demos/checkpoint_provenance.json` |
| Demo runtime | 151.90 s for one five-gene pair, 1,000 DDIM steps, one CB4 RTX 6000 Ada GPU |
| New data | [Input preparation and pretrained feature export](usage.md) |
| Reproduction | [Equations, training settings and verification scope](reproduction.md) |
| License | MIT, SPDX `MIT` |
| Version | Package `0.3.0`; identify this manuscript repair by its exact commit until a separate release is tagged |
| Independent trial | [First-run form](https://github.com/LHY1007/FOCUS/issues/new?template=first-run.yml) |

The bundled checkpoint was newly trained for 100 updates from random initialization on one actual 10x mouse-pup training tile, batch 1. Its conditions include official 10x cell and nuclear polygons, native nuclear morphology complexity, independent P0 muscle scRNA, and genuine official scGPT-spatial, Prov-GigaPath and frozen BioBERT features. The co-expression prior uses the training tile. The resulting preview is visibly speckled; no held-out accuracy or paper figure result is attributed to this checkpoint. See [real_data.md](real_data.md) for sources, permissions and limits.

## Reference policies

- [Nature code and software submission checklist](https://www.nature.com/documents/nr-software-policy.pdf)
- [Nature guidelines for authors submitting code and software](https://www.nature.com/documents/GuidelinesCodePublication.pdf)
