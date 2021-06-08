# data2text-macro-plan-py
This repo contains code for [Data-to-text Generation with Macro Planning](https://arxiv.org/abs/2102.02723) (Ratish Puduppully and Mirella Lapata;  In Transactions of the Association for Computational Linguistics (TACL)); this code is based on an earlier (version 0.9.2) fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Citation
```
@article{puduppully-2021-macro,
  author    = {Ratish Puduppully and Mirella Lapata},
  title     = {Data-to-text Generation with Macro Planning},
  journal = {Transactions of the Association for Computational Linguistics},
  year      = {2021},
  url       = {https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00381/101876/Data-to-text-Generation-with-Macro-Planning},
  pages = {510--527},
  volume = {9},
}
```

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Code Details
The `main` branch contains code to generate macro plans from input verbalization. The code for training summary generation is in `summary_gen` branch.

The test outputs and trained models can be downloaded from the google drive link https://drive.google.com/drive/folders/1jJjq5IvuBKNLTAe7fuwlDYParrxpK-WD?usp=sharing

The steps for training and inference for RotoWire dataset are given in [README_RotoWire](README_RotoWire.md), and for MLB dataset are given in [README_MLB](README_MLB.md).

