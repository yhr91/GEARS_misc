To recreate Fig 2:

- Run this [notebook](https://github.com/yhr91/GEARS_misc/blob/main/paper/reproduce_preprint_results.ipynb)

To recreate Fig 4:
- First train the model using this [script](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py)
- Then run inference for all combinations using this [script](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_predict.py)
- After that you can produce the UMAP using this [notebook](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4.ipynb)

The code here will not install GEARS from the [main repository](https://github.com/snap-stanford/GEARS). It will use the local path to GEARS in this repository `../gears`

For other baselines:
- CPA: See `CPA_reproduce`
- GRN: See `GRN`
- CellOracle: See `CellOracle`

Please raise an issue or email yhr@cs.stanford.edu in case of any problems/questions
