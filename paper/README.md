To recreate Fig 2:

- Download and unzip preprocessed Norman2019 dataloader for GEARS [here](https://dataverse.harvard.edu/api/access/datafile/6894431)
- Move the uncompressed `norman2019.tar.gz` folder to `./data`. It should contain the subdirectory `data_pyg`
- Move `essential_norman.pkl` and `go_essential_norman.csv` to `./data`
- So `./data` should contain `essential_norman.pkl`, `go_essential_norman.csv` and `norman2019/data_pyg/`
- Run `fig2_train.py` to train the model

The code here will not install GEARS from the [main repository](https://github.com/snap-stanford/GEARS). It will use the local path to GEARS in this repository `../gears`

For other baselines:
- CPA: See `CPA_reproduce`
- GRN: See `GRN`
- CellOracle: See `CellOracle`

Please raise an issue or email yhr@cs.stanford.edu in case of any problems/questions
