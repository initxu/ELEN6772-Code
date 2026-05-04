# Datasets

## NSL-KDD (auto-downloaded)

`src/data_loader.py` fetches `KDDTrain+.txt` and `KDDTest+.txt` from
<https://github.com/defcom17/NSL_KDD> on first import. No manual step needed.

## UNSW-NB15 (manual)

Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the
official page and drop them in this directory:

<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

After both datasets are present, this folder should contain:

```
KDDTrain+.txt
KDDTest+.txt
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```
