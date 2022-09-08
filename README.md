# Project Guidelines

## Overall

* what are the input (data) and output (submit to leaderboard)?
* how many modules? 
* what is the fuction of each module? what are the input and output of each module?
* what the students shoud not do? (e.g., cannot use sklearn and other off-the-shelf pacakges for ranking)
* How the students can improve the pipeline? (e.g., do stem and more advanced word segmentation method ... implement more effective ranking models)

## Requirements 
* python 3.6-3.9
* pytorch 
* Numpy

## Data statistics

| Datasets                          | File                     |Num                       |Num                       |
| ------------------                | ------------------------ |------------------------ |------------------------ |
| Passages                          | 250000                   | 250000                   |  |
| Queries on the training set       | 3000                     |3000                     |  |
| Queries on the validation set     | 3000                     |                            |  |
| Queries on the test set.          | 3000                     |                      |  |
| Labels on the training set                           | 3000                     |                   |  |
| Labels on the validation set                          | 3000                     |                   |  |
| Labels on the on the test set (unavailable)          | 3000                     |

## Pipleline
Please run the following scripts sequentially.

### Full ranking on the training, validation and test sets
```
Python main.py --mode infer --ranking_type full_ranking --dataset training
Python main.py --mode infer --ranking_type full_ranking --dataset validation
Python main.py --mode infer --ranking_type full_ranking --dataset test
```

### Re ranking: training on the training set, and inference on the validation and test sets
```
Python main.py --mode train --ranking_type re_ranking --dataset training --replicability
Python main.py --mode infer --ranking_type re_ranking --dataset validation
Python main.py --mode infer --ranking_type re_ranking --dataset test
```

### Evaluation in terms of full ranking and re ranking on the validation and test sets
```
Python main.py --mode evaluate --ranking_type full_ranking --dataset validation
Python main.py --mode evaluate --ranking_type re_ranking --dataset validation
Python main.py --mode evaluate --ranking_type full_ranking --dataset test
Python main.py --mode evaluate --ranking_type re_ranking --dataset test
```

## Result
```
# The result of full ranking on the validation set
MRR@100: 0.3137
NDCG@100: 0.3803

# The result of re ranking on the validation set
MRR@100: 0.3172
NDCG@100: 0.3843

# The result of full ranking on the test set
MRR@100: 0.2831
NDCG@100: 0.3506

# The result of re ranking on the test set
MRR@100: 0.2949
NDCG@100: 0.3602
```
## Submission to the leaderboard
