# Project Guidelines

## Overall
...

## Requirements 
* python 3.6-3.9
* pytorch 1.2.0-1.4.0

## Data statistics

| File name          | Num Records              |
| ------------------ | ------------------------ |
| Passage.json       | 250000                   |
| train_queries.json | 3000                     |


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
