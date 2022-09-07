# IR0 Project

## Dataset
Passage num: 3473
Query num: 3000 (train), 300 (validation), 300 (test)

## Pipleline
Please run the following scripts sequentially
```
### Full ranking on the training, validation and test sets
Python main.py --mode inference --ranking full_ranking --dataset train
Python main.py --mode inference --ranking full_ranking --dataset validation
Python main.py --mode inference --ranking full_ranking --dataset test
```

### Re ranking: training on the training set, and inference on the validation and test sets
```
Python main.py --mode train --ranking re_ranking --dataset train --replicability
Python main.py --mode inference --ranking re_ranking --dataset validation
Python main.py --mode inference --ranking re_ranking --dataset test
```

### Evaluation in terms of full ranking and re ranking on the validation and test sets
```
Python main.py --mode evaluation --ranking full_ranking --dataset validation
Python main.py --mode evaluation --ranking re_ranking --dataset validation
Python main.py --mode evaluation --ranking full_ranking --dataset test
Python main.py --mode evaluation --ranking re_ranking --dataset test
```

## Result
```
# The result of full ranking on the validation set
MRR@100: 0.3148
NDCG@100: 0.6032

# The result of re ranking on the validation set
MRR@100: 0.3203
NDCG@100: 0.6111

# The result of full ranking on the test set
MRR@100: 0.2858
NDCG@100: 0.5578

# The result of re ranking on the test set
MRR@100: 0.2885
NDCG@100: 0.5623
```
