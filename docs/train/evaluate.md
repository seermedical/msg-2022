# Evaluation

- [Go back to Main](../README.md)

## Evaluate Your Results

You can make use of the `evaluate()` function in the `evaluate.py` file to check how well your model's predictions are performing, using the same evaluation metric that is used in the leaderboard. 

```python
from evaluate import evaluate

score = evaluate(preds=preds, labels=labels)
print(f"SCORE IS: {score}")
#> SCORE IS: 0.4444444444444444
```

This assumes that you have a `preds` dataframe containing predictions that looks something like this:

```
| filepath                                 | prediction |
| ---------------------------------------- | ---------- |
| 2002/000/UTC-2020_12_06-21_20_00.parquet | 0.03987    |
| 2002/000/UTC-2020_12_06-21_10_00.parquet | 0.37411    |
| 2002/001/UTC-2020_12_07-03_50_00.parquet | 0.71543    |
| 2002/001/UTC-2020_12_07-03_40_00.parquet | 0.51344    |
| 1869/000/UTC-2019_11_11-16_50_00.parquet | 0.43246    |
| 1869/000/UTC-2019_11_11-17_00_00.parquet | 0.18766    |
| ...                                      | ...        |

```

And a corresponding `labels` dataframe that looks something like this:

```
| filepath                                 | label |
| ---------------------------------------- | ----- |
| 2002/000/UTC-2020_12_06-21_20_00.parquet | 1     |
| 2002/000/UTC-2020_12_06-21_10_00.parquet | 0     |
| 2002/001/UTC-2020_12_07-03_50_00.parquet | 1     |
| 2002/001/UTC-2020_12_07-03_40_00.parquet | 1     |
| 1869/000/UTC-2019_11_11-16_50_00.parquet | 1     |
| 1869/000/UTC-2019_11_11-17_00_00.parquet | 1     |
| ...                                      | ...   |

```

