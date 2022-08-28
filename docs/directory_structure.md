# Expected Directory Structure

- [Go back to Main](README.md)


## Directory Structure for Training

```
data_dir
    train
        1110
            000
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                ...
            001
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                ...
            ...
        1869
            000
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                ...
            001
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                ...
            ...
        ...

    train_labels.csv

```



## Directory Structure for Testing

The directory structure of the test set that is used to evaluate your submission on our end will be the same, exept for the following differences:

1. The test samples will be within a separate `test` (instead of `train`) subdirectory.
2. `data_dir` needs to be set to the absolute path `/dataset/` in your docker container.

See [Docker submission inputs page](submission/inputs.md) for more details.

