# Docker Submission - Inputs

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)


Your application must load data from the `/dataset/test`  directory in the docker container.

The data is stored as separate files with the following directory structure:

```bash
dataset
    test 
        NNNN
            SSS
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
                UTC-YYYY_MM_DD-hh_mm_ss.parquet
            SSS
                UTC-YYYY_MM_DD-hh_mm_ss.parquet

# --------------------------------------------------------------
# KEYS
# --------------------------------------------------------------
# NNNN                              = Patient id
# SSS                               = Session id 
# UTC-YYYY_MM_DD-hh_mm_ss.parquet   = Single input to your model
```
