# Docker Submission - Required Outputs

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)


Your application should generate predictions, and save them as a `csv` file.

- The csv file should be saved to `/submission/submission.csv` in the docker container.
  - **Note** that `/submission` is an absolute directory path, mounted on the root directory. 
  - It is not a path relative to the current working directory.
- The csv file shoud have two columns:
  - `filepath`
  - `prediction`
- Structure of csv should look like this:

```
filepath,prediction
path_to_file,probability
path_to_file,probability
path_to_file,probability
...
```

- The `path_to_file` value should be the path of the input file, relative to the data split directory (eg `train`, `valid`, `test` directory).
- The `probability` value must be a probability value, between `0` and `1` (inclusively).


- Example:

```
filepath,prediction
1234/000/UTC-2020_12_06-21_00_00.parquet,0.417022004702574
1234/000/UTC-2020_12_06-21_10_00.parquet,0.7203244934421581
1234/000/UTC-2020_12_06-21_20_00.parquet,0.00011437481734488664
1234/001/UTC-2020_12_07-03_00_00.parquet,0.30233257263183977
1234/001/UTC-2020_12_07-03_10_00.parquet,0.14675589081711304
3456/002/UTC-2020_12_08-03_00_00.parquet,0.0923385947687978
3456/002/UTC-2020_12_08-03_10_00.parquet,0.1862602113776709
5678/000/UTC-2020_12_08-03_50_00.parquet,0.34556072704304774
5678/000/UTC-2020_12_08-04_00_00.parquet,0.39676747423066994
5678/003/UTC-2020_12_09-03_30_00.parquet,0.538816734003357
5678/003/UTC-2020_12_09-03_40_00.parquet,0.4191945144032948
5678/003/UTC-2020_12_09-03_50_00.parquet,0.6852195003967595
```
