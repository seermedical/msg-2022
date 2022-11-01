# Docker Submission - Test it Locally

- [Go back to Main](../README.md)


It is **strongly recomended** that you test run your docker container before making a submission. This will allow you catch any potential bugs you may have in your code.

To test that your Docker container does the right thing, do the following:

## 1. Prepare dummy version of the test dataset.

- Download this [zip file](https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/dummy_test.zip) containing dummy test data.
- Extract it somewhere.
- You should end up with a path that looks like `/path_to/dummy_test/test/`, and directory structure like:

```
path_to
    dummy_test
        test
            1110
                000
                    UTC-2020_02_26-23_10_00.parquet
                001
                    UTC-2020_02_27-17_30_00.parquet
                    UTC-2020_02_27-17_40_00.parquet
            1869
                000
                    UTC-2019_11_11-16_50_00.parquet
                    UTC-2019_11_11-17_00_00.parquet
                001
                    UTC-2019_11_12-16_00_00.parquet
                    UTC-2019_11_12-16_10_00.parquet
            1876
            ...
            ...
```


## 2. Run the container

```bash
# ========================
# SETTINGS
# ========================
# Change this one to point to where your `dummy_data` directory is stored.
# NOTE: `LOCAL_DATA_DIR` and `LOCAL_PREDICTIONS_DIR` must be an absolute paths.
# NOTE: `$(pwd)`  will evaluate to an absolute path of the working directory
#       on a linux computer.
LOCAL_DATA_DIR="$(pwd)/data/dummy_test"
LOCAL_PREDICTIONS_DIR="$(pwd)/submission"

# ========================
# RUN DOCKER CONTAINER.
# ========================
# NOTE: substitute `evalai-submission` with the docker image name and tag you created.
# This will:
# - Mount your local data dir to `/dataset` on the container.
# - Mount your local predictions dir to `/submission` on the container.
#   Which will allow you to see the results once docker contaier finishes
#   running.
# - Run the entrypoint script you specified in your docker container.
docker run --rm\
    --name mysubmission\
    -v ${LOCAL_DATA_DIR}:/dataset\
    -v ${LOCAL_PREDICTIONS_DIR}:/submission\
    evalai-submission
```

- Running this will create a new directory `submission` in your current working directory.

## 3. Check the outputs

- If the docker container ran correctly, there should be a `./submission/submission.csv` file saved relative to your working directory.
- Open up the `submission/submission.csv` file and ensure it is structured correctly.
- It should contain 23 rows of predictions, and look something like this:

```
filepath,prediction
filepath,prediction
2002/000/UTC-2020_12_06-21_20_00.parquet,0.417022004702574
2002/000/UTC-2020_12_06-21_10_00.parquet,0.7203244934421581
2002/001/UTC-2020_12_07-03_50_00.parquet,0.00011437481734488664
2002/001/UTC-2020_12_07-03_40_00.parquet,0.30233257263183977
1869/000/UTC-2019_11_11-16_50_00.parquet,0.14675589081711304
1869/000/UTC-2019_11_11-17_00_00.parquet,0.0923385947687978
1869/001/UTC-2019_11_12-16_00_00.parquet,0.1862602113776709
1869/001/UTC-2019_11_12-16_10_00.parquet,0.34556072704304774
1965/000/UTC-2020_08_24-15_10_00.parquet,0.39676747423066994
1965/000/UTC-2020_08_24-15_00_00.parquet,0.538816734003357
1965/001/UTC-2020_08_24-15_40_00.parquet,0.4191945144032948
1965/001/UTC-2020_08_24-15_30_00.parquet,0.6852195003967595
1110/000/UTC-2020_02_26-23_10_00.parquet,0.20445224973151743
1110/001/UTC-2020_02_27-17_40_00.parquet,0.8781174363909454
1110/001/UTC-2020_02_27-17_30_00.parquet,0.027387593197926163
1904/002/UTC-2020_03_25-16_30_00.parquet,0.6704675101784022
1904/002/UTC-2020_03_25-16_20_00.parquet,0.41730480236712697
1904/001/UTC-2020_03_22-16_10_00.parquet,0.5586898284457517
1904/001/UTC-2020_03_22-16_00_00.parquet,0.14038693859523377
1876/000/UTC-2019_12_19-20_30_00.parquet,0.1981014890848788
1876/000/UTC-2019_12_19-20_20_00.parquet,0.8007445686755367
1876/001/UTC-2019_12_19-21_30_00.parquet,0.9682615757193975
1876/001/UTC-2019_12_19-21_40_00.parquet,0.31342417815924284
```
