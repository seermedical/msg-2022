# Docker Submission - Test it Locally

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)


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
            1234
                000
                    UTC-2020_12_06-21_00_00.parquet
                    UTC-2020_12_06-21_10_00.parquet
                    UTC-2020_12_06-21_20_00.parquet
                001
                    UTC-2020_12_07-03_00_00.parquet
                    UTC-2020_12_07-03_10_00.parquet
            3456
                002
                    UTC-2020_12_08-03_00_00.parquet
                    UTC-2020_12_08-03_10_00.parquet
            5678
                000
                    UTC-2020_12_08-03_50_00.parquet
                    UTC-2020_12_08-04_00_00.parquet
                003
                    UTC-2020_12_09-03_30_00.parquet
                    UTC-2020_12_09-03_40_00.parquet
                    UTC-2020_12_09-03_50_00.parquet
```


## 2. Run the container

```bash
# ========================
# SETTINGS
# ========================
# Change this one to point to where your `dummy_data` directory is stored.
# NOTE: `LOCAL_DATA_DIR` and `LOCAL_PREDICTIONS_DIR` must be an absolute paths.
#       `$(pwd)`  will evaluate to an absolute path of the working directory
#       on a linux computer.
LOCAL_DATA_DIR="$(pwd)/dummy_test"
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

- To test GPU, you will need to install container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html. You will also need to use `--gpus all` flag with docker.  

- Running this will create a new directory `submission` in your current working directory.

## 3. Check the outputs

- If the docker container ran correctly, there should be a `./submission/submission.csv` file saved relative to your working directory.
- Open up the `submission/submission.csv` file and ensure it is structured correctly.
- It should contain 12 rows of predictions, and look something like this:

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
