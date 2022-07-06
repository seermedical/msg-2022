# Docker Submission Example


## Requirements of your docker submission

1. Your application must load up data from the `/data/test`  directory in the docker container.

    - The data is structured as separate files with the following directory structure:

    ```bash
    data
        test 
            NNNN
                SSS
                    UTC-YYYY_MM_DD-hh_mm_ss.parquet
                    UTC-YYYY_MM_DD-hh_mm_ss.parquet
                SSS
                    UTC-YYYY_MM_DD-hh_mm_ss.parquet

    # NNNN                              = Patient id
    # SSS                               = Session id 
    # UTC-YYYY_MM_DD-hh_mm_ss.parquet   = Single input to your model
    ```

2. Your application should generate predictions, and save it as a csv.
    - Should be saved to `/predictions/predictions.csv` in the docker container.
    - Structure of csv should look like:

    ```
    filepath,prediction
    path_to_file,probability
    path_to_file,probability
    path_to_file,probability
    ...
    ```

    -  For example:

    ```
    filepath,prediction
    1111/002/UTC-2020_02_27-13_00_00.parquet,0.417022004702574
    1111/002/UTC-2020_02_27-12_00_00.parquet,0.7203244934421581
    1111/001/UTC-2020_02_26-22_00_00.parquet,0.00011437481734488664
    ```


## Build Docker Image

```bash
docker build --tag evalai-submission .

```

## Test your Docker container locally

To test that your docker container does the right thing, run the following.

```bash
# SETTINGS
# Change this one to point to where your `data` directory is stored.
# NOTE: `LOCAL_DATA_DIR` and `LOCAL_PREDICTIONS_DIR` must be an absolute paths..
LOCAL_DATA_DIR="$(pwd)/data"
LOCAL_PREDICTIONS_DIR="$(pwd)/predictions"

# RUN DOCKER CONTAINER.
# This will:
#    - Mount your local data dir to `/data` on the container.
#    - Mount your local predictions dir to `/predictions` on the container.
#      Which will allow you to see the results once docker contaier finishes
#      running.
#    - Run the entrypoint script you specified in your docker contianer.
docker run --rm\
    --name evalai-submission\
    -v ${LOCAL_DATA_DIR}:/data\
    -v ${LOCAL_PREDICTIONS_DIR}:/predictions\
    evalai-submission
```

- Running this script will create a new directory `predictions` in your current working directory.
- If the docker container is set up correctly, there should be a `predictions.csv` file saved in that directory.
- Open up the file to make sure it has the output structured correctly.
