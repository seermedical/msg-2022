# My Seizure Gauge Forecasting Challenge 2022 - Docker Submission Example

**Contents of this page**

- [1. Requirements For Docker Submission](#1-requirements-for-docker-submission)
- [2. Build Docker Image](#2-build-docker-image)
- [3. Test Your Docker Container Locally](#3-test-your-docker-container-locally)
- [4. Submit your solution](#4-submit-your-solution)
  - [4.1. Preparation](#41-preparation)
  - [4.2. Make actual submission](#42-make-actual-submission)

## 1. Requirements For Docker Submission

1. Your application must load data from the `/dataset/test`  directory in the docker container.

    - The data is stored as separate files with the following directory structure:

    ```bash
    dataset
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
    - Should be saved to `/submission/submission.csv` in the docker container.
    - Structure of csv should look like:

    ```
    filepath,prediction
    path_to_file,probability
    path_to_file,probability
    path_to_file,probability
    ...
    ```

    - Example:

    ```
    filepath,prediction
    1111/002/UTC-2020_02_27-13_00_00.parquet,0.417022004702574
    1111/002/UTC-2020_02_27-12_00_00.parquet,0.7203244934421581
    1111/001/UTC-2020_02_26-22_00_00.parquet,0.00011437481734488664
    ```


## 2. Build Docker Image

1. Create your code, and [be mindful of compatibility issues](compatibility.md) ahead of time.
2. Edit the dockerfile, and [be mindful of these docker image tips](dockerfile.md).
3. Build it

    ```bash
    docker build --tag evalai-submission .
    ```

## 3. Test Your Docker Container Locally

To test that your Docker container does the right thing, do the following.

1. Download dummy version of the test dataset.
    - Download this zip file https://eval-ai-msg-data.s3.ap-southeast-2.amazonaws.com/dummy_test.zip
    - Extract it somewhere.
    - You should end up with a path that looks like `/path_to/dummy_test/test/`
2. Run the following commands:

    ```bash
    # ========================
    # SETTINGS
    # ========================
    # Change this one to point to where your `dummy_data` directory is stored.
    # NOTE: `LOCAL_DATA_DIR` and `LOCAL_PREDICTIONS_DIR` must be an absolute paths.
    LOCAL_DATA_DIR="$(pwd)/dummy_test"
    LOCAL_PREDICTIONS_DIR="$(pwd)/submission"

    # ========================
    # RUN DOCKER CONTAINER.
    # ========================
    # This will:
    # - Mount your local data dir to `/dataset` on the container.
    # - Mount your local predictions dir to `/submission` on the container.
    #   Which will allow you to see the results once docker contaier finishes
    #   running.
    # - Run the entrypoint script you specified in your docker container.
    docker run --rm\
        --name evalai-submission\
        -v ${LOCAL_DATA_DIR}:/dataset\
        -v ${LOCAL_PREDICTIONS_DIR}:/submission\
        evalai-submission
    ```

   - Running this will create a new directory `predictions` in your current working directory.
   - If the docker container is set up correctly, there should be a `predictions.csv` file saved in that directory.

3. Check the outputs of your script.
   - Open up the `submission/submission.csv` file and ensure it is structured correctly.
    - It should contain 12 rows of predictions, and look something like this:

    ```
    filepath,prediction
    1050/002/UTC-2019_11_14-02_00_00.parquet,0.417022004702574
    1050/002/UTC-2019_11_14-04_00_00.parquet,0.7203244934421581
    1050/002/UTC-2019_11_14-05_00_00.parquet,0.00011437481734488664
    1050/002/UTC-2019_11_14-03_00_00.parquet,0.30233257263183977
    1050/000/UTC-2019_11_11-17_00_00.parquet,0.14675589081711304
    1050/000/UTC-2019_11_11-16_00_00.parquet,0.0923385947687978
    1050/001/UTC-2019_11_12-15_00_00.parquet,0.1862602113776709
    1001/000/UTC-2020_02_27-19_00_00.parquet,0.34556072704304774
    1001/000/UTC-2020_02_27-17_00_00.parquet,0.39676747423066994
    1001/000/UTC-2020_02_27-18_00_00.parquet,0.538816734003357
    1001/001/UTC-2020_02_28-14_00_00.parquet,0.4191945144032948
    1001/001/UTC-2020_02_28-15_00_00.parquet,0.6852195003967595
    ```


## 4. Submit your solution
### 4.1. Preparation

1. Sign up for [the competition on Eval AI](https://eval.ai/web/challenges/challenge-page/1693/overview).
2. Install evalai cli tool

    ```bash
    pip install evalai
    ```

3. Add your eval AI authentication token to the cli tool. You can get the token by going to your [eval.ai profile](https://eval.ai/web/profile)

    ```bash
    evalai set_token XXXXXXXXXXXX
    ```

### 4.2. Make actual submission

```bash
# DEVELOPMENT PHASE
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase my-dev-1693
evalai push evalai-submission:latest --phase my-dev-1693

# ACTUAL SUBMISSION PHASE DURING COMPETITION
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase my-test-1693
evalai push evalai-submission:latest --phase my-test-1693

```
