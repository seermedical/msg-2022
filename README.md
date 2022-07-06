# Docker Submission Example


## Requireements of your docker submission

<!-- 1. Your application must load up data from the `/data/test`  directory. -->

2. Must downoad data from bucket specified in the `DATA_BUCKET_FILE` environment variable.
    - There is a `curl` command in `submission.sh` that does this already.
    - However, you can use a different way of downlaoding from the bucket if you like (e.g. if your docker base image does not come with curl installed).
3. Once you have created your predictions, you must save them as:

    ```bash
    XXX: TODO: how should predictions be saved? 
                As a csv? where?
    ```

## Build Docker Image

```bash
docker build --tag evalai-submission .

```

## Test your Docker container locally

```bash
# XXX: TODO: use a dummy bucket file that actually has data of real structure.
# DUMMY_BUCKET_DATA="https://ronny-test-evalai-data.s3.ap-southeast-2.amazonaws.com/scrap_data.csv"
    # --env DATA_BUCKET_FILE=${DUMMY_BUCKET_DATA}\

LOCAL_DATA_DIR="$(pwd)/data"
docker run --rm\
    --name evalai-submission\
    -v ${LOCAL_DATA_DIR}:/data\
    evalai-submission


```

