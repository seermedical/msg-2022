# Docker Submission Example


## Requireements of your docker submission

1. Your application must load up data from the `/data/test`  directory.

    - The data is structured as:

    ```
    TODO: visual representation of the data.
    ```

2. Your application should generate predictions, and save it as a csv.

    - Structure of csv should look like:

    ```bash
    TODO: structure of CSV, title and data

    ```

    - Csv file should be saved as `TODO: filename and location?`

## Build Docker Image

```bash
docker build --tag evalai-submission .

```

## Test your Docker container locally

To test that your docker 
```bash
# Change this one to point to where your `data` directory is stored.
# NOTE: `LOCAL_DATA_DIR` must be an absolute path.
# NOTE: there must be a `test/` subdirectory within there.
LOCAL_DATA_DIR="$(pwd)/data"

docker run --rm\
    --name evalai-submission\
    -v ${LOCAL_DATA_DIR}:/data\
    evalai-submission

```
