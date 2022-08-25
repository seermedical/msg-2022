# Editing the Dockerfile 

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)
- [Go back to Creating Docker Image](create_docker.md)

## Tips

1. If using GPU, for maximum compatibility with deep-learning libraries, we recomend building docker image with  cuda toolkit version `11.2`.
2. Use `python -m pip install` instead of `pip install` for installing python libraries.
3. Set the `--no-cache-dir` argument when installing python libraries. This keeps docker images smaller.

    ```bash
    # Example
    python -m pip install --no-cache-dir pandas
    ```

4. After installing linux packages, delete the repository cache (this keeps docker images smaller)

    ```bash
    # Exammple of code to delete the apt package cache
    rm -rf /var/lib/apt/lists/*
    ```

5. Only install the bare minimum you need to run predictions. It will keep docker images smaller.
    - During the early stages, you will probably make use of a lot of libraries to perform exploratory analysis. E.g., for visualizing data, creating plots, and exporting images.
    - Do not install all of those libraries on your system onto the Docker image that you submit. 
        - E.g. there is probably no need to be installing `matplotlib` or `plotly` to make predictions.
