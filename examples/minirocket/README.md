This example uses Minirocket. The logistic regression is built with Tensorflow. This contains example for building data pipeline with tf.Data apis (`create_dataset` function in `train_model.py`).

There are 2 docker files:
- `Docker_train`: This is used to train the classification model.
- `Dockerfile`: This is used for submission.

The `run.sh` file contains commands to train model and test submission. The trained models will be saved in the `trained_model` folder. This will be added the submission's docker image.

This example uses GPU (note `--gpus all` flag).
