This example uses Minirocket. The logistic regression is built with Tensorflow. This contains example for building data pipeline with tf.Data apis (`create_dataset` function in `train_model.py`).

Links to paper: [KDD 2021](https://dl.acm.org/doi/abs/10.1145/3447548.3467231) / [arXiv:2012.08791](https://arxiv.org/abs/2012.08791) (preprint)

Training can be done without Docker. However, if you want to create an isolated environment with docker, you can refer to the `Dockerfile_train` for reference.
Here are example command lines to run docker files locally:
- Build docker image for training: `docker build -f Dockerfile_train -t msg-docker-train`
- Train with GPU: `docker run --gpus all -v $path_to_dataset:/dataset -v  $model_output_path:/trained_model msg-docker-train`
- Build docker image for submission: `docker build -f Dockerfile -t msg-docker-submission`
- Test submission docker: `docker run --gpus all -v `pwd`/../../../../data/msg:/dataset -v `pwd`/submission:/submission msg-docker-submission`

This example uses GPU (note `--gpus all` flag).
