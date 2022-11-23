This example uses InceptionTime. Please refer to the original InceptionTime paper: [KDD 2020](https://link.springer.com/article/10.1007/s10618-020-00710-y) / [arXiv:2012.08791](https://arxiv.org/pdf/1909.04939.pdf) (preprint).

InceptionTime has also been used to detect interictal epileptiform discharges from epilepsy patients. [Paper](https://www.worldscientific.com/doi/10.1142/S0129065723500016.)

This also contains example for building data pipeline with tf.Data apis (`create_dataset` function in `train_model.py`).

The example contains the following files:
- `inception.py`: this file contains the modified version of InceptionTime.
- `classifier.py`: this file contains the code for the classifier.
- `train_model.py`: this is the main file which create the dataset and train the model.
- `preprocess_data`: this is the code for preprocessing the data.
- `Dockerfile`: the script for building docker image for submission.

Once the model is trained, you can follow the following steps to build and test your Docker submission:
- Build docker image for submission: `docker build -f Dockerfile -t msg-submission`
- Test submission docker: 
```sh
docker run --gpus all cpus 4 -v `pwd`/../../../../data/msg:/dataset -v `pwd`/submission:/submission msg-submission`
```
This limits `cpus` to 4 to match the specifications of the provided server.