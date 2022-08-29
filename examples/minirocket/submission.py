"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import time
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from train_model import train_model, tf_load_parquet
import numpy as np


start_time = time.time()

# SETTINGS
SEED = 42
TRAIN_DATA_DIR = Path("/dataset/train/")  # Location of input test data
TEST_DATA_DIR = Path("/dataset/test/")  # Location of input test data
PREDICTIONS_FILEPATH = "/submission/submission.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.


# TRAIN MODEL
print(f"GPU available:   {tf.test.is_gpu_available()}")  # Use this if using tensorflow
train_labels = pd.read_csv(os.path.join(TRAIN_DATA_DIR, "train_labels.csv"))
train_labels["filepath"] = train_labels["filepath"].map(
    lambda x: os.path.join(TRAIN_DATA_DIR, x)
)

# # SAMPLING DATA FOR DEMO
# train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
# sampled_data = []
# for patient, group in train_labels.groupby("patient"):
#     pos_samples = group[group["label"] == 1]
#     pos_samples = pos_samples.sample(np.min([100, pos_samples.shape[0]]))
#
#     neg_samples = group[group["label"] == 0].sample(100)
#     neg_samples = neg_samples.sample(np.min([100, neg_samples.shape[0]]))
#
#     sampled_data.extend([pos_samples, neg_samples])
#
# sampled_data = pd.concat(sampled_data)

X_train, X_validation, y_train, y_validation = train_test_split(
    train_labels["filepath"], train_labels["label"], test_size=0.2, random_state=SEED
)
###
print("Training model")
lr_model, minirocket = train_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    max_dilations=32,
    kernel_num=5000,
    epochs=2,
)

# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
test_files = []
for patient in os.listdir(TEST_DATA_DIR):
    for session in os.listdir(TEST_DATA_DIR / patient):
        for filename in os.listdir(TEST_DATA_DIR / patient / session):
            test_files.append(Path(patient) / session / filename)
predictions = pd.DataFrame({"filepath": test_files})

# CREATE PREDICTIONS
def minirocket_transform(x):
    return minirocket.transform(x.numpy())


def tf_minirocket_transform(x):
    return tf.py_function(minirocket_transform, [x], tf.float64)


print("Creating predictions.")

x_test = predictions["filepath"].map(lambda x: str(TEST_DATA_DIR / x))
test_dataset = (
    tf.data.Dataset.from_tensor_slices(x_test)
    .map(tf_load_parquet)
    .batch(32, drop_remainder=False)
    .map(tf_minirocket_transform)
)
y_pred = lr_model.predict(test_dataset).ravel()
predictions["prediction"] = y_pred

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Done! Finished in {duration:.2f}s")
