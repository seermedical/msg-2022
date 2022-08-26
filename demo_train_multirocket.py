"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# SETTINGS
from models.multirocket import MultiRocket

# import tensorflow as tf
# import torch

DATA_DIR = Path("./data/train/")  # Location of input train data
PREDICTIONS_FILEPATH = "./submission/submission.csv"  # Output file.
TRAIN_LABELS_FILEPATH = "./data/train_labels.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

# DEBUGGING INFO
print(f"Submission version {VERSION}")
# print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
# print(f"GPU available:   {tf.test.is_gpu_available()}") # Use this if using tensorflow

# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
train_files = []
for patient in os.listdir(DATA_DIR):
    for session in os.listdir(DATA_DIR / patient):
        for filename in os.listdir(DATA_DIR / patient / session):
            train_files.append(Path(patient) / session / filename)
n_files = len(train_files)

# TRAIN MultiRocket
print("Training MultiRocket.")
x_train = []
y_train = []
train_labels = pd.read_csv(TRAIN_LABELS_FILEPATH)
all_train_labels = train_labels.filepath.values
for i in range(n_files):
    # Load up the input data
    filepath = train_files[i]
    train_labels_key = str(filepath).replace("\\", "/")
    assert train_labels_key in all_train_labels
    X = pd.read_parquet(DATA_DIR / filepath)

    # Print progress
    if (i) % 500 == 0:
        print(f"{(i + 1) / n_files * 100:0.2f}% ({filepath})")

    X = X.fillna(0)
    X = X.transpose()
    X = X.values
    x_train.append(X)
    y_train.append(train_labels.loc[train_labels.filepath == train_labels_key]["label"].values[0])
    break

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)

model = MultiRocket(verbose=2)
model.fit(x_train, y_train)

#
# np.random.seed(1)
# predictions = []
# for i in range(n_files):
#     # Load up the input data
#     filepath = train_files[i]
#     X = pd.read_parquet(DATA_DIR/filepath)
#
#     # Print progress
#     if (i) % 500 == 0:
#         print(f"{(i+1) / n_files * 100:0.2f}% ({filepath})")
#
#     try:
#         # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#         # YOUR MAGIC SAUCE HERE
#         # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#         # Feed the input to some machine learning model, and get a prediction.
#         prediction = np.random.rand()
#
#         # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#     except Exception as err:
#         print(f"Failed to predict on file ({i+1}/{n_files}) {filepath}")
#         print(repr(err))
#         raise
#
#     # Append to your predictions (along with the file it corresponds with)
#     predictions.append([str(filepath), prediction])
#
# # SAVE PREDICTIONS TO A CSV FILE
# print("Saving predictions.")
# predictions = pd.DataFrame(predictions, columns=["filepath", "prediction"])
# predictions.to_csv(PREDICTIONS_FILEPATH, index=False)
#
# print("Done!")
