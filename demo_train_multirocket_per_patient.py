"""
This is the main script that will train the models per patient on input dataset and save the models.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal

from models.multirocket import MultiRocket

# SETTINGS
DATA_DIR = Path("D:/Dataset/msg_contest_data/train")  # Location of input train dataset
MODELS_DIR = "./models_out"
PREDICTIONS_DIR = "./submission"
PREDICTIONS_FILEPATH = f"{PREDICTIONS_DIR}/submission.csv"  # Output file.
TRAIN_LABELS_FILEPATH = "D:/Dataset/msg_contest_data/train_labels.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# DEBUGGING INFO
print(f"Submission version {VERSION}")
# print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
print(f"GPU available:   {tf.test.is_gpu_available()}")  # Use this if using tensorflow

# GET LIST OF ALL THE PARQUET FILES TO TRAIN ON
print("Getting list of files to train on.")
for patient in os.listdir(DATA_DIR):
    train_files = []
    for session in os.listdir(DATA_DIR / patient):
        for filename in os.listdir(DATA_DIR / patient / session):
            train_files.append(Path(patient) / session / filename)
    n_files = len(train_files)

    # TRAIN MULTIROCKET
    print(f"Training MultiRocket on {patient} with {n_files} files.")
    x_train = []
    y_train = []
    train_labels = pd.read_csv(TRAIN_LABELS_FILEPATH)
    all_train_labels = train_labels.filepath.values
    for i in range(n_files):
        # Load up the input dataset
        filepath = train_files[i]
        train_labels_key = str(filepath).replace("\\", "/")
        assert train_labels_key in all_train_labels
        X = pd.read_parquet(DATA_DIR / filepath)

        # Print progress
        if (i) % 100 == 0:
            print(f"{(i + 1) / n_files * 100:0.2f}% ({filepath})")

        X = X.fillna(0)
        X = X.transpose()
        X = X.values

        # resample
        X = signal.resample_poly(X, up=64, down=128, axis=-1)

        x_train.append(X)
        y_train.append(train_labels.loc[train_labels.filepath == train_labels_key]["label"].values[0])

        # for local test
        # if i == 100:
        #     break

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(x_train.shape, y_train.shape, len(np.unique(y_train)))

    save_path = MODELS_DIR + "/" + patient
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = MultiRocket(
        classifier="logistic",
        verbose=2,
        save_path=save_path
    )
    model.fit(x_train, y_train)

    # SAVE MODEL
    print("Saving model.")
    model.save()

print("Done!")
