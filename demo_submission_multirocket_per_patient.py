"""
This is the main script that will create the predictions on input dataset and save a predictions file.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from models.multirocket import MultiRocket

# SETTINGS
DATA_DIR = Path("/dataset/dummy_test/test")  # Location of input test dataset
MODELS_DIR = "/models_out"
PREDICTIONS_DIR = "/submission"
PREDICTIONS_FILEPATH = f"{PREDICTIONS_DIR}/submission.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

print(os.listdir(os.getcwd()))
print(os.listdir(DATA_DIR))
print(os.listdir(PREDICTIONS_DIR))
print(os.listdir(MODELS_DIR))

if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# DEBUGGING INFO
print(f"Submission version {VERSION}")
# print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
print(f"GPU available:   {tf.test.is_gpu_available()}")  # Use this if using tensorflow

# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
test_files = []
patients = []
for patient in os.listdir(DATA_DIR):
    for session in os.listdir(DATA_DIR / patient):
        for filename in os.listdir(DATA_DIR / patient / session):
            test_files.append(Path(patient) / session / filename)
            patients.append(patient)
n_files = len(test_files)

dummy_test_map = {
    "1234": "1110",
    "3456": "2002",
    "5678": "1869",
}
# CREATE PREDICTIONS
print("Creating predictions.")
np.random.seed(1)
predictions = []
for i in range(n_files):
    # Load up the input dataset
    filepath = test_files[i]
    patient = patients[i]
    X = pd.read_parquet(DATA_DIR / filepath)
    X = X.fillna(0)
    X = X.transpose()
    X = X.values
    X = np.array(X)
    X = X.reshape((1, X.shape[0], X.shape[1]))
    # for local test
    X = X[:, :, :10:]
    # Print progress
    if (i) % 500 == 0:
        print(f"{(i + 1) / n_files * 100:0.2f}% ({filepath})")

    # LOAD MODEL
    print("Loading models.")
    model = MultiRocket(
        classifier="logistic",
        verbose=2,
        save_path=MODELS_DIR + "/" + dummy_test_map[patient] + "/"  # for dummy test
    )
    model.load()

    try:
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # YOUR MAGIC SAUCE HERE
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # Feed the input to some machine learning model, and get a prediction.
        prediction = model.predict(X)

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    except Exception as err:
        print(f"Failed to predict on file ({i + 1}/{n_files}) {filepath}")
        print(repr(err))
        raise

    # Append to your predictions (along with the file it corresponds with)
    predictions.append([str(filepath), prediction])

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions = pd.DataFrame(predictions, columns=["filepath", "prediction"])
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

print("Done!")
