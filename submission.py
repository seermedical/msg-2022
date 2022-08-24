"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
# import tensorflow as tf
# import torch

# SETTINGS
DATA_DIR = Path("/dataset/test/") # Location of input test data
PREDICTIONS_FILEPATH = "/submission/submission.csv" # Output file.
VERSION = "v0.1.0" # Submission version. Optional and purely for logging purposes.

# DEBUGGING INFO
print(f"Submission version {VERSION}")
# print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
# print(f"GPU available:   {tf.test.is_gpu_available()}") # Use this if using tensorflow

# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
test_files = []
for patient in os.listdir(DATA_DIR):
    for session in os.listdir(DATA_DIR/patient):
        for filename in os.listdir(DATA_DIR/patient/session):
            test_files.append(Path(patient)/session/filename)
n_files = len(test_files)

# CREATE PREDICTIONS
print("Creating predictions.")
np.random.seed(1)
predictions = []
for i in range(n_files):
    # Load up the input data
    filepath = test_files[i]
    X = pd.read_parquet(DATA_DIR/filepath)

    # Print progress
    if (i) % 500 == 0:
        print(f"{(i+1) / n_files * 100:0.2f}% ({filepath})")

    try:
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # YOUR MAGIC SAUCE HERE
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # Feed the input to some machine learning model, and get a prediction.
        prediction = np.random.rand()

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    except Exception as err:
        print(f"Failed to predict on file ({i+1}/{n_files}) {filepath}")
        print(repr(err))
        raise

    # Append to your predictions (along with the file it corresponds with)
    predictions.append([str(filepath), prediction])

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions = pd.DataFrame(predictions, columns=["filepath", "prediction"])
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

print("Done!")
