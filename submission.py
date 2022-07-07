"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np


# SETTINGS
DATA_DIR = Path("/data/test/") # Location of input test data
PREDICTIONS_FILEPATH = "/predictions/predictions.csv" # Output file.


# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
test_files = []
for patient in os.listdir(DATA_DIR):
    for session in os.listdir(DATA_DIR/patient):
        for filename in os.listdir(DATA_DIR/patient/session):
            test_files.append(Path(patient)/session/filename)

# CREATE PREDICTIONS
print("Creating predictions.")
np.random.seed(1)
predictions = []
for filepath in test_files:
    # Load up the input data
    X = pd.read_parquet(DATA_DIR/filepath)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # YOUR MAGIC SAUCE HERE
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Feed the input to some machine learning model, and get a prediction.
    print(f"Predicting on file {filepath}")
    prediction = np.random.rand()
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    # Append to your predictions (along with the file it corresponds with)
    predictions.append([str(filepath), prediction])

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions = pd.DataFrame(predictions, columns=["filepath", "prediction"])
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)
