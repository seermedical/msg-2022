import os
from pathlib import Path
import pandas as pd
import numpy as np

# Python Dependencies
# pyarrow
# fastparquet

# SETTINGS
DATA_DIR = Path("/data/test/") # Location of input test data
PREDICTIONS_FILEPATH = "./predictions.csv"  # Output file


# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS FOR
test_files = []
for patient in os.listdir(DATA_DIR):
    for session in os.listdir(DATA_DIR/patient):
        for filename in os.listdir(DATA_DIR/patient/session):
            test_files.append(Path(patient)/session/filename)


# CREATE PREDICTIONS
np.random.seed(1)
predictions = []
for filepath in test_files:
    # Load up the input data
    X = pd.read_parquet(DATA_DIR/filepath)

    # Feed the input to some machine learning model, and get a prediction.
    prediction = np.random.rand()

    # Append to your predictions (along with the file it corresponds with)
    predictions.append([str(filepath), prediction])

# SAVE PREDICTIONS TO A CSV FILE
predictions = pd.DataFrame(predictions, columns=["filepath", "prediction"])
predictions.to_csv(PREDICTIONS_FILEPATH)
