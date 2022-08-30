"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import time
import pickle
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from train_model import tf_load_parquet


start_time = time.time()

# SETTINGS
TRAINED_MODEL_DIR = Path("/trained_model/")  # Location of input test data
TEST_DATA_DIR = Path("/dataset/test/")  # Location of input test data
PREDICTIONS_FILEPATH = "/submission/submission.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

lr_model = load_model(TRAINED_MODEL_DIR)
with open(TRAINED_MODEL_DIR/"minirocket.pkl", "rb") as f:
    minirocket = pickle.load(f)

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
