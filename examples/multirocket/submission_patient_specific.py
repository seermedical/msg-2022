"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from train_model import tf_load_parquet

start_time = time.time()

# SETTINGS
TRAINED_MODEL_DIR = Path("/trained_model/")  # Location of input test data
TEST_DATA_DIR = Path("/dataset/test/")  # Location of input test data
PREDICTIONS_FILEPATH = "/submission/submission.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

# GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
print("Getting list of files to run predictions on.")
test_files = []
for patient in os.listdir(TEST_DATA_DIR):
    for session in os.listdir(TEST_DATA_DIR / patient):
        for filename in os.listdir(TEST_DATA_DIR / patient / session):
            test_files.append(Path(patient) / session / filename)
test_data = pd.DataFrame({"filepath": test_files})
test_data["patient"] = test_data["filepath"].map(lambda x: str(x).split("/")[0])

print("Creating predictions.")

predictions = []
for patient, group in test_data.groupby("patient"):
    x_test = group["filepath"].map(lambda x: str(TEST_DATA_DIR / x))
    lr_model = tf.keras.models.load_model(TRAINED_MODEL_DIR / patient)
    with open(TRAINED_MODEL_DIR / patient / "multirocket.pkl", "rb") as f:
        multirocket = pickle.load(f)


    # CREATE PREDICTIONS
    def multirocket_transform(x):
        return multirocket.transform(x.numpy())


    def tf_multirocket_transform(x):
        return tf.py_function(multirocket_transform, [x], tf.float64)


    test_dataset = (
        tf.data.Dataset.from_tensor_slices(x_test)
            .map(tf_load_parquet)
            .batch(32, drop_remainder=False)
            .map(tf_multirocket_transform)
    )
    y_pred = lr_model.predict(test_dataset).ravel()
    group["prediction"] = y_pred
    predictions.append(group[["filepath", "prediction"]])

predictions = pd.concat(predictions)

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Done! Finished in {duration:.2f}s")
