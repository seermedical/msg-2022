"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import SGDW
from multiprocessing import Pool, cpu_count
from preprocess_data import transform_data
from p_tqdm import p_map
from classifier import Classifier

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
predictions = pd.DataFrame({"filepath": test_files})

# CREATE PREDICTIONS
print("Creating predictions.")

x_test = predictions["filepath"].map(lambda x: str(TEST_DATA_DIR / x)).tolist()

y_pred = []
batch_size = 128
n_batches = np.ceil(len(x_test) / batch_size)
pool = Pool(cpu_count())

classifier = Classifier(input_shape=[121, 9 * 129])
classifier.load(TRAINED_MODEL_DIR, load_weights=False)

duration_tb = []
for i in range(0, len(x_test), batch_size):
    batch_start_time = time.time()
    batch = x_test[i : i + batch_size]

    a = np.array(pool.map(transform_data, batch))
    pred = classifier.predict(a, verbose=False)
    y_pred.extend(pred.ravel())
    
    batch_end_time = time.time()
    duration = batch_end_time - batch_start_time
    duration_tb.append(duration)
    remaining_batches = n_batches - (i / batch_size)
    print(
        f"Batches: {remaining_batches}/{n_batches} \t Batch Duration: {duration:.2f}s \t ETA: {np.mean(duration_tb) * remaining_batches:.2f}s "
    )

pool.close()
pool.terminate()

predictions["prediction"] = y_pred

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Done! Finished in {duration:.2f}s")
