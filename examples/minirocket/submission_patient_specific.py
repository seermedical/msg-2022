"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from train_model import load_parquet
from multiprocessing import Pool, cpu_count

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

batch_size = 128
pool = Pool(cpu_count())
predictions = []

for patient, group in test_data.groupby("patient"):
    n_batches = np.ceil(len(group) / batch_size)

    y_pred = []
    x_test = group["filepath"].map(lambda x: str(TEST_DATA_DIR / x))

    lr_model = load_model(TRAINED_MODEL_DIR/patient)
    with open(TRAINED_MODEL_DIR / patient /"minirocket.pkl", "rb") as f:
        minirocket = pickle.load(f)

    duration_tb = []
    for i in range(0, len(x_test), batch_size):
        batch_start_time = time.time()
        dt = np.array(pool.map(load_parquet, x_test[i: i + batch_size]))
        a = minirocket.transform(dt)
        pred = lr_model.predict(np.array(a), verbose=False)
        y_pred.extend(pred.ravel())

        batch_end_time = time.time()
        duration = batch_end_time - batch_start_time
        duration_tb.append(duration)
        remaining_batches = n_batches - (i / batch_size)
        print(
        f"Batches: {remaining_batches}/{n_batches} \t Batch Duration: {duration:.2f}s \t ETA: {np.mean(duration_tb) * remaining_batches:.2f}s "
        )

    group["prediction"] = y_pred
    predictions.append(group[["filepath", "prediction"]])
     
pool.close()
pool.terminate()


predictions = pd.concat(predictions)

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Done! Finished in {duration:.2f}s")
