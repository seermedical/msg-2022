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
from train_model import tf_load_parquet, load_parquet
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessPool

start_time = time.time()

# SETTINGS
TRAINED_MODEL_DIR = Path("/trained_model/")  # Location of input test data
TEST_DATA_DIR = Path("/dataset/test/")  # Location of input test data
PREDICTIONS_FILEPATH = "/submission/submission.csv"  # Output file.
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.

lr_model = load_model(TRAINED_MODEL_DIR)
with open(TRAINED_MODEL_DIR / "minirocket.pkl", "rb") as f:
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
    x = load_parquet(x)
    # x = np.expand_dims(x, axis=0)
    return minirocket.transform(x)  # .values.ravel()


# def tf_minirocket_transform(x):
#     return tf.py_function(minirocket_transform, [x], tf.float64)


print("Creating predictions.")

x_test = predictions["filepath"].map(lambda x: str(TEST_DATA_DIR / x)).tolist()

y_pred = []
batch_size = 128
pool = Pool(cpu_count())
n_batches = np.ceil(len(x_test) / batch_size)

# read_start_time = time.time()
# a = np.array(pool.map(load_parquet, x_test))
# pool.close()
# pool.terminate()
# read_end_time = time.time()
#
# read_duration = read_end_time - read_start_time
# print(f"Finished reading in {read_duration:.2f}s")
#
# transform_start_time = time.time()
# a = minirocket.transform(a)
# transform_end_time = time.time()
# transform_duration = transform_end_time - transform_start_time
# print(f"Finished transforming in {transform_duration:.2f}s")
#
# y_pred = lr_model.predict(a, verbose=True).ravel()

duration_tb = []
for i in range(0, len(x_test), batch_size):
    batch_start_time = time.time()

    a = np.array(pool.map(load_parquet, x_test[i : i + batch_size]))
    a = minirocket.transform(a)
    pred = lr_model.predict(np.array(a), verbose=False)
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

# y_pred = lr_model.predict(np.array(transformed_x_test), verbose=True)

# print(x_test.shape)
# test_dataset = (
#     tf.data.Dataset.from_tensor_slices(x_test)
#     .map(tf_load_parquet, num_parallel_calls=tf.data.AUTOTUNE)
#     .batch(32, drop_remainder=False)
#     .map(tf_minirocket_transform, num_parallel_calls=tf.data.AUTOTUNE)
#     .prefetch(tf.data.experimental.AUTOTUNE)
# )
# data_options = tf.data.Options()
# data_options.threading.max_intra_op_parallelism = 1
# data_options.experimental_distribute.auto_shard_policy = (
#     tf.data.experimental.AutoShardPolicy.DATA
# )
# test_dataset = test_dataset.with_options(data_options)

predictions["prediction"] = y_pred

# SAVE PREDICTIONS TO A CSV FILE
print("Saving predictions.")
predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

end_time = time.time()
duration = end_time - start_time

print(f"Done! Finished in {duration:.2f}s")
