"""
This is the main script that will create the predictions on input data and save a predictions file.
"""
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from multirocket import MultiRocket
from tools import load_parquets

# SETTINGS
TRAINED_MODEL_DIR = "./trained_model/"  # Location of input test data
TEST_DATA_DIR = Path("../../dataset/dummy_test/test/")  # Location of input test data
PREDICTIONS_DIR = "./submission"
VERSION = "v0.1.0"  # Submission version. Optional and purely for logging purposes.
NUM_CPUS = 4
RAM_SIZE_THRESHOLD = 16 * 1024 * 1024 * 1024  # 16GB


def predict_general_model(data_path, models_dir, num_cpus):
    # GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
    print("Getting list of files to run predictions on.")
    test_files = []
    test_files2 = []
    for patient in os.listdir(data_path):
        for session in os.listdir(data_path / patient):
            for filename in os.listdir(data_path / patient / session):
                filepath = Path(patient) / session / filename
                test_files.append(filepath)
                test_files2.append(data_path / filepath)

    # LOAD DATA
    x_test, files = load_parquets(test_files2, y=None, n_sample=-1, num_cpus=num_cpus)

    # LOAD MODEL
    print("Loading models.")
    model = MultiRocket(
        classifier="logistic",
        verbose=2,
        num_threads=num_cpus,
        save_path=models_dir + "/general"
    )
    model.load()

    # CREATE PREDICTIONS
    print("Creating predictions.")
    prediction = model.predict(x_test)

    predictions = pd.DataFrame({
        "filepath": test_files,
        "prediction": prediction.reshape((-1,))
    })

    return predictions


def predict_patient_specific_model(data_path, models_dir, num_cpus, patients=None):
    if patients is None:
        patients = os.listdir(data_path)
    # GET LIST OF ALL THE PARQUET FILES TO DO PREDICTIONS ON
    print("Getting list of files to run predictions on.")
    test_files = []
    test_files2 = []
    for patient in patients:
        for session in os.listdir(data_path / patient):
            for filename in os.listdir(data_path / patient / session):
                filepath = Path(patient) / session / filename
                test_files.append(filepath)
                test_files2.append(data_path / filepath)
    test_data = pd.DataFrame({"filepath": test_files})
    test_data["patient"] = test_data["filepath"].map(lambda x: str(x).split("\\")[0])

    # LOAD DATA
    mem = psutil.virtual_memory()
    print(f"MEM Available: {mem.available}, Threshold: {RAM_SIZE_THRESHOLD}")

    if mem.available > RAM_SIZE_THRESHOLD:
        x_test, files = load_parquets(test_files2, y=None, n_sample=-1, num_cpus=num_cpus)
        files = [str(x).replace(str(data_path) + "\\", "") for x in files]

        unique_patients = np.unique(test_data["patient"])

        predictions = []
        for patient in unique_patients:
            _x_test = []
            _files = []
            for i in range(len(files)):
                if patient in files[i]:
                    _x_test.append(x_test[i])
                    _files.append(files[i])

            _x_test = np.array(_x_test)

            # LOAD MODEL
            print("Loading models.")
            model = MultiRocket(
                classifier="logistic",
                verbose=2,
                num_threads=num_cpus,
                save_path=models_dir + "/" + patient
            )
            model.load()

            # CREATE PREDICTIONS
            print("Creating predictions.")
            prediction = model.predict(_x_test)

            predictions.append(pd.DataFrame({
                "filepath": _files,
                "prediction": prediction.reshape((-1,))
            }))

        predictions = pd.concat(predictions)

        return predictions
    else:
        predictions = []
        for patient, group in test_data.groupby("patient"):
            x_test = group["filepath"].map(lambda x: str(TEST_DATA_DIR / x))

            # LOAD MODEL
            print("Loading models.")
            model = MultiRocket(
                classifier="logistic",
                verbose=2,
                num_threads=num_cpus,
                save_path=models_dir + "/" + patient
            )
            model.load()
            y_pred = model.predict_large(x_test)
            group["prediction"] = y_pred
            predictions.append(group[["filepath", "prediction"]])

        predictions = pd.concat(predictions)

        return predictions


MAIN_HERE = True
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default=TEST_DATA_DIR, required=False
    )
    arg_parser.add_argument(
        "--models-dir", type=str, default=TRAINED_MODEL_DIR, required=False
    )
    arg_parser.add_argument(
        "--predictions-path", type=str, default=PREDICTIONS_DIR, required=False
    )
    arg_parser.add_argument(
        "--prediction-mode",
        type=str,
        default="2002",  # 1110,1869,1876,1904,1965,2002, patient-specific, general
        required=False,
    )
    arg_parser.add_argument(
        "--threads",
        type=int,
        default=NUM_CPUS,
        required=False
    )
    arg_parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        required=False
    )
    args = arg_parser.parse_args()

    data_path = Path(args.data_path)
    predictions_path = Path(args.predictions_path)
    models_dir = args.models_dir
    num_cpus = args.threads
    n_sample = args.sample
    prediction_mode = args.prediction_mode

    PREDICTIONS_FILEPATH = f"{predictions_path}/submission.csv"  # Output file.

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    # DEBUGGING INFO
    print(f"Submission version {VERSION}")
    # print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
    print(f"GPU available:   {tf.config.list_physical_devices('GPU')}")  # Use this if using tensorflow

    start_time = time.perf_counter()

    if prediction_mode == "general":
        predictions = predict_general_model(data_path=data_path, models_dir=models_dir, num_cpus=num_cpus)
    else:
        patients = None if prediction_mode == "patient-specific" else prediction_mode.split(",")
        predictions = predict_patient_specific_model(
            data_path=data_path,
            models_dir=models_dir,
            num_cpus=num_cpus,
            patients=patients
        )

    # SAVE PREDICTIONS TO A CSV FILE
    print("Saving predictions.")
    predictions.to_csv(PREDICTIONS_FILEPATH, index=False)

    duration = time.perf_counter() - start_time

    print(f"Done! Finished in {duration:.2f}s")
