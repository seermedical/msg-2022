"""
This is the main script that will train the models per patient on input dataset and save the models.
"""
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from multirocket import MultiRocket
from tools import load_parquets, get_data

# SETTINGS
DATA_DIR = "D:/Dataset/msg_contest_data/train"  # Location of input train dataset
MODELS_DIR = "./trained_model"
TRAIN_LABELS_FILEPATH = "D:/Dataset/msg_contest_data/train/train_labels.csv"  # Output file.
VERSION = "v0.1.0"  # Training version. Optional and purely for logging purposes.
NUM_CPUS = 4
RAM_SIZE_THRESHOLD = 16 * 1024 * 1024 * 1024  # 16GB


def train_general_model(group, n_sample=-1, num_cpus=-1):
    train_files = group["filepath"].values
    y_train = group["label"].values

    mem = psutil.virtual_memory()
    print(f"MEM Available: {mem.available}, Threshold: {RAM_SIZE_THRESHOLD}")

    save_path = models_dir + "/general"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = MultiRocket(
        classifier="logistic",
        verbose=2,
        num_threads=num_cpus,
        save_path=save_path
    )

    if mem.available > RAM_SIZE_THRESHOLD:
        x_train, y_train, _ = load_parquets(train_files, y_train, n_sample=n_sample, num_cpus=num_cpus)

        print(x_train.shape, y_train.shape, len(np.unique(y_train)))

        model.fit(x_train, y_train)

        # SAVE MODEL
        print("Saving model.")
        model.save()

        del model, x_train, y_train
    else:
        sampled_data = get_data(group, n_sample=n_sample)

        model.fit_large(sampled_data)

        # SAVE MODEL
        print("Saving model.")
        model.save()

        del model


def train_patient_specific_model(train_labels, patients=None, n_sample=-1, num_cpus=-1):
    if patients is not None:
        train_labels = train_labels.loc[train_labels.patient.isin(patients)]

    mem = psutil.virtual_memory()
    print(f"MEM Available: {mem.available}, Threshold: {RAM_SIZE_THRESHOLD}")

    for patient, group in train_labels.groupby("patient"):
        train_files = group["filepath"].values
        y_train = group["label"].values

        save_path = models_dir + "/" + patient
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model = MultiRocket(
            classifier="logistic",
            verbose=2,
            num_threads=num_cpus,
            save_path=save_path
        )

        if mem.available > RAM_SIZE_THRESHOLD:
            x_train, y_train, _ = load_parquets(train_files, y_train, n_sample=n_sample, num_cpus=num_cpus)

            print(x_train.shape, y_train.shape, len(np.unique(y_train)))

            model.fit(x_train, y_train)

            # SAVE MODEL
            print("Saving model.")
            model.save()

            del model, x_train, y_train

        else:
            sampled_data = get_data(group, n_sample=n_sample)

            model.fit_large(sampled_data)

            # SAVE MODEL
            print("Saving model.")
            model.save()

            del model


MAIN_HERE = True
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default=DATA_DIR, required=False
    )
    arg_parser.add_argument(
        "--models-dir", type=str, default=MODELS_DIR, required=False
    )
    arg_parser.add_argument(
        "--training-mode",
        type=str,
        default="general",  # 1110,1869,1876,1904,1965,2002, patient-specific, general
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
    train_label_path = data_path / "train_labels.csv"
    models_dir = args.models_dir
    num_cpus = args.threads
    n_sample = args.sample
    training_mode = args.training_mode

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # DEBUGGING INFO
    print(f"Training version {VERSION}")
    # print(f"GPU available:   {torch.cuda.is_available()}")  # Use this if using pytorch
    print(f"GPU available:   {tf.config.list_physical_devices('GPU')}")  # Use this if using tensorflow

    # GET LIST OF ALL THE PARQUET FILES TO TRAIN ON
    print("Getting list of files to train on.")
    train_labels = pd.read_csv(train_label_path)
    train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    train_labels["filepath"] = [Path(x) for x in train_labels["filepath"]]
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(data_path, x)
    )
    all_train_labels = train_labels.filepath.values

    if training_mode == "general":
        train_general_model(train_labels, n_sample=n_sample, num_cpus=num_cpus)
    else:
        patients = None if training_mode == "patient-specific" else training_mode.split(",")
        train_patient_specific_model(train_labels, patients, n_sample=n_sample, num_cpus=num_cpus)

    print("Done!")
