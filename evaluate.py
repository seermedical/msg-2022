import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate(preds, labels):
    """Given two dataframes, one containing the predictions and one containing
    the labels, it returns the Patient-Wise ROC AUC (average value of the
    separate ROC-AUC scores for each patient).

    Args:
        preds: Pandas dataframe with the following two columns `filepath`,
            `prediction`.
        labels: Pandas dataframe with the following two columns `filepath`,
            `label`.

    Returns:
        score: Float
    """
    # Combine `preds` and `labels` into one DataFrame
    df = labels.merge(preds, on="filepath")

    # Add a column to select predictions by patient ID easily
    df["patient_id"] = df["filepath"].apply(lambda fp: fp.split("/")[0])

    # Separate results by patient id
    patient_ids = list(df["patient_id"].unique())
    patient_aucs = {}
    for patient_id in patient_ids:
        selection  = df[df["patient_id"] == patient_id]
        patient_aucs[patient_id] = roc_auc_score(selection["label"], selection["prediction"])

    # Average of AUC results accross patients
    score = pd.Series(patient_aucs).mean()
    return score

