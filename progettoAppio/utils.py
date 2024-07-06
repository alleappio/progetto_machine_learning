import pandas as pd
import argparse
from matplotlib import pyplot as plt
import seaborn as sea
from json import dumps
import datetime
import numpy as np
import general_params as parameters
from sklearn import metrics

def init_log_file(filename, title, clean=False):
    if clean:
        with open(filename, "w", encoding = "utf-8") as f:
            f.write("")
    with open(filename, "a", encoding = "utf-8") as f:
        f.write(f"\n######## {title} ########\n")

def print_pretty_metrics(context, metrics):
    metrics_string=dumps(metrics, indent=2)
    print(
        f"{context}: {metrics_string}"
    )

def get_metrics(y_test, y_pred):
    calculated_metrics={
        "MAE": metrics.mean_absolute_error(y_test, y_pred),
        "MSE": metrics.mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        "R2": metrics.r2_score(y_test,y_pred),
    }
    return calculated_metrics

def save_metrics_to_file(context, metrics, filename):
    metrics_string=dumps(metrics, indent=2)
    with open(filename, "a", encoding = "utf-8") as f:
        f.write(f"\n{context}: {metrics_string}\n")

def save_features_to_file(context, features, filename):
    features_string=dumps(list(features), indent=2)
    with open(filename, "a", encoding = "utf-8") as f:
        f.write(f"\n{context}: {features_string}\n")

def cut_dataset(X, y, records_fraction):
    X = X.sample(frac=records_fraction)
    y = y[X.index]
    return X,y

def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str,
                        default='DataSet/superconductvty.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--title", type=str, default="o")
    parser.add_argument("--clean_file", type=str2bool, default=False)
    parser.add_argument("--feature_selection", type=str, default='filter')
    args = parser.parse_args()
    
    parameters.VERBOSE = args.verbose
    parameters.FEATURE_SELECTION_METHOD = args.feature_selection
    parameters.CV = args.cv
    return args

def print_info(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.columns)
    print(len(df.columns))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
