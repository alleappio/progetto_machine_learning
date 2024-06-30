import pandas as pd
import argparse
from matplotlib import pyplot as plt
import seaborn as sea
from json import dumps

def print_pretty_metrics(context, metrics):
    metrics_string=dumps(metrics, indent=2)
    print(
        f"{context}: {metrics_string}"
    )

def save_metrics_to_file(context, metrics, filename):
    metrics_string=dumps(metrics, indent=2)
    with open(filename, "a", encoding = "utf-8") as f:
        f.write(f"{context}: {metrics_string}\n")

def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str,
                        default='DataSet/superconductvty.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--cv", type=int, default=5)

    args = parser.parse_args()

    return args

def print_info(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.columns)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
