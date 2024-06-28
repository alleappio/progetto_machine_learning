import pandas as pd
import argparse
from matplotlib import pyplot as plt
import seaborn as sea

def plot_corr_matrix(df: pd.DataFrame, verbose: bool = False):
    corr_matrix = df.corr()
    sea.heatmap(corr_matrix, annot=True)
    plt.show()
    if verbose:
        print("\nCorrelation of each column to MPG (the target).")
        print(corr_matrix['MPG'])


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
