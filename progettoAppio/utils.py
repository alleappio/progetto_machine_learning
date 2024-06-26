import pandas as pd
import argparse


def read_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path_train', type=str,
                        default='DATABASE PET CEREBRALI ANONIMO-Train.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument('--dataset_path_test', type=str,
                        default='DATABASE PET CEREBRALI ANONIMO-Test.csv',
                        help='Path to the file containing the test set.')
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
