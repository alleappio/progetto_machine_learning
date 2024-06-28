from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

import utils
import general_params as parameters
from prepare import PrepareData
from feature_selector import FeatureSelector
# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def main():
    print("reading args")
    args = utils.read_args()

    if os.path.isfile(args.dataset_path):
        dataset = pd.read_csv(args.dataset_path)
    else:
        # fetch dataset
        print("fetch dataset")
        superconductivty_data = fetch_ucirepo(id=464)

        # data (as pandas dataframes)
        print("saving data in veriable")
        dataset = superconductivty_data.data.original
        print("saving data in file")
        dataset.to_csv("DataSet/superconductvty.csv", index=False)
    
    s = FeatureSelector(dataset, parameters.TARGET)
    selected_features = s.select_from_threshold(parameters.FEATURE_CORRELATION_THRESHOLD)

    # Prepare the dataset
    new_data = PrepareData(dataset, parameters.TARGET)
    # train data
    X_train, y_train = new_data.get_train_data()
    # test data
    X_test, y_test = new_data.get_test_data()
    
    #print(X_train)
    #print(y_train)
    #print(X_test)
    #print(y_test)


if __name__=='__main__':
    main()
