from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

import utils
import general_params as parameters
from prepare import PrepareData
from feature_selector import FeatureSelector
from MLStrategies import LR
from MLStrategies import KNNR

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def main():
    args = utils.read_args()

    if os.path.isfile(args.dataset_path):
        dataset = pd.read_csv(args.dataset_path)
    else:
        # fetch dataset
        if args.verbose:
            print("fetch dataset")
        superconductivty_data = fetch_ucirepo(id=464)

        # data (as pandas dataframes)
        if args.verbose:
            print("saving data in veriable")
        dataset = superconductivty_data.data.original
    
        if args.verbose:
            print("saving data in file")
        dataset.to_csv("DataSet/superconductvty.csv", index=False)
    
    s = FeatureSelector(dataset, parameters.TARGET)
    selected_features = s.select_from_threshold(parameters.FEATURE_CORRELATION_THRESHOLD)
   
    if args.verbose:
        print(f"selected feaures: {selected_features}")
    
    new_data = dataset[selected_features]
    # Prepare the dataset
    new_data = PrepareData(new_data, parameters.TARGET)
    # train data
    X_train, y_train = new_data.get_train_data()
    # test data
    X_test, y_test = new_data.get_test_data()
    
    if args.verbose: 
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)
   
    linear = LR(X_train, y_train, X_test, y_test)
    linear.train()
    print(linear.get_scores())

if __name__=='__main__':
    main()
