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
from MLStrategies import DT
from MLStrategies import SVM

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
    linear_metrics = linear.get_scores()
    utils.print_pretty_metrics("Linear regression", linear_metrics)
    utils.save_metrics_to_file("Linear regression", linear_metrics, parameters.FILENAME_SAVE_METRICS)

    knn = KNNR(X_train, y_train, X_test, y_test)
    knn.train()
    knn_metrics = knn.get_scores()
    utils.print_pretty_metrics("K nearest neighbors", knn_metrics)
    utils.save_metrics_to_file("K nearest neighbors", knn_metrics, parameters.FILENAME_SAVE_METRICS)

    dt = DT(X_train, y_train, X_test, y_test)
    dt.train()
    dt_metrics = dt.get_scores()
    utils.print_pretty_metrics("Decision Tree", dt_metrics)
    utils.save_metrics_to_file("Decision Tree", dt_metrics, parameters.FILENAME_SAVE_METRICS)

    svr = SVM(X_train, y_train, X_test, y_test)
    svr.train()
    svr_metrics = svr.get_scores()
    utils.print_pretty_metrics("Support Vector Regression", svr_metrics)
    utils.save_metrics_to_file("Support Vector Regression", svr_metrics, parameters.FILENAME_SAVE_METRICS)

if __name__=='__main__':
    main()
