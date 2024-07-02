from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

import utils
import general_params as parameters
from prepare import PrepareData
from feature_selector import FeatureSelectorFilter
from feature_selector import FeatureSelectorWrapper
from MLStrategies import LR
from MLStrategies import KNNR
from MLStrategies import DT
from MLStrategies import SVM

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def get_dataset(args):
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
    
    return dataset

def run_model(model, model_name, X_train, y_train):
    if parameters.FEATURE_SELECTION_METHOD == 'wrapper':
        s=FeatureSelectorWrapper(X_train, y_train, model.get_model())
        s.cut_dataset()
        s.calc_rfe()
        model.set_selected_features(s.get_selected_features())
    
    model.train()
    model_metrics = model.get_scores()
    utils.print_pretty_metrics(model_name, model_metrics)
    utils.save_metrics_to_file(model_name, model_metrics, parameters.FILENAME_SAVE_METRICS)

def main():
    args = utils.read_args()
    dataset = get_dataset(args)
     
    utils.init_log_file(parameters.FILENAME_SAVE_METRICS, args.title, args.clean_file) 
    
    if parameters.FEATURE_SELECTION_METHOD == 'filter': 
        s = FeatureSelectorFilter(dataset, parameters.TARGET)
        selected_features = s.select_from_threshold(parameters.FEATURE_CORRELATION_THRESHOLD)
       
        #if args.verbose:
        print(f"selected feaures: {selected_features}")
        new_data = dataset[selected_features]
        utils.print_info(new_data)
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
    
    if parameters.FEATURE_SELECTION_METHOD == 'wrapper': 
        new_data = PrepareData(dataset, parameters.TARGET)
        # train data
        X_train, y_train = new_data.get_train_data()
        # test data
        X_test, y_test = new_data.get_test_data()
    

    linear = LR(X_train, y_train, X_test, y_test)
    run_model(linear, 'Linear regression', X_train, y_train) 
    
    knn = KNNR(X_train, y_train, X_test, y_test)
    run_model(knn, 'K Nearest Neighbors', X_train, y_train) 
    
    dt = DT(X_train, y_train, X_test, y_test)
    run_model(dt, 'Decision tree', X_train, y_train) 

    svr = SVM(X_train, y_train, X_test, y_test)
    run_model(svr, 'Support Vector Machine', X_train, y_train)

if __name__=='__main__':
    main()
