from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate

import utils
import param_grids
import general_params as parameters
from prepare import PrepareData 
from feature_selector import FeatureSelectorFilter
from feature_selector import FeatureSelectorWrapper
from plotter import Plotter

from MLStrategies import LR
from MLStrategies import KNNR
from MLStrategies import DT
from MLStrategies import RF
from MLStrategies import SVM

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo
def verbose_log(msg):
    if parameters.VERBOSE:
        print(msg)

def get_dataset(args):
    if os.path.isfile(args.dataset_path):
        dataset = pd.read_csv(args.dataset_path)
    else:
        # fetch dataset
        verbose_log("fetch dataset")
        superconductivty_data = fetch_ucirepo(id=464)

        # data (as pandas dataframes)
        verbose_log("saving data in veriable")
        dataset = superconductivty_data.data.original
    
        verbose_log("saving data in file")
        dataset.to_csv("DataSet/superconductvty.csv", index=False)
    
    return dataset

def run_model(model, X_train, y_train):
    verbose_log(f"Training model {model.get_model_name()}\nCross validation: {str(parameters.CROSS_VALIDATION)}")
    model.set_X_y_train(X_train, y_train)
    if not parameters.CROSS_VALIDATION:
        model.train()
    if parameters.CROSS_VALIDATION:
        model.cv_train()

def find_best(models, X_train, y_train):
    scoring_rules = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    results = {}
    best_score = -np.inf
    decision_rule = 'test_r2'

    for model in models:
        result = cross_validate(model.get_model(), X_train, y_train, cv = 5, scoring = scoring_rules, return_train_score = False, n_jobs=-1)
        results[model] = result
    
    for model in models:
        current_score = np.mean(results[model][decision_rule]) 
        if current_score > best_score:
            best_score = current_score
            best_model = model
    
    return models.index(best_model)

def log(model_name, model_metrics, predictions, plotter_obj):
    utils.print_pretty_metrics(model_name, model_metrics)
    utils.save_metrics_to_file(model_name, model_metrics, parameters.FILENAME_SAVE_METRICS)
    plotter_obj.add_prediction(model_name, predictions[0], predictions[1])

def main():
    args = utils.read_args()
    
    dataset = get_dataset(args)
    
    plotter_obj = Plotter()

    utils.init_log_file(parameters.FILENAME_SAVE_METRICS, args.title, args.clean_file) 

    lr = LR()
    knn = KNNR()
    dt = DT()
    rf = RF()
    svr = SVM()

    lr.set_param_grid(param_grids.linear_regression)
    knn.set_param_grid(param_grids.KNN)
    dt.set_param_grid(param_grids.decision_tree)
    rf.set_param_grid(param_grids.random_forest)
    svr.set_param_grid(param_grids.SVR)
    
    model_list = [lr, knn, dt, rf, svr]

    pre_processed_data = PrepareData(dataset, parameters.TARGET) 
    # train data
    X_train, y_train = pre_processed_data.get_train_data()
    # test data
    X_test, y_test = pre_processed_data.get_test_data()
    
    X_train_cut, y_train_cut = utils.cut_dataset(X_train, y_train, parameters.DATASET_CUT_FRACTION)
    
    if parameters.FEATURE_SELECTION_METHOD == 'filter': 
        s = FeatureSelectorFilter(dataset, parameters.TARGET)
        selected_features = s.select_from_threshold(parameters.FEATURE_CORRELATION_THRESHOLD)
        utils.save_features_to_file(f"filter method", selected_features, parameters.FILENAME_SAVE_FEATURES)
        
        verbose_log(f"selected feaures: {selected_features}")
        
        X_train = X_train[selected_features] 
        X_train_cut = X_train_cut[selected_features] 
        X_test = X_test[selected_features]
        for model in model_list: 
            model.set_X_y_train(X_train,y_train)
            model.set_selected_features(selected_features)

    if parameters.FEATURE_SELECTION_METHOD == 'wrapper': 
        for model in model_list:
            s=FeatureSelectorWrapper(X_train_cut, y_train_cut, model.get_model())
            
            verbose_log(f"Calculating feature selection for model: {model.get_model_name()}")
            s.calc_sfs()

            selected_features = s.get_selected_features()
            utils.save_features_to_file(f"{model.get_model_name()} wrapper method", selected_features, parameters.FILENAME_SAVE_FEATURES)
            verbose_log(f"selected feaures: {selected_features}")
            
            model.set_X_y_train(X_train,y_train)
            model.set_selected_features(selected_features)

    for model in model_list: 
        run_model(model, X_train_cut, y_train_cut)
        print(model.get_model())

    best_model_index = find_best(model_list, X_train, y_train)
    best_model = model_list[best_model_index]

    best_model.set_X_y_train(X_train, y_train)
    best_model.train()
    scores = best_model.get_scores(X_test, y_test)
    log(best_model.get_model_name(), scores, best_model.get_predictions(), plotter_obj)

    y_pred, y_test = best_model.get_predictions()
    plotter_obj.save_single_plot(y_pred, y_test, parameters.DIRECTORY_SAVE_GRAPHS, args.title)
if __name__=='__main__':
    main()
