from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os
from json import dumps

from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import utils
import hyper_param_grids as param_grids
import general_params as parameters
from prepare import PrepareData 

from model_creator import ModelCreator

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

def find_best(models, X_train, y_train, scoring_rule):
    scoring_rules = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    results = {}
    metrics = {}
    best_score = -np.inf
    decision_rule = f'test_{scoring_rule}'

    for model in models:
        print(model.get_pipe())
        result = cross_validate(model.get_pipe(), X_train, y_train, cv = 5, scoring = scoring_rules, return_train_score = False, n_jobs=-1)
        results[model] = result
        metrics[model.name] = {
            "MAE": np.abs(np.mean(result[f'test_{scoring_rules[0]}'])),
            "MSE": np.abs(np.mean(result[f'test_{scoring_rules[1]}'])),
            "RMSE": np.abs(np.mean(result[f'test_{scoring_rules[2]}'])),
            "R2": np.mean(result[f'test_{scoring_rules[3]}']),
        }
    verbose_log("metrics:")
    verbose_log(dumps(metrics, indent=2))

    for model in models:
        current_score = np.mean(results[model][decision_rule]) 
        if current_score > best_score:
            best_score = current_score
            best_model = model
    return models.index(best_model)

def main():
    args = utils.read_args()
     
    dataset = get_dataset(args)
    
    pre_processed_data = PrepareData(dataset, parameters.TARGET) 
    # train data
    X_train, y_train = pre_processed_data.get_train_data()
    # test data
    X_test, y_test = pre_processed_data.get_test_data()
    
    X_train_cut, y_train_cut = utils.cut_dataset(X_train, y_train, parameters.DATASET_CUT_FRACTION)
    verbose_log(f"X_train_cut:{X_train_cut.shape} y_train_cut:{y_train_cut.shape}") 
    
    knn = ModelCreator('k nearest neighbors', 'knn')
    knn.set_model_estimator(KNeighborsRegressor())
    knn.set_pipe_estimator()
    
    knn_fs = ModelCreator('k nearest neighbors-feature selection', 'knn_fs')
    knn_fs.set_model_estimator(KNeighborsRegressor())
    knn_fs.set_pipe_corr_feature_selection(parameters.FEATURE_CORRELATION_THRESHOLD)
    knn_fs.set_pipe_estimator()
    
    dt = ModelCreator('decision tree', 'dt')
    dt.set_model_estimator(DecisionTreeRegressor())
    dt.set_pipe_estimator()
    
    dt_fs = ModelCreator('decision tree-feature selection', 'dt_fs')
    dt_fs.set_model_estimator(DecisionTreeRegressor())
    dt_fs.set_pipe_corr_feature_selection(parameters.FEATURE_CORRELATION_THRESHOLD)
    dt_fs.set_pipe_estimator()

    rf = ModelCreator('random forest', 'rf')
    rf.set_model_estimator(RandomForestRegressor())
    rf.set_pipe_estimator()
    
    rf_fs = ModelCreator('random forest-feature selection', 'rf_fs')
    rf_fs.set_model_estimator(RandomForestRegressor())
    rf_fs.set_pipe_corr_feature_selection(parameters.FEATURE_CORRELATION_THRESHOLD)
    rf_fs.set_pipe_estimator()

    svr = ModelCreator('SVR', 'svr')
    svr.set_model_estimator(SVR())
    svr.set_pipe_estimator()
    
    svr_fs = ModelCreator('SVR-feature selection', 'svr_fs')
    svr_fs.set_model_estimator(SVR())
    svr_fs.set_pipe_corr_feature_selection(parameters.FEATURE_CORRELATION_THRESHOLD)
    svr_fs.set_pipe_estimator()
   
    # model_list = [knn, knn_fs, dt, dt_fs, rf, rf_fs, svr, svr_fs]
    model_list = [knn_fs, dt_fs, rf_fs, svr_fs]
    # model_list = [knn, dt, rf, svr]
    hparam_dic = {
        'knn': param_grids.KNN,
        'dt': param_grids.decision_tree,
        'rf': param_grids.random_forest,
        'svr': param_grids.SVR,
        'knn_fs': param_grids.KNN,
        'dt_fs': param_grids.decision_tree,
        'rf_fs': param_grids.random_forest,
        'svr_fs': param_grids.SVR
    }

    for model in model_list:
        verbose_log(f"creating {model.name}")
        model.assemble_pipe()
        model.do_pipe_grid_search(hparam_dic[model.abbreviation], parameters.GENERAL_SCORING_RULE, parameters.CV, X_train_cut, y_train_cut)
        verbose_log(f"best {model.name}: {model.get_pipe()}")
     
    best_model_index = find_best(model_list, X_train_cut, y_train_cut, parameters.GENERAL_SCORING_RULE)
    best_model = model_list[best_model_index]
    best_model_pipe = best_model.get_pipe()
    verbose_log(f"Best model:{best_model.name}")

    best_model_pipe = best_model_pipe.fit(X_train, y_train)
    
    y_pred = best_model_pipe.predict(X_test)
    metrics = utils.get_metrics(y_pred, y_test)

    utils.print_pretty_metrics(best_model.name, metrics)
    utils.save_target_plot(y_pred, y_test, parameters.DIRECTORY_SAVE_GRAPHS, args.title, best_model.name, parameters.TARGET)
    utils.init_log_file(parameters.FILENAME_SAVE_METRICS, args.title, clean=args.clean_file)
    utils.save_metrics_to_file(best_model.name, metrics, parameters.FILENAME_SAVE_METRICS)

if __name__=='__main__':
    main()
