VERBOSE = False
TARGET = 'critical_temp'
FEATURE_CORRELATION_THRESHOLD = 0.5
FILENAME_SAVE_METRICS = "log/log_metrics"
STRATEGIES_LIST = ['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR']
#FEATURE_SELECTION_METHOD = 'filter'
FEATURE_SELECTION_METHOD = 'wrapper'
FEATURE_SELECTION_CUT_STEP = 5
FEATURE_SELECTION_CUT_DROP_NUM = 5 
