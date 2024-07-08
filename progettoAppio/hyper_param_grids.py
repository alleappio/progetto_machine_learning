
decision_tree = {
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__min_samples_split': [2, 5]
}

linear_regression = {}

SVR = {
    'estimator__kernel': ['linear', 'rbf'],
    'estimator__C': [1, 10, 100],
    'estimator__gamma': [0.001, 0.01]
}

KNN = {
    'estimator__n_neighbors': [3, 5, 7, 9]
}        

random_forest = {
    'estimator__n_estimators': [100, 200, 300],  # Numero di alberi nella foresta
    'estimator__max_depth': [10, 20, 30]       # Profondità massima degli alberi
}
