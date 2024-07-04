
decision_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5]
}

linear_regression = {}

SVR = {
    'kernel': ['linear', 'rbf'],
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01]
}

KNN = {
    'n_neighbors': [3, 5, 7, 9]
}        

random_forest = {
    'n_estimators': [50, 100, 200],  # Numero di alberi nella foresta
    'max_depth': [10, 20, 30]       # Profondità massima degli alberi
}
