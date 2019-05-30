from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes

thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# KNN
knn_dict = {'model': neighbors.KNeighborsClassifier(),
            'name': 'KNN',
            'params': {'metric': ['minkowski'],
                       'n_neighbors': [50],
                       'weights': ['uniform']}}

# Logistic regression
lr_dict = {'model': linear_model.LogisticRegression(),
           'name': 'Logistic regression',
           'params': {'penalty': ['l1', 'l2'],
                      'C': [1.0, 0.75, 0.5, 0.1]}}

# Decision tree
dtree_dict = {'model': tree.DecisionTreeClassifier(),
              'name': 'Decision tree',
              'params': {'criterion': ['gini', 'entropy'],
                         'splitter': ['best', 'random'],
                         'max_depth': [1,2,3,5,10,None]}}

# SVM
svm_dict = {'model': svm.SVC(),
            'name': 'SVM',
            'params': {'C': [1.0],
                       'kernel': ['rbf'],
                       'max_iter': [5],
                       'probability':[True]}}

# Random forest
rf_dict = {'model': ensemble.RandomForestClassifier(),
           'name': 'Random forest',
           'params': {'n_estimators': [10, 50, 100],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1,2,3,5,10,None],
                      'random_state': [100]}}               

# Bagging
bag_dict = {'model': ensemble.BaggingClassifier(),
            'name': 'Bagging',
            'params': {'base_estimator': [linear_model.LogisticRegression()],
                       'n_estimators': [10, 50, 100],
                       'max_samples': [1, 5, 10],
                       'max_features': [1, 5, 10],
                       'random_state': [100]}}

#Boosting
boost_dict = {'model': ensemble.GradientBoostingClassifier(),
              'name': 'Gradient boosting',
              'params': {'loss': ['deviance', 'exponential'],
                         'learning_rate': [0.5, 0.01, 0.01],
                         'n_estimators': [50, 100, 200],
                         'max_depth': [1, 3, 10]}}

# Naive Bayes
bayes_dict = {'model': naive_bayes.GaussianNB(),
              'name': 'Gaussian Naive Bayes',
              'params': {}}

# List of all models
model_list = [knn_dict, lr_dict, dtree_dict, svm_dict, rf_dict, bag_dict, boost_dict, bayes_dict]


