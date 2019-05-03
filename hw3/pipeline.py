'''
CAPP30254 S'19: Assignment 3
Improving the Machine Learning Pipeline

Alec MacMillen
'''

import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors, datasets, linear_model, tree, svm, ensemble, metrics, utils
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature
import graphviz


def load_data(filename, dtypes=None):
    '''
    Load data in a CSV to a Pandas dataframe

    Inputs: filename (str), the file path to the input CSV
      dtypes (dict), dictionary of column name to input type (default None)
    Returns: pandas data frame
    '''
    df = pd.read_csv(filename, dtype=dtypes)
    return df


def generate_metadata(dataframe):
    '''
    Generate a metadata data frame with information about column types
    and null values.

    Inputs: dataframe (pd df), the data frame you want to generate
      a summary of
    Returns: pandas data frame with metadata
    '''
    cols = list(dataframe.columns)
    meta = pd.DataFrame(cols, columns=["colname"])

    meta.loc[:, "type"] = meta["colname"].apply(lambda x:
        type(dataframe[x].iloc[0]))

    meta.loc[:, "pct_null"] = meta["colname"].apply(lambda x:
        dataframe[x].isna().sum() / len(dataframe))

    return meta


def generate_summary(dataframe, outliers=True):
    '''
    Return summary statistics of numeric variables

    Inputs: dataframe (pd df), the dataframe with numeric variables
        to summarize
      outliers (Bool), True if to include outliers and False if to
        exclude (based on IQR)

    Returns: pandas data frame with summary statistics
    '''
    meta = generate_metadata(dataframe)
    numeric = []
    for row in meta.iterrows():
        if np.issubdtype(row[1]["type"], np.number):
            numeric.append(row[1]["colname"])

    if not numeric:
        return None

    if not outliers:
        cols = ['colname', 'mean', 'median', 'min', 'max', 'std_dev', 'count']
        summary = pd.DataFrame(columns=cols)
        for var in numeric:
            varsum = pd.DataFrame([var], columns=["colname"])
            outliers = identify_outliers(dataframe, var)
            df = dataframe[~outliers]
            varsum.loc[:, "mean"] = varsum["colname"].apply(lambda x:
                df[x].mean())
            varsum.loc[:, "median"] = varsum["colname"].apply(lambda x:
                df[x].median())
            varsum.loc[:, "min"] = varsum["colname"].apply(lambda x: 
                min(df[x]))
            varsum.loc[:, "max"] = varsum["colname"].apply(lambda x:
                max(df[x]))
            varsum.loc[:, "std_dev"] = varsum["colname"].apply(lambda x:
                df[x].std())
            varsum.loc[:, "count"] = varsum["colname"].apply(lambda x:
                len(df[x]))
            summary = pd.concat([summary, varsum])

    else:
        summary = pd.DataFrame(numeric, columns=["colname"])
        summary.loc[:, "mean"] = summary["colname"].apply(lambda x:
            dataframe[x].mean())
        summary.loc[:, "median"] = summary["colname"].apply(lambda x:
            dataframe[x].median())
        summary.loc[:, "min"] = summary["colname"].apply(lambda x: 
            min(dataframe[x]))
        summary.loc[:, "max"] = summary["colname"].apply(lambda x:
            max(dataframe[x]))
        summary.loc[:, "std_dev"] = summary["colname"].apply(lambda x:
            dataframe[x].std())
        summary.loc[:, "count"] = summary["colname"].apply(lambda x:
            len(dataframe[x]))

    return summary


def output_numeric_vars(dataframe):
    '''
    Output only the numeric fields of input dataframe.

    Inputs: dataframe (pandas dataframe)
    Returns: dataframe containing only the numeric variables
    '''
    meta = generate_metadata(dataframe)
    numeric = []
    for row in meta.iterrows():
        if np.issubdtype(row[1]["type"], np.number):
            numeric.append(row[1]["colname"])

    return_df = dataframe[numeric]
    return return_df


def identify_outliers(dataframe, colname):
    '''
    Determine whether individual observations are outliers using IQR

    Inputs: dataframe (pandas dataframe)
      colname (str), the column for which to analyze outliers

    Returns: pandas series showing truth value (True/False) for whether
      observation at given index location in "dataframe" is an outlier 
      for variable "colname"
    '''
    q1 = dataframe[colname].quantile(0.25)
    q3 = dataframe[colname].quantile(0.75)
    iqr = q3 - q1
    outliers = (dataframe[colname] > (q3 + 1.5*iqr)) | \
        (dataframe[colname] < (q1 - 1.5*iqr))
    return outliers


def return_outliers(dataframe, colname):
    '''
    Return full data frame of outliers along a particular attribute

    Inputs: dataframe (pandas dataframe)
      colname (str), the column for which to analyze outliers

    Returns: pandas dataframe consisting of observations that have outlier
      (unusually high/low) values for the given colname
    '''
    idx = identify_outliers(dataframe, colname)
    return_df = dataframe[idx]
    return return_df


def generate_histogram(dataframe, colname, color, binwidth, title):
    '''
    Generates a histogram for a given variable while EXCLUDING outlier values
    (both high and low) for that given variable. Rounds max and min values
    to the nearest (binwidth) to establish range.

    Inputs: dataframe (pandas dataframe)
      colname (str), the column to return a histogram for
      color (str), color for the histogram bars
      binwidth (int or float), number describing the width of histogram bins
      title (str), plot title

    Returns: matplotlib plot object (inline)
    '''
    outliers = identify_outliers(dataframe, colname)
    df = dataframe[~outliers]
    maximum = binwidth * round(max(df[colname])/binwidth)
    minimum = binwidth * round(min(df[colname])/binwidth)
    bw = (maximum-minimum)/binwidth
    plt.hist(df[colname], color=color, edgecolor='black', bins=int(bw))
    plt.title(title)
    plt.xlabel(colname)
    plt.ylabel('Count')
    plt.show()


def generate_boxplot(dataframe, colname, category=None, hue=None):
    '''
    Generate a boxplot using pyplot to be printed inline

    Inputs: dataframe (pandas dataframe)
      colname (str): column of analysis
      category (str): by-group category
      hue (str): second by-group category
    '''
    outliers = identify_outliers(dataframe, colname)
    df = dataframe[~outliers]
    if category:
        outliers = identify_outliers(dataframe, category)
        df = df[~outliers]
        ax = sns.boxplot(x=colname, y=category, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + category
        plt.title(title)
        plt.show()

    elif hue:
        outliers = identify_outliers(dataframe, hue)
        df = df[~outliers]
        ax = sns.boxplot(x=colname, hue=hue, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + hue
        plt.title(title)
        plt.show()

    elif hue and category:
        outliers = identify_outliers(dataframe, category)
        df = df[~outliers]
        outliers = identify_outliers(dataframe, hue)
        df = df[~outliers]
        ax = sns.boxplot(x=colname, y=category, hue=hue, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + " and ".join([category, hue])
        plt.title(title)
        plt.show()

    else:
        ax = sns.boxplot(x=colname, data=df, palette='Set1')
        title = "Box plot of " + colname
        plt.title(title)
        plt.show()


def correlation_heatmap(dataframe, size=10):
    '''
    Create a correlation heatmap from a dataframe's numeric variables.
    Inspiration from https://stackoverflow.com/questions/39409866/correlation-heatmap

    Inputs: dataframe (pandas dataframe)
      size (int): plot size to be displayed
    '''
    corr = dataframe.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def plot_scatter(dataframe, x, y, outliers=True):
    '''
    Create a scatter plot using matplotlib.pyplot.

    Inputs: dataframe (pandas df)
      x (str): horizontal-axis var, column from dataframe
      y (str): vertical-axis var, column from dataframe

    Returns: nothing, prints graphic inline
    '''
    df = dataframe
    if not outliers:
        outlier_x = identify_outliers(dataframe, x)
        outlier_y = identify_outliers(dataframe, y)
        df = dataframe[~(outlier_x|outlier_y)]
    ax = plt.scatter(df[x], df[y], s=1, c="black")
    title = "Plot of " + x + " against " + y
    plt.title(title)
    plt.show()


def fill_na_values(dataframe, colname, how="median"):
    '''
    Fill NaN or missing values from a numeric column with either the
    median or mean of the nonmissing values from that column.

    Inputs: dataframe (pandas df)
      colname (str): the column name for which to impute missing variables
      how (str): "median" or "mean" to tell pandas which measure of central
        tendency to use. Default is "median", if something other than
        "median" or "mean" is entered, NaN's are filled with 0's.

    Returns: updated pandas df
    '''
    df = dataframe
    if how == "median":
        fill = df[colname].median()
    elif how == "mean":
        fill = df[colname].mean()
    else:
        fill = 0
    
    df[colname].fillna(value=fill, inplace=True)
    return df


def discretize_continuous(dataframe, colname, bins, labels):
    '''
    Discretize continuous variables by placing them into discrete "buckets".

    Inputs: dataframe (pandas df)
      colname (str): column to discretize
      bins (list): list of bucket boundaries
      labels (list): labels for categories

    Returns: updated pandas df with select continuous variables converted to categorical
    '''
    newcol = colname + "_cat"
    dataframe[newcol] = pd.cut(
        dataframe[colname], bins=bins, labels=labels, include_lowest=True, right=False)
    return dataframe


def dummify_categorical(dataframe, colnames):
    '''
    Convert categorical variables to dummy variables. Use this to prepare
    encoded data to be used in learning/testing ML model.

    Inputs: dataframe (pandas df)
      colnames (list): column names of categorical variables to convert
        to dummies

    Returns:
      encoded pandas df with categorical variables converted to numeric dummies
    '''
    return_df = pd.get_dummies(dataframe[colnames], drop_first=True)
    return return_df


def create_random_splits(
    dataframe, features, target, test_size, random_state=1000):
    '''
    Create train-test split from master dataset for use in learning and testing
    ML model.

    Inputs:
      dataframe (pandas df)
      features (list of features)
      target (str, variable for which to predict outcome)

    Returns:
      Train and test df's for feature (x) and classification label (y) variables
    '''
    x = dataframe[features]
    y = dataframe[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def create_date_splits(
    dataframe, features, target, date_col, train_dates, test_dates, convert=False):
    '''
    Create train-test split based on input dates for use in learning and testing
    ML model. 

    Inputs:
      dataframe (pandas df): dataframe of all input data
      features (list of str): column names of feature/predictor vars
      target (str): column name of target variable to label/predict
      date_col (str): name of date column in dataset to separate date sets on
      train_dates (tuple of str): takes format ('mm-dd-yyyy', 'mm-dd-yyyy') to
        delineate the training data period
      test_dates (tuple of str): takes format ('mm-dd-yyyy', 'mm-dd-yyyy') to
        delineate the testing data period
      convert (bool, default False): if True, convert date column using 
        pd.to_datetime() string method

    Returns:
      Train and test df's for feature (x) and classification label (y) variables
    '''
    if convert:
        dataframe['date_use'] = pd.to_datetime(dataframe[date_col])
    else:
        dataframe['date_use'] = dataframe[date_col]

    train_start, train_end = train_dates
    test_start, test_end = test_dates
    
    train_filter = (dataframe['date_use'] >= train_start) & (dataframe['date_use'] <= train_end)
    train_df = dataframe[train_filter]

    test_filter = (dataframe['date_use'] >= test_start) & (dataframe['date_use'] <= test_end)
    test_df = dataframe[test_filter]

    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    return x_train, x_test, y_train, y_test




####################
#K-NEAREST NEIGHBORS
####################
def train_knn(features, target, n, weights='uniform', metric='euclidean', p=2):
    '''
    Instantiate an object of the KNN model class and train it on 
    given features to predict a given target value.

    Inputs:
      features (pd df or list of series): pandas dataframe containing 
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      n (int): number of neighbors on which to train the model
      weights (str): weight of points, default is 'uniform' (all points
        count the same regardless of weight), can change to 'distance'
        to weight by inverse of distance
      metric (str): default 'euclidean', can be 'manhattan', 'chebyshev',
        or 'minkowski'

    Returns:
      trained k-nearest neighbors model object
    '''
    knn = neighbors.KNeighborsClassifier(n, weights=weights, metric=metric, p=p)
    knn.fit(features, target)
    return knn


def knn_loop(
    xtrain, ytrain, xtest, ytest, train_date, test_date, neighbors, metrics_list, weights, thresholds):
    '''
    Loop through KNN models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data
      neighbors, metrics_list, weights (lists): lists of knn parameters
        to iterate over 
      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','neighbors','metric', 'weights',
        'threshold','accuracy','precision','recall','f1','auc'])
    for neighbor in neighbors:
        for metric in metrics_list:
            for weight in weights:
                for threshold in thresholds:
                    knn = train_knn(xtrain, ytrain, n=neighbor, weights=weight, metric=metric)
                    pred_scores = knn.predict_proba(xtest)
                    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                    accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                    precision = calculate_precision_at_threshold(ytest, pred_labels)
                    recall = calculate_recall_at_threshold(ytest, pred_labels)
                    f1 = metrics.f1_score(ytest, pred_labels)
                    auc = metrics.roc_auc_score(ytest, pred_labels)
                    summary.loc[len(summary)] = ['K-Nearest Neighbors',
                        train_date, test_date, neighbor, metric, weights, 
                        threshold, accuracy, precision, recall, f1, auc]
    return summary




####################
#LOGISTIC REGRESSION
####################
def train_logistic(
    features, target, penalty='l2', C=1.0, class_weight=None, random_state=100):
    '''
    Instantiate an object of the logistic regression model class and train it
    on given features to predict a given target value for a new input observation.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      penalty (str): 'l1' or 'l2' (default), specifies the norm used in
        penalization. 'l2' is ridge regression
      C (list of floats or int): specifies the regularization strength. Smaller
        C represents stronger regularization 
      class_weight (dict or 'balanced'): if not given, all classes have weight 1.
        'balanced' mode uses the values of y to automatically adjust weights 
        inversely proportional to class frequencies in the input data
      random_state (int): random seed to initialize the model

    Returns:
      trained logistic regression model object
    '''
    logistic = linear_model.LogisticRegression(
        penalty=penalty, C=C, class_weight=class_weight, random_state=random_state)
    logistic.fit(features, target)
    return logistic


def logistic_loop(
    xtrain, ytrain, xtest, ytest, train_date, test_date, penalties, cs, thresholds):
    '''
    Loop through logistic regression models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data
      penalties, cs (lists): lists of logistic regression parameters to iterate over
      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','penalty','C',
        'threshold','accuracy','precision','recall','f1','auc'])
    for penalty in penalties:
        for c in cs:
            for threshold in thresholds:
                logistic = train_logistic(xtrain, ytrain, penalty=penalty, C=c)
                pred_scores = logistic.predict_proba(xtest)
                pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                precision = calculate_precision_at_threshold(ytest, pred_labels)
                recall = calculate_recall_at_threshold(ytest, pred_labels)
                f1 = metrics.f1_score(ytest, pred_labels)
                auc = metrics.roc_auc_score(ytest, pred_labels)
                summary.loc[len(summary)] = ['Logistic Regression',
                    train_date, test_date, penalty, c, threshold, accuracy,
                    precision, recall, f1, auc]
    return summary




##############
#DECISION TREE
##############
def train_decision_tree(features, target, criterion='gini', splitter='best',
    max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight=None, random_state=100):
    '''
    Instantiate an object of the decision tree model class and train it
    on given features to predict a given target value for a new input observation.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      criterion (str): function to measure the quality of a split, either "gini"
        for impurity or "entropy" for information gain
      splitter (str): strategy used to choose split at each node, either "best"
        for best split or "random" for best random split
      max_depth (int): maximum depth of the tree. If none, nodes are expanded
        until all leaves pure or all leaves contain less than min_samples_split
      min_samples_split (int): minimum number of samples required to split a node
      min_samples_leaf (int): minimum number of samples required to be at a leaf node
      class_weight (dict or 'balanced'): if not given, all classes have weight 1.
        'balanced' mode uses the values of y to automatically adjust weights 
        inversely proportional to class frequencies in the input data
      random_state (int): random seed to initialize the model

    Returns:
      trained decision tree model object
    '''
    decision_tree = tree.DecisionTreeClassifier(
        criterion=criterion, splitter=splitter, max_depth=max_depth, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
        class_weight=class_weight, random_state=random_state)
    decision_tree.fit(features, target)
    return decision_tree


def decision_tree_loop(
    xtrain, ytrain, xtest, ytest, train_date, test_date, criteria, splitters, depths, thresholds):
    '''
    Loop through d-tree models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data
      criteria, splitters, depths (lists): lists of decision tree parameters
        ovr which to iterate
      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','criteria','splitter','max_depth',
        'threshold','accuracy','precision','recall','f1','auc'])
    for criterion in criteria:
        for splitter in splitters:
            for depth in depths:
                for threshold in thresholds:
                    dtree = train_decision_tree(xtrain, ytrain, criterion=criterion,
                        splitter=splitter, max_depth=depth)
                    pred_scores = dtree.predict_proba(xtest)
                    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                    accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                    precision = calculate_precision_at_threshold(ytest, pred_labels)
                    recall = calculate_recall_at_threshold(ytest, pred_labels)
                    f1 = metrics.f1_score(ytest, pred_labels)
                    auc = metrics.roc_auc_score(ytest, pred_labels)
                    summary.loc[len(summary)] = ['Decision Tree', train_date, test_date, 
                                                 criterion, splitter, depth, threshold, accuracy,
                                                 precision, recall, f1, auc]
    return summary




########################
#SUPPORT VECTOR MACHINES
########################
def train_svm(features, target, C=1.0, kernel='rbf', degree=3, gamma='auto',
    class_weight=None, max_iter=-1, probability=True, random_state=100):
    '''
    Instantiate an object of the support vector machines model class and train it
    on given features to predict a given target value for a new input observation.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      C (float):
      kernel (int):
      degree (int):
      gamma (str):
      class_weight (dict, 'balanced'):
      max_iter (int):
      decision_function_shape (str):
      random_state (int):

    Returns:
      trained SVM model object
    '''
    machine = svm.SVC(
        C=C, kernel=kernel, degree=degree, gamma=gamma, class_weight=class_weight,
        max_iter=max_iter, probability=probability, random_state=random_state)
    machine.fit(features, target)
    return machine


def svm_loop(
    xtrain, ytrain, xtest, ytest, train_date, test_date, cs, kernels, degrees, thresholds):
    '''
    Loop through SVM models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data
      cs, kernels, degrees (lists): lists of SVM parameters over which
        to iterate
      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','C','kernel','degree',
        'threshold','accuracy','precision','recall','f1','auc'])
    for c in cs:
        for kernel in kernels:
            for degree in degrees:
                for threshold in thresholds:
                    svm = train_svm(xtrain, ytrain, C=c, kernel=kernel, degree=degree)
                    pred_scores = svm.predict_proba(xtest)
                    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                    accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                    precision = calculate_precision_at_threshold(ytest, pred_labels)
                    recall = calculate_recall_at_threshold(ytest, pred_labels)
                    f1 = metrics.f1_score(ytest, pred_labels)
                    auc = metrics.roc_auc_score(ytest, pred_labels)
                    summary.loc[len(summary)] = ['Support Vector Machines', train_date, test_date, 
                                                 c, kernel, degree, threshold, accuracy,
                                                 precision, recall, f1, auc]
    return summary




########
#BAGGING
########
def train_bagging(features, target, base_est=None, n_est=10, max_samp=1.0, max_feat=1.0, 
            bootstrap=True, bootstrap_feat=False, random_state=1000):
    '''
    Instantiate an object of the bagging (boostrap aggregation) model class and train it
    on given features to predict a given target value for a new input observation.
    Uses default specifications for all base estimator options.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      base_est (name of base estimator class)
      n_est (int): number of base estimators in the ensemble
      max_samp (int): number of samples to draw from X to train each base estimator
      max_feat (int): number of features to draw from X to train each base estimator
      bootstrap (Bool, default True): whether samples are drawn with replacement
      boostrap_feat (Bool, default False): whether features are drawn with replacement
      random_state (int): random seed with which to instantiate the training

    Returns:
      trained bagging model object
    '''
    bagging = ensemble.BaggingClassifier(base_estimator=base_est, n_estimators=n_est,
        max_samples=max_samp, max_features=max_feat, bootstrap=bootstrap,
        bootstrap_features=bootstrap_feat, random_state=random_state)
    bagging.fit(features, target)
    return bagging


def bagging_loop(
        xtrain, ytrain, xtest, ytest, train_date, test_date, bases, estimators, thresholds):
    '''
    Loop through bagging models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data

      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','base','estimators',
        'threshold','accuracy','precision','recall','f1','auc'])
    for base in bases:
        for estimator in estimators:
            for threshold in thresholds:
                bag = train_bagging(xtrain, ytrain, 
                    base_est=base, n_est=estimator)
                pred_scores = bag.predict_proba(xtest)
                pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                precision = calculate_precision_at_threshold(ytest, pred_labels)
                recall = calculate_recall_at_threshold(ytest, pred_labels)
                f1 = metrics.f1_score(ytest, pred_labels)
                auc = metrics.roc_auc_score(ytest, pred_labels)
                summary.loc[len(summary)] = ['Bagging (LR)', train_date, test_date, base,
                                            estimator, threshold, accuracy,
                                            precision, recall, f1, auc]
    return summary




#########
#BOOSTING
#########
def train_boosting(features, target, loss='deviance', learning_rate=0.1, n_est=100, min_samples_split=2,
    min_samples_leaf=1, max_depth=3, random_state=1000, max_features=None, max_leaf_nodes=None):
    '''
    Instantiate an object of the boosting ensemble model class and train it
    on given features to predict a given target value for a new input observation.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      loss (str, default 'deviance'): loss function to be optimized,
        'deviance' refers to classification with probabilistic outputs
      learning_rate (float, default=0.1): learning rate shrinks the contribution
        of each tree by learning_rate
      n_est (int): number of trees to make up the forest
      criterion (str): function to measure the quality of a split, either "gini"
        for impurity or "entropy" for information gain
      max_depth (int): maximum depth of the tree. If none, nodes are expanded
        until all leaves pure or all leaves contain less than min_samples_split
      min_samples_split (int): minimum number of samples required to split a node
      min_samples_leaf (int): minimum number of samples required to be at a leaf node
      class_weight (dict or 'balanced'): if not given, all classes have weight 1.
        'balanced' mode uses the values of y to automatically adjust weights 
        inversely proportional to class frequencies in the input data
      random_state (int): random seed to initialize the model
      max_features (int, float or None, default None): number of features to consider
        when looking for the best split
      max_leaf_nodes (int or None, default None): grow trees with max_leaf_nodes
        in best-first fashion, defined as impurity reduction. None means unlimited
        leaf nodes

    Returns:
      trained boosting ensemble model object
    '''
    boosting = ensemble.GradientBoostingClassifier(loss=loss, learning_rate=learning_rate,
        n_estimators=n_est, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_depth=max_depth, random_state=random_state, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    boosting.fit(features, target)
    return boosting


def boosting_loop(
        xtrain, ytrain, xtest, ytest, train_date, test_date, losses, learning_rates, estimators, depths, thresholds):
    '''
    Loop through boosting models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data

      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','loss','learning_rate','estimators','max_depth',
        'threshold','accuracy','precision','recall','f1','auc'])
    for loss in losses:
        for learning_rate in learning_rates:
            for estimator in estimators:
                for depth in depths:
                    for threshold in thresholds:
                        boosting = train_boosting(xtrain, ytrain, loss=loss, learning_rate=learning_rate, 
                            n_est=estimator, max_depth=depth)
                        pred_scores = boosting.predict_proba(xtest)
                        pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                        accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                        precision = calculate_precision_at_threshold(ytest, pred_labels)
                        recall = calculate_recall_at_threshold(ytest, pred_labels)
                        f1 = metrics.f1_score(ytest, pred_labels)
                        auc = metrics.roc_auc_score(ytest, pred_labels)
                        summary.loc[len(summary)] = ['Boosting', train_date, test_date, loss, learning_rate,
                                                     estimator, depth, threshold, accuracy,
                                                     precision, recall, f1, auc]
    return summary




#########################
#RANDOM FOREST CLASSIFIER
#########################
def train_forest(features, target, n_est=100, criterion='gini', max_depth=None, min_samples_split=2,
    min_samples_leaf=1, class_weight=None, random_state=100):
    '''
    Instantiate an object of the random forest model class and train it
    on given features to predict a given target value for a new input observation.

    Inputs:
      features (pd df or list of series): pandas dataframe containing
        the features/predictors used to train
      target (pd df or series): pandas dataframe or series containing
        values of the class that is targeted for prediction
        (these are the training values of the target class)
      n_est (int): number of trees to make up the forest
      criterion (str): function to measure the quality of a split, either "gini"
        for impurity or "entropy" for information gain
      max_depth (int): maximum depth of the tree. If none, nodes are expanded
        until all leaves pure or all leaves contain less than min_samples_split
      min_samples_split (int): minimum number of samples required to split a node
      min_samples_leaf (int): minimum number of samples required to be at a leaf node
      class_weight (dict or 'balanced'): if not given, all classes have weight 1.
        'balanced' mode uses the values of y to automatically adjust weights 
        inversely proportional to class frequencies in the input data
      random_state (int): random seed to initialize the model

    Returns:
      trained random forest model object
    '''
    forest = ensemble.RandomForestClassifier(n_estimators=n_est, criterion=criterion,
        max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=100)
    forest.fit(features, target)
    return forest


def forest_loop(
    xtrain, ytrain, xtest, ytest, train_date, test_date, estimators, criteria, depths, thresholds):
    '''
    Loop through random forest models using the parameters specified and return
    a summary dataset of model specifications and evaluation metrics.

    Inputs:
      xtrain (pd df or series): training instances of predictors
      ytrain (pd df or series): training instances of target
      xtest (pd df or series): testing instances of predictors
      ytest (pd df or series): testing instances of target
      train_date (str): date range of training data
      test_dat (str): date range of testing data

      threshold (list of float): classification thresholds to iterate over

    Returns:
      summary (df): summary of all model specifications and
        evaluation metric values
    '''
    summary = pd.DataFrame(columns=[
        'model','train_date','test_date','estimators','criteria','max_depth',
        'threshold','accuracy','precision','recall','f1','auc'])
    for estimator in estimators:
        for criterion in criteria:
            for depth in depths:
                for threshold in thresholds:
                    forest = train_forest(xtrain, ytrain, n_est=estimator, 
                        criterion=criterion, max_depth=depth)
                    pred_scores = forest.predict_proba(xtest)
                    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
                    accuracy = calculate_accuracy_at_threshold(ytest, pred_labels)
                    precision = calculate_precision_at_threshold(ytest, pred_labels)
                    recall = calculate_recall_at_threshold(ytest, pred_labels)
                    f1 = metrics.f1_score(ytest, pred_labels)
                    auc = metrics.roc_auc_score(ytest, pred_labels)
                    summary.loc[len(summary)] = ['Random Forest', train_date, test_date, 
                                                 estimator, criterion, depth, threshold, accuracy,
                                                 precision, recall, f1, auc]
    return summary






###########
#EVALUATION
###########
def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Greens):
    '''
    Plot confusion matrix with true/false positives and true/false negatives
    to visualize prediction mistakes. Code adapted from: https://scikit-learn.org/
    stable/auto_examples/model_selection/plot_confusion_matrix.html.

    Inputs:
      y_true (list): list of *true* class labels (target variable test data)
      y_pred (list): list of *predicted* class labels (result of predict_proba
        from a trained model)
      normalize (Bool): if True, present confusion matrix figures as proportions.
        If False, present as integers (raw count).
      cmap (plt color): set the matplotlib color

    Returns:
      Plotted confusion matrix
    '''
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    cm = metrics.confusion_matrix(y_true, y_pred)
    #classes = classes[utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, title=title,
           ylabel="True label", xlabel="Predicted label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", 
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #return ax


def calculate_accuracy_at_threshold(true_labels, pred_labels):
    '''
    Calculates accuracy score for given true_labels and pred_labels

    Inputs:
      true_labels (list): list of testing data for target class variable
        (actual values)
      pred_labels (list): list of predicted values from feature
        testing data

    Returns:
      (float): accuracy score
    '''
    tn, fp, fn, tp = metrics.confusion_matrix(true_labels, pred_labels).ravel()
    return 1.0 * (tp + tn) / (tn + fp + fn + tp)


def calculate_precision_at_threshold(true_labels, pred_labels):
    '''
    Calculates precision for given true_labels and pred_labels

    Inputs:
      true_labels (list): list of testing data for target class variable
        (actual values)
      pred_labels (list): list of predicted values from feature
        testing data

    Returns:
      (float): precision (proportion of identified positives that
        were in fact positives)
    '''
    _, fp, _, tp = metrics.confusion_matrix(true_labels, pred_labels).ravel()
    return 1.0 * tp / (fp + tp)


def calculate_recall_at_threshold(true_labels, pred_labels):
    '''
    Calculates recall for given true_labels and pred_labels

    Inputs:
      true_labels (list): list of testing data for target class variable
        (actual values)
      pred_labels (list): list of predicted values from feature
        testing data

    Returns:
      (float): recall (proportion of all true positives that 
        were identified as such)
    '''
    _, _, fn, tp = metrics.confusion_matrix(true_labels, pred_labels).ravel()
    return 1.0 * tp / (fn + tp)


def calculate_roc_curve(model, xtest, ytest, threshold):
    '''
    Calculate the false positive rate (fpr) and true positive rate (tpr)
    for plotting of the ROC curve, as well as the total area under
    the curve (roc_auc)

    Inputs:
      model: trained model
      xtest (df): dataframe of predictors
      ytest (df/list): list of observed test points
      threshold (float): threshold at which to cut off predicted positives

    Returns:
      fpr (array): false positives (x-points of ROC)
      tpr (array): true positives (y-points of ROC)
      roc_auc (float): area under the curve 
    '''
    pred_scores = model.predict_proba(xtest)
    pred_labels = [1 if x[1] > threshold else 0 for x in pred_scores]
    fpr, tpr, _ = metrics.roc_curve(ytest, pred_labels)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, auc_roc, color, title):
    '''
    Plot ROC curve on a matplotlib plot

    Inputs:
      fpr (array): false positives (x-points of ROC)
      tpr (array): true positives (y-points of ROC)
      roc_auc (float): area under the curve 
      color (str): color for ROC curve line
      title (str): plot title

    Returns:
      None, displays plot in-place
    '''
    plt.figure()
    plt.plot(fpr, tpr, color=color, marker='.',
        lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


#def plot_precision_recall_curve(true_labels, pred_labels, color, title):
#    '''
#    '''
#    precision, recall, _ = metrics.precision_recall_curve(true_labels, pred_labels)
#    step_kwargs = ({'step':'post'}
#        if 'step' in signature(plt.fill_between).parameters)
#    plt.step(recall, precision, color=color, alpha=0.2, where='post')
#    plt.fill_between(recall, precision, alpha=0.2, color=color, **step_kwargs)
#    plt.xlabel("Recall")
#    plt.ylabel("Precision")
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title(title)
#    plt.show()

