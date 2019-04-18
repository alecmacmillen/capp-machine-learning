'''
CAPP30254 S'19: Assignment 2
Machine Learning Pipeline

Alec MacMillen
'''

import sys
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
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


def plot_scatter(dataframe, x, y):
    '''
    Create a scatter plot using matplotlib.pyplot.

    Inputs: dataframe (pandas df)
      x (str): horizontal-axis var, column from dataframe
      y (str): vertical-axis var, column from dataframe

    Returns: nothing, prints graphic inline
    '''
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


def create_splits(
    dataframe, selected_features, target_class, test_size, random_state=1000):
    '''
    Create train-test split from master dataset for use in learning and testing
    ML model.

    Inputs:
      dataframe (pandas df)
      selected_features (list of features)
      target_class (str, variable for which to predict outcome)

    Returns:
      Train and test df's for feature (x) and classification label (y) variables
    '''
    x = dataframe[selected_features]
    y = dataframe[target_class]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def fit_decision_tree(x_train, y_train):
    '''
    Fit a decision tree model using training data from feature (x) and
    class label (y) variables.

    Inputs: x_train, y_train (pandas dfs): containing training observations

    Returns: dec_tree (sklearn.DecisionTreeClassifier obj): trained model
    '''
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(x_train, y_train)
    return dec_tree


def predict_dec_tree_scores(dec_tree, x_test):
    '''
    Predict scores for testing data using trained decision tree model
    and test data for features (predictors/x data)

    Inputs: dec_tree (sklearn DecisionTreeClassifier obj)
      x_test (pandas df): feature information from test data

    Returns: predicted_scores_test (series): classification judgments
      for test data (predicted probability that observation will be 
      a member of the label=1 class)
    '''
    predicted_scores_test = dec_tree.predict_proba(x_test)[:,1]
    return predicted_scores_test


def plot_pst(predicted_scores_test):
    '''
    Plot predicted scores test as a histogram

    Inputs:
      predicted_scores_test (series), the result of running the 
        predict_dec_tree_scores function

    Returns:
      plots histogram in-place
    '''
    plt.hist(predicted_scores_test)
    plt.show()


def calculate_accuracy(predicted_scores_test, y_test, threshold):
    '''
    Calculate accuracy of learned ML model using predicted_scores_test
    comparing against y_test observations using "threshold" level to assign
    labels to different probability values.

    Inputs:
      predicted_scores_test (series), the result of running the 
        predict_dec_tree_scores function of a dec_tree model on test data
      y_test (pandas df): actual class label values for testing data
      threshold (float): threshold level above/below which to assign values
        from the predicted_scores_test to a class label of 1 or 0

    Returns:
      (float): the overall accuracy of the trained model's predictions.
    '''
    calc_threshold = lambda x,y: 0 if x < y else 1
    predicted_test = np.array([calc_threshold(
        score, threshold) for score in predicted_scores_test])
    test_acc = accuracy(predicted_test, y_test)
    return test_acc
