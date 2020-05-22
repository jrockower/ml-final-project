'''
Helper functions for Classification
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, \
    mean_absolute_error, r2_score
import datetime


def get_summary_stats(df):
    '''
    Prints out general summary statistics on a Pandas DataFrame

    Inputs:
        df: a Pandas DataFrame
        corr (bool): an optional parameter to print out correlations between
                     variables

    Returns:
        Prints summary statistics
    '''
    print(df.describe())
    print(df.corr())


def split_data(df, random_state, test_size=0.2):
    '''
    Split a DataFrame into Training and Testing using Scikit Learn

    Inputs:
        df: a Pandas DataFrame
        random_state (int): a random seed
        test_size (float): the percentage of the data for the testing data

    Returns:
        a tuple consisting of training and testing data
    '''

    return train_test_split(df, test_size=test_size, random_state=random_state)


def find_outliers(df, col, min_value, max_value):
    '''
    Identify the outliers in a particular column, e.g. to find negative income values.
    Inputs:
        df: a dataframe
        col: the column to inspect
        min_value: the lowest reasonable value for the column
        max_value: the highest reasonable value for the column
    Returns:
        the number of records falling outside the specified bounds, what those values are,
        and those records
    '''
    rv = df[(df[col] < min_value) | (df[col] > max_value)]
    print("Number of outliers found: {}".format(len(rv)))
    print("Outlier values found: {}".format(rv[col].unique()))

    return rv


def impute_missing(train_df, test_df, num_cols):
    '''
    Impute missing values of continuous variables using the median value from
    the training data

    Inputs:
        train_df: the training DataFrame
        test_df: the testing DataFrame
        num_cols (list): a list of column names to impute missing values on

    Returns:
        the DataFrame with imputed values
    '''

    for col in num_cols:
        med = train_df[col].median()
        if train_df[col].isnull().values.any():
            print('Imputing {} missing values with median {}'.format(col, med))
            train_df.loc[:, col] = train_df[col].fillna(med)
            test_df.loc[:, col] = test_df[col].fillna(med)

    return train_df, test_df


def normalize(train, test, atts):
    '''
    Normalize continuous variables based on the training data
    Input:
        train (pandas dataframe): dataframe of features for train observations
        test (pandas dataframe): dataframe of features for test observations
        atts (list of strings): list of attribute names to normalize
    Return:
        train_x (pandas dataframe): dataframe of features for train observations
        test_x (pandas dataframe): dataframe of features for test observations
    '''
    scaler = StandardScaler()
    train[atts] = scaler.fit_transform(train[atts])
    test[atts] = scaler.transform(test[atts])
    return train, test


def hot_encode(df, cols):
    '''
    Perform one-hot encoding of categorical variables

    Inputs:
        df: a Pandas DataFrame
        cols (list): a list of categorical columns on which to perform
                     one-hot encoding

    Returns:
        a DataFrame with one-hot encoding performed
    '''

    return pd.get_dummies(df, columns=cols)


def discretize(df, col, bins):
    '''
    Discretize a variable into a specified number of bins

    Inputs:
        df: a Pandas DataFrame
        col (str): the name of the column to discretize
        bins (int): the number of bins to split along

    Returns:
        A series of discretized values
    '''

    return pd.cut(df[col], bins)


def grid_search(train_f, train_t, test_f, test_t, models, grid):
    '''
    Build and evaluate classifiers for input models and grid

    Inputs:
        train_f (df): the training features
        train_t (series): the training target
        test_f (df): the testing features
        test_t (series): the testing target
        models (dict): a dictionary of models
        grid (dict): a dictionary of parameters for grid search

    Returns:
    '''

    # Begin timer
    start = datetime.datetime.now()

    # Initialize results data frame
    results = pd.DataFrame(columns=['model', 'parameters', 'accuracy_score',
                                    'mean_squared_error',
                                    'mean_absolute_error'])

    # Loop over models
    for model_key in models.keys():

        # Loop over parameters
        for params in grid[model_key]:
            print("Training model:", model_key, "|", params)

            # Create model
            model = models[model_key]
            model.set_params(**params)

            # Fit model on training set
            model.fit(train_f, train_t)

            # Predict on testing set
            target_predict = model.predict(test_f)

            # Evaluate predictions
            evaluation = evaluate_classifiers(model_key, str(params), test_t,
                                              target_predict)

            # Store results in your results data frame
            results = results.append(evaluation, ignore_index=True)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return results


def evaluate_classifiers(model_key, params, test_t, test_p):
    '''
    Return the accuracy score, MSE, MAE, and R^2 for a regression

    Inputs:
        model:
        params:
        test_t: the test target
        test_p: the test prediction

    Returns:
        a tuple of evaluation metrics
    '''

    acc = accuracy_score(test_t, test_p)
    mse = mean_squared_error(test_t, test_p)
    mae = mean_absolute_error(test_t, test_p)

    return {'model': model_key, 'parameters': params,
            'accuracy_score': acc, 'mean_squared_error': mse,
            'mean_absolute_error': mae}


def top_pred(model, pred_names, num_preds=5):
    '''
    Determine the most important features in a logistic regression model.
    Inputs:
        model: the logistic regression model
        pred_names: (list-like) the names of the predictors
        num_preds: how many predictors to include
    returns:
        the top predictors
    '''
    #code source: adapted from a suggestion on
    #https://stackoverflow.com/questions/43576614/logistic-regression-how-to-find-top-three-feature-that-have-highest-weights
    coefs = model.coef_[0]
    top = np.argpartition(coefs, -1 * num_preds)[-1* num_preds:]

    return pred_names[top]

def adj_r2(r2, n, p):
    comp_r2 = 1 - r2
    num = n - 1
    denom = n - p - 1

    return (1 - (comp_r2 * (num/denom)))
