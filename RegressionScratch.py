import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# data - training set of features - (m x n) matrix.
# y - a vector of expected output values - (m x 1) vector.
# theta - current model parameters - (n x 1) vector.
# alpha - learning rate, the size of gradient step we need to take on each iteration.
# lambda - regularization parameter.
# numb_iterations - number of iterations we will take for gradient descent.

lamb = 0
alpha = 0.1
num = 50

def feature_normalize(df):
    mean_mapping = {}
    std_mapping = {}
    for col in df.columns:
        mean_mapping[col] = df[col].mean()
        std_mapping[col] = df[col].std()
    for col in df.columns:
        df[col] = df[col] - mean_mapping[col]
        df[col] = df[col].div(std_mapping[col])
    return df


# theta is the parameters for the regression
def hypothesis(df, theta):
    predictions = df.dot(theta)
    return predictions


def gradient_step(df, y, theta, alpha, lamb):
    m = df.shape[0]
    predictions = hypothesis(df, theta)
    difference = predictions - y
    regularization_param = 1 - alpha * lamb / m
    theta = theta * regularization_param - alpha * (1 / m) * (difference.transpose().dot(df)).transpose()
    # Cost function is the sum of squares of the difference from prediction vs actual
    # Calculate partial derivative of cost function on each theta
    # Subtract from theta to approach minima
    # We should NOT regularize the parameter theta_zero.
    theta[0] = theta[0] - alpha * (1 / m) * (difference.transpose().dot(df.iloc[:, 0]))
    return theta
    # Result is a series


# theta_res = gradient_step(data, result, initial_theta, alpha, lamb)


def cost_function(df, y, theta, lamb):
    m = df.shape[0]
    differences = hypothesis(df, theta) - y

    # Calculate regularization parameter.
    # Remember that we should not regularize the parameter theta_zero.
    theta_cut = theta.drop(theta.index[0])
    regularization_param = lamb * theta_cut.dot(theta_cut)

    # Calculate current predictions cost
    cost = (1 / 2 * m) * (differences.transpose().dot(differences) + regularization_param)
    return cost


def gradient_descent(df, y, theta, alpha, lamb, num_iterations):
    # m = df.shape[0]
    J_history = []
    for iteration in range(num_iterations):
        theta = gradient_step(df, y, theta, alpha, lamb)
        J_history.append(cost_function(df, y, theta, lamb))
    return theta


def linear_regr_train(test_data, df, y, alpha, lamb, num_iterations):
    # m = df.shape[0]
    n = len(df.columns)
    df_norm = feature_normalize(df)
    df_norm.insert(0, 'ones', 1, allow_duplicates=True)
    initial_theta = np.zeros(n + 1)
    test_data_norm = feature_normalize(test_data)
    test_data_norm.insert(0, 'ones', 1, allow_duplicates=True)
    y_test = hypothesis(test_data_norm, gradient_descent(df_norm, y, initial_theta, alpha, lamb, num_iterations))
    print(y_test)