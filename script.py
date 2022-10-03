# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
# COMPLETE

# https://scikit-learn.org/stable/modules/kernel_ridge.html

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, ensemble
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
X, diabetes_y = datasets.load_diabetes(return_X_y=True)
names = datasets.load_diabetes().feature_names


# title:  Title of the used regression
# method: The function call of the regression
# num:    The number of the model used
def runModel(title, method, num):
    print("\n\n " + title.upper())
    plt.figure(num)
    A = [['Metric', 'Coefficients', 'Mean Squared Error', 'Coeff. of Det.'],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        ['Averages']]
        
    for i in range(0,10):
        # Use only one feature
        diabetes_X = X[:, np.newaxis, i]

        # Split the data into training/testing sets
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]

        # Split the targets into training/testing sets
        diabetes_y_train = diabetes_y[:-20]
        diabetes_y_test = diabetes_y[-20:]

        # Create linear regression object
        regr = method

        # Train the model using the training sets
        regr.fit(diabetes_X_train, diabetes_y_train)

        # Make predictions using the testing set
        diabetes_y_pred = regr.predict(diabetes_X_test)

        # Plot outputs

        A[i+1].append(str(names[i]))
        A[i+1].append(str(round(regr.coef_[0],2)))
        A[i+1].append(str(round(mean_squared_error(diabetes_y_test, diabetes_y_pred),2)))
        A[i+1].append(str(round(r2_score(diabetes_y_test, diabetes_y_pred),2)))

        plt.subplot(2,5,i+1)
        plt.title("Metric: %s" % (names[i]))
        plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
        plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

        plt.xticks(())
        plt.yticks(())
    mean_co = 0
    mean_squared = 0
    mean_co_det = 0
    for j in range(0,10):
        mean_co += float(A[j+1][1])
        mean_squared += float(A[j+1][2])
        mean_co_det += float(A[j+1][3])
    A[-1].append(round(mean_co/10, 2))
    A[-1].append(round(mean_squared/10, 2))
    A[-1].append(round(mean_co_det/10, 2))
    print('\n'.join([''.join(['{:^20}'.format(item) for item in row]) 
        for row in A]))

    plt.suptitle(title, fontsize=14)


runModel("Linear Regression", linear_model.LinearRegression(), 1)
runModel("Kernel Ridge Regression", linear_model.RidgeCV(), 2)
runModel("SGD Regressor", linear_model.SGDRegressor(max_iter=100000), 3)
runModel("LASSO Regression", linear_model.LassoCV(), 4)

print("\n\nCoefficients: Higher is better")
print("Mean squared error: Lower is better")
print("Coefficient of determination: Higher is better\n")
plt.show()