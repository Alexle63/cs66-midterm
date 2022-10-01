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


"""              LINEAR REGRESSION START                """
print("\n\n LINEAR REGRESSION")
plt.figure(1)
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
     []]
# Make a linear regression for every feature
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
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Plot outputs

    # print("Metric: %s" % (names[i]))
    # # The coefficients
    # print("Coefficients: ", regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f\n" % r2_score(diabetes_y_test, diabetes_y_pred))

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

print('\n'.join([''.join(['{:^20}'.format(item) for item in row]) 
    for row in A]))

plt.suptitle("Linear Regression", fontsize=14)
"""              LINEAR REGRESSION END                  """









"""           KERNEL RIDGE REGRESSION START             """
print("\n\n KERNEL RIDGE REGRESSION")
plt.figure(2)
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
     []]
# Make a regression for every feature
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
    regr = linear_model.RidgeCV()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Plot outputs

    # print("Metric: %s" % (names[i]))
    # # The coefficients
    # print("Coefficients: ", regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f\n" % r2_score(diabetes_y_test, diabetes_y_pred))

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

print('\n'.join([''.join(['{:^20}'.format(item) for item in row]) 
    for row in A]))

plt.suptitle("Kernel Ridge Regression", fontsize=14)
"""            KERNEL RIDGE REGRESSION END              """







"""             SGD REGRESSOR START               """
print("\n\n SGD REGRESSOR")
plt.figure(3)
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
     []]
# Make a regression for every feature
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
    regr = linear_model.SGDRegressor(max_iter=100000)

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Plot outputs

    # print("Metric: %s" % (names[i]))
    # # The coefficients
    # print("Coefficients: ", regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f\n" % r2_score(diabetes_y_test, diabetes_y_pred))

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

print('\n'.join([''.join(['{:^20}'.format(item) for item in row]) 
    for row in A]))

plt.suptitle("SGD Regression", fontsize=14)
"""              SGD REGRESSOR END                """







"""             LASSO REGRESSOR START               """
print("\n\n LASSO REGRESSOR")
plt.figure(4)
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
     []]
# Make a regression for every feature
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
    regr = linear_model.LassoCV()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Plot outputs

    # print("Metric: %s" % (names[i]))
    # # The coefficients
    # print("Coefficients: ", regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f\n" % r2_score(diabetes_y_test, diabetes_y_pred))

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

print('\n'.join([''.join(['{:^20}'.format(item) for item in row]) 
    for row in A]))

plt.suptitle("LASSO Regression", fontsize=14)
"""              LASSO REGRESSOR END                """



print("\n\nMean squared error: Lower is better")
print("Coefficient of determination: Higher is better\n")
plt.show()