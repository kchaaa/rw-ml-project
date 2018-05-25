# Classification of Red Wine Data
# Source: https://archive.ics.uci.edu/ml/rw_datasets/wine+quality

# Libraries
# Math
import numpy as np

# Data Processing
import pandas as pd
from sklearn.model_selection import train_test_split

# Data Visualization
from matplotlib import pyplot as plt

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Optimization
from sklearn.model_selection import GridSearchCV, cross_val_score

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Read in data ----
file_path = 'C:\\Users\\phato_000\\Documents\\ChaKProjects\\rw_project\\processed\\redwine.csv'
rw_data = pd.read_csv(file_path)

# Prediction Target
Y = rw_data["quality"]

# Predictors
X = rw_data.drop("quality", axis=1)

# Split into training and validation data sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=0)

# Models ----
# Gaussian Naive Bayes
# Define Model
naive_bayes = GaussianNB()
# Score
naive_bayes_scores = cross_val_score(naive_bayes, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Naive Bayes Cross-validation Scores:", naive_bayes_scores)
print("Mean:", naive_bayes_scores.mean())
print("Standard Deviation:", naive_bayes_scores.std())

# Logistic Regression
# Define Model
log_reg = LogisticRegression()
# Score
log_reg_scores = cross_val_score(log_reg, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Logistic Regression Cross-validation Scores:", log_reg_scores)
print("Mean:", log_reg_scores.mean())
print("Standard Deviation:", log_reg_scores.std())

# Linear Discriminant Analysis
# Define Model
lin_da = LinearDiscriminantAnalysis()
# Score
lin_da_scores = cross_val_score(lin_da, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Linear Discriminant Analysis Cross-validation Scores:", lin_da_scores)
print("Mean:", lin_da_scores.mean())
print("Standard Deviation:", lin_da_scores.std())

# Support Vector Machines
# Define Model
svm = SVC()
# Score
svm_scores = cross_val_score(svm, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Support Vector Machines Cross-validation Scores:", svm_scores)
print("Mean:", svm_scores.mean())
print("Standard Deviation:", svm_scores.std())

# KNN
# Define Model
knn = KNeighborsClassifier(n_neighbors=3)
# Score
knn_scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("KNN Cross-validation Scores:", knn_scores)
print("Mean:", knn_scores.mean())
print("Standard Deviation:", knn_scores.std())

# Decision Tree
# Define Model
dec_tree = DecisionTreeClassifier()
# Score
dec_tree_scores = cross_val_score(dec_tree, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Decision Tree Cross-validation Scores:", dec_tree_scores)
print("Mean:", dec_tree_scores.mean())
print("Standard Deviation:", dec_tree_scores.std())

# Random Forest
# Define Model
rand_forest = RandomForestClassifier(n_estimators=100)
# Score
rand_forest_scores = cross_val_score(rand_forest, X_train, Y_train, cv=10, scoring="accuracy")
# Print
print("Random Forest Cross-validation Scores:", rand_forest_scores)
print("Mean:", rand_forest_scores.mean())
print("Standard Deviation:", rand_forest_scores.std())

# Results
# Create Table Showing Best One
ml_scores = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression', 'Linear Discriminant Analysis', 'Support Vector Machines',
              'K-Nearest Neighbors', 'Random Forest', 'Decision Tree'],
    'Avg Score': [naive_bayes_scores.mean(), log_reg_scores.mean(), lin_da_scores.mean(), svm_scores.mean(),
                  knn_scores.mean(), rand_forest_scores.mean(),  dec_tree_scores.mean()],
    'SD': [naive_bayes_scores.std(), log_reg_scores.std(), lin_da_scores.std(), svm_scores.std(), knn_scores.std(),
           rand_forest_scores.std(),  dec_tree_scores.std()]
    })
# Sort by Highest Score => Lowest Score
ml_scores_df = ml_scores.sort_values(by=['Avg Score', 'Model', 'SD'], ascending=False)
ml_scores_df = ml_scores_df.set_index('Model')
print(ml_scores_df)

# Best Model: Random Forest ----
# Random Forest
rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(X_train, Y_train)
Y_pred = rand_forest.predict(X_val)
acc_rand_forest = accuracy_score(Y_val, Y_pred) * 100
print("Random Forest Accuracy Score: ", round(acc_rand_forest, 2,), "%")

# Random Forest (Improved-Drop Feature)
importance = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rand_forest.feature_importances_,3)})
importance = importance.sort_values('importance',ascending=False).set_index('feature')

importance.head(15)
importance.plot.bar()

# Drop Less Important Feature
rw_data = rw_data.drop('residual_sugar', axis=1)
rw_data = rw_data.drop('free_sulfur_dioxide', axis=1)
rw_data = rw_data.drop('total_sulfur_dioxide', axis=1)
rw_data = rw_data.drop('pH', axis=1)

# Prediction Target
Y = rw_data["quality"]
# Predictors
X = rw_data.drop("quality", axis=1)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=0)

rand_forest = RandomForestClassifier(n_estimators=100, oob_score=True)
rand_forest.fit(X_train, Y_train)
Y_pred = rand_forest.predict(X_val)
acc_rand_forest = accuracy_score(Y_val, Y_pred) * 100
print("Random Forest Accuracy Score: ", round(acc_rand_forest, 2,), "%")

# Out-of-bag Samples
print("oob score:", round(rand_forest.oob_score_, 3)*100, "%")

# Hyperparameter Tuning
# Create the parameter grid based on the results of random search
param_grid = {
    "criterion" : ["gini", "entropy"],
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [100, 200, 300, 1000],
    'max_depth': [80, 90, 100, 110],
    'min_samples_leaf': [1, 5, 10, 25, 50, 70],
    'min_samples_split': [2, 4, 10, 12, 16, 18, 25, 35]
}

# Use GridSearchCV() to evaluate which are the best hyperparameters to use)
rf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
clf.fit(X_train, Y_train)
clf.best_params_

# Random Forest (with best hyperparameters)
rand_forest = RandomForestClassifier(criterion="entropy",
                                     max_depth=80,
                                     max_features='auto',
                                     min_samples_leaf=1,
                                     min_samples_split=2,
                                     n_estimators=200,
                                     oob_score=True,
                                     random_state=1,
                                     n_jobs=-1)
rand_forest.fit(X_train, Y_train)
Y_pred= rand_forest.predict(X_val)
rand_forest.score(X_train, Y_train)
print("oob score:", round(rand_forest.oob_score_, 3)*100, "%")

# Confusion Matrix
confusion_matrix(Y_val, Y_pred)

# Classification Report
class_report = classification_report(Y_val, Y_pred)
print(class_report)

# ROC AUC Curve
# Getting the probabilities of our predictions
y_scores = rand_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

# Calculate False Positive and True Positive
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

# Plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None) :
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

# Score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)