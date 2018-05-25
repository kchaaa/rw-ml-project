# Data Exploration of Red Wine Data Set
# Source: https://archive.ics.uci.edu/ml/rw_datasets/wine+quality

# Import Libraries ----
# Data Processing
import pandas as pd

# Math
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

# Functions ----
def plot_hist(data, alpha, title, xlabel, ylabel) :
   bins = np.linspace(math.ceil(min(data)),
                   math.floor(max(data)),
                   20)
   plt.hist(data, bins=bins, alpha=alpha, edgecolor='black')

   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)

   plt.show()

# Read in data ----
file_path = '..\\data\\winequality-red.csv'
rw_data = pd.read_csv(file_path, sep=';')
# Change column names
rw_data.columns = rw_data.columns.str.replace(" ", "_")

# View Data ----
# view summary of rw_data 
rw_data.info()
# View Statistical Summary of rw_data
rw_data.describe()
# view First 15 Rows of rw_data 
rw_data.head(15)

# Single Distribution ----
# fixed_acidity
rw_data['fixed_acidity'].describe()
plot_hist(rw_data['fixed_acidity'], 0.5, 'Fixed Acidity Distribution', 'Fixed Acidity', 'Count')
# volatile_acidity
rw_data['volatile_acidity'].describe()
plot_hist(rw_data['volatile_acidity'], 0.5, 'Volatile Acidity Distribution', 'Volatile Acidity', 'Count')
# citric_acid
rw_data['citric_acid'].describe()
plot_hist(rw_data['citric_acid'], 0.5, 'Citric Acid Distribution', 'Citric Acid', 'Count')
# residual_sugar
rw_data['residual_sugar'].describe()
plot_hist(rw_data['residual_sugar'], 0.5, 'Residual Sugar Distribution', 'Residual Sugar', 'Count')
# chlorides
rw_data['chlorides'].describe()
plot_hist(rw_data['chlorides'], 0.5, 'Chlorides Distribution', 'Chlorides', 'Count')
# free_sulfur_dioxide
rw_data['free_sulfur_dioxide'].describe()
plot_hist(rw_data['free_sulfur_dioxide'], 0.5, 'Free Sulfur Dioxide Distribution', 'Free Sulfur Dioxide', 'Count')
# total_sulfur_dioxide
rw_data['total_sulfur_dioxide'].describe()
plot_hist(rw_data['total_sulfur_dioxide'], 0.5, 'Total Sulfur Dioxide Distribution', 'Total Sulfur Dioxide', 'Count')
# density
rw_data['density'].describe()
plot_hist(rw_data['density'], 0.5, 'Fixed Acidity Distribution', 'Fixed Acidity', 'Count')
# pH
rw_data['pH'].describe()
plot_hist(rw_data['pH'], 0.5, 'pH Distribution', 'pH', 'Count')
# sulphates
rw_data['sulphates'].describe()
plot_hist(rw_data['sulphates'], 0.5, 'Sulphates Distribution', 'Sulphates', 'Count')
# alcohol
rw_data['alcohol'].describe()
plot_hist(rw_data['alcohol'], 0.5, 'Alcohol Distribution', 'Alcohol', 'Count')

# Pair Plots ----
# Scatterplot Matrix
sns.pairplot(rw_data, hue = 'quality', diag_kind = 'kde')

# Correlation Matrix
correlation = rw_data.corr()
sns.heatmap(correlation,
			annot=True,
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
