# Data Preprocessing of Red Wine Data Set
# Source: https://archive.ics.uci.edu/ml/rw_datasets/wine+quality

# Import Libraries ----
# Data Processing
import pandas as pd

# Read in data ----
file_path = 'C:\\Users\\phato_000\\Documents\\ChaKProjects\\rw_project\\data\\winequality-red.csv'
rw_data = pd.read_csv(file_path, sep=';')
# Change column names
rw_data.columns = rw_data.columns.str.replace(" ", "_")

# Conversion Features: quality ----
# View Numerical Distribution
rw_data['quality'].value_counts()
# quality >= 7 : good (1) and quality < 7  : bad (0)
rw_data['quality'] = rw_data['quality'].mask(rw_data['quality'] < 7, 0)
rw_data['quality'] = rw_data['quality'].mask(rw_data['quality'] >= 7, 1)

# Creating Categories: fixed_acidity ----
# Decile Binning
# 0 : D1	1 : D2	2 : D3	3 : D4	4 : D5	5 : D6	6 : D7	7 : D8	8 : D9	9 : D10
rw_data['fixed_acidity'] = pd.qcut(rw_data['fixed_acidity'], q=10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], precision=1.0)

# Creating Categories: volatile_acidity ----
# Decile Binning
# 0 : D1	1 : D2	2 : D3	3 : D4	4 : D5	5 : D6	6 : D7	7 : D8	8 : D9	9 : D10
rw_data['volatile_acidity'] = pd.qcut(rw_data['volatile_acidity'], q=10,
									  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], precision=1.0)

# Creating Categories: citric_acid ----
# Quantile Binning
# 0 : Q1		1 : Q2		2 : Q3		3 : Q4
rw_data['citric_acid'] = pd.qcut(rw_data['citric_acid'], q=4, labels=[0, 1, 2, 3], precision=1.0)

# Creating Categories: residual_sugar ----
# Divide by those between avg alcohol levels (12-15) and those that aren't
# Extra Dry(<1 g/L) :: 0, Dry(1-10 g/L) :  1, Off-dry(10-35 g/L) : 2
rw_data['residual_sugar'] = rw_data['residual_sugar'].mask(rw_data['residual_sugar'] < 1, 0)
rw_data['residual_sugar'] = rw_data['residual_sugar'].mask((rw_data['residual_sugar'] >= 1) & (rw_data['residual_sugar'] <= 10), 1)
rw_data['residual_sugar'] = rw_data['residual_sugar'].mask(rw_data['residual_sugar'] > 10, 2)

# Creating Categories: chlorides ----
# Quantile Binning
# 0 : Q1		1 : Q2		2 : Q3		3 : Q4
rw_data['chlorides'] = pd.qcut(rw_data['chlorides'], q=4, labels=[0, 1, 2, 3], precision=1.0)

# Creating Categories: free_sulfur_dioxide ----
# Divide depending if it follows the rule
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 92), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.8) & (rw_data['pH'] < 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 78) & (rw_data['free_sulfur_dioxide'] < 92), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.7) & (rw_data['pH'] < 3.8) &
																(rw_data['free_sulfur_dioxide'] >= 64) & (rw_data['free_sulfur_dioxide'] < 78), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.8) & (rw_data['pH'] < 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 50) & (rw_data['free_sulfur_dioxide'] < 64), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.7) & (rw_data['pH'] < 3.8) &
																(rw_data['free_sulfur_dioxide'] >= 40) & (rw_data['free_sulfur_dioxide'] < 50), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.8) & (rw_data['pH'] < 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 31) & (rw_data['free_sulfur_dioxide'] < 40), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.7) & (rw_data['pH'] < 3.8) &
																(rw_data['free_sulfur_dioxide'] >= 27) & (rw_data['free_sulfur_dioxide'] < 31), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.8) & (rw_data['pH'] < 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 21) & (rw_data['free_sulfur_dioxide'] < 27), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.7) & (rw_data['pH'] < 3.8) &
																(rw_data['free_sulfur_dioxide'] >= 17) & (rw_data['free_sulfur_dioxide'] < 21), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] >= 3.8) & (rw_data['pH'] < 3.9) &
																(rw_data['free_sulfur_dioxide'] >= 12) & (rw_data['free_sulfur_dioxide'] < 17), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask((rw_data['pH'] < 3.0) &
																(rw_data['free_sulfur_dioxide'] < 12), 1)
rw_data['free_sulfur_dioxide'] = rw_data['free_sulfur_dioxide'].mask(rw_data['free_sulfur_dioxide'] != 1, 0)

# Creating Categories: total_sulfur_dioxide ----
rw_data['total_sulfur_dioxide'] = rw_data['total_sulfur_dioxide'].mask(rw_data['free_sulfur_dioxide'] == 0, 0)
rw_data['total_sulfur_dioxide'] = rw_data['total_sulfur_dioxide'].mask(rw_data['free_sulfur_dioxide'] == 1, 1)

# Creating Categories: density ----
# Quantile Binning
# 0 : Q1		1 : Q2		2 : Q3		3 : Q4
rw_data['density'] = pd.qcut(rw_data['density'], q=4, labels=[0, 1, 2, 3], precision=1.0)

# Creating Categories: pH ----
# Divide by those between  and those that aren't
rw_data['pH'] = rw_data['pH'].mask(rw_data['pH'] > 3.6, 2)
rw_data['pH'] = rw_data['pH'].mask(rw_data['pH'] >= 3.3, 1)
rw_data['pH'] = rw_data['pH'].mask(rw_data['pH'] < 3.3, 0)

# Creating Categories: sulphates ----
# Quantile Binning
# 0 : Q1		1 : Q2		2 : Q3		3 : Q4
rw_data['sulphates'] = pd.qcut(rw_data['sulphates'], q=4, labels=[0, 1, 2, 3], precision=1.0)

# Creating Categories: alcohol ----
# Bin based on alcohol content
rw_data['alcohol'] = rw_data['alcohol'].mask(rw_data['alcohol'] < 10, 0)
rw_data['alcohol'] = rw_data['alcohol'].mask((rw_data['alcohol'] >= 10) & (rw_data['alcohol'] < 11.5), 1)
rw_data['alcohol'] = rw_data['alcohol'].mask((rw_data['alcohol'] >= 11.5) & (rw_data['alcohol'] < 13.5), 2)
rw_data['alcohol'] = rw_data['alcohol'].mask(rw_data['alcohol'] >= 13.5, 3)

# Write into csv for models ----
file_path2='C:\\Users\\phato_000\\Documents\\ChaKProjects\\rw_project\\processed\\redwine.csv'
rw_data.to_csv(file_path2, index=False)