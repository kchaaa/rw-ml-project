# Red Wine Machine Learning Project
A machine learning project focused on the predicting the quality of red wine. Data from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). You can view the report [here](https://kchaaa.github.io/red-wine/).

## Installation
### Download Data
* Clone this repository into your computer
* Get into the folder using `cd redwine-ml-project`
* Run `mkdir data`
* Run `mkdir processed` (for later)
* Get into the data folder using `cd data`
* Download the data directly from [here](https://archive.ics.uci.edu/ml/datasets/wine+quality)
	* Click *Data Folder*
	* Click *winequality-red.csv* 
	* Move .csv file to data folder
* Switch back into the redwine-ml-project directory using `cd ...`

### Install requirements
All of the packages used can downloaded for free by downloading from [Anaconda](https://anaconda.org/). If you want to download the packages individually, download the following:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Usage
* Run `rw_exploration.py` to explore the data.
* Run `rw_preprocess.py` to convert the output variable into dummy variables and categorize the other features.
  * Writes new .csv file to the *processed folder*
* Run `rw_classifcation.py` to run the machine learning models and evaluation
  * Will print cross-validation scores of all the models, 
  * Will print the best model's accuracy score, out-of-bag score, and best hyperparameters to use
  * Will also show the evaluation of the model's predictions (Confusion Matrix, Classification Report, ROC AUC Curve)
