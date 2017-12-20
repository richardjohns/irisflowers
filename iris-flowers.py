# Python Project Template

# Initialisation and common commands

# python3.6 -m venv my_env - creates python environment
# pip3 install pandas matplotlib sklearn scipy
# source my_env/bin/activate - activates Python environment
# python3 iris-flowers.py - runs this file
# deactivate - exits virtual environment

# 1. Prepare Problem

# a) Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# b) Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
dataset = read_csv(filename, names=names)

# 2. Summarize Data

# a) Descriptive statistics
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# b) Data visualizations
# Univariate plots to better understand each attribute.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
pyplot.show()
# histograms to get an idea of the distribution - where Gaussian, we can use algos to exploit this assumption
dataset.hist()
pyplot.show()

# Multivariate plots to better understand the relationships between attributes
# scatter plot matrix to spot structured relationships between input pairs
scatter_matrix(dataset)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms

# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use