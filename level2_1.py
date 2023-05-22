#### IMPORTING RELEVANT LIBRARIES

# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#### CREATING A WRAPPER FOR THRESHOLDS

from sklearn.base import BaseEstimator, ClassifierMixin

class CustomThreshold(BaseEstimator, ClassifierMixin):
    """ Custom threshold wrapper for binary classification"""
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold
    def fit(self, *args, **kwargs):
        self.base.fit(*args, **kwargs)
        return self
    def predict(self, X):
        return (self.base.predict_proba(X)[:, 1] > self.threshold).astype(int)


#### IMPORTING DATA TO PYTHON

## Train Data

raw_train = pd.read_csv('heart.csv')

## raw_train['HeartDisease'] = raw_train['HeartDisease'].astype(object)

#### DATA UNDERSTANDING

#### Column names

print(raw_train.columns)

#### Summary statistics of numeric attributes

raw_train.describe()

#### Summary statistics of categorical attributes

raw_train.describe(include='O')

#### Top 10 records

raw_train.head(10)

#### Missing values

raw_train.isnull().sum()

#### Value Counts

raw_train['Sex'].value_counts()
raw_train['ChestPainType'].value_counts()
raw_train['RestingECG'].value_counts()
raw_train['ExerciseAngina'].value_counts()
raw_train['ST_Slope'].value_counts()

raw_train['HeartDisease'].value_counts()

#### DATA PREPARATION

raw_train = raw_train[raw_train['RestingBP']>0]

# ExerciseAngina feature in train
raw_train.ExerciseAngina.replace(['N', 'Y'], [0, 1], inplace=True)

raw_train['ExerciseAngina'] = raw_train['ExerciseAngina'].astype(object)


#### One hot encoding

# Converting Nominal feature
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ohe = OneHotEncoder()
ohe.fit(raw_train[['Sex']])
encoded_values = ohe.transform(raw_train[['Sex']])
raw_train[ohe.categories_[0]] = encoded_values.toarray()
raw_train = raw_train.drop('Sex', axis=1)

ohe = OneHotEncoder()
ohe.fit(raw_train[['ChestPainType']])
encoded_values = ohe.transform(raw_train[['ChestPainType']])
raw_train[ohe.categories_[0]] = encoded_values.toarray()
raw_train = raw_train.drop('ChestPainType', axis=1)

ohe = OneHotEncoder()
ohe.fit(raw_train[['RestingECG']])
encoded_values = ohe.transform(raw_train[['RestingECG']])
raw_train[ohe.categories_[0]] = encoded_values.toarray()
raw_train = raw_train.drop('RestingECG', axis=1)


ohe = OneHotEncoder()
ohe.fit(raw_train[['ST_Slope']])
encoded_values = ohe.transform(raw_train[['ST_Slope']])
raw_train[ohe.categories_[0]] = encoded_values.toarray()
raw_train = raw_train.drop('ST_Slope', axis=1)

raw_train.iloc[:,8:20] = raw_train.iloc[:,8:20].astype(int)
raw_train.iloc[:,8:20] = raw_train.iloc[:,8:20].astype(object)

#### Data partioning

raw_train_cols = raw_train.columns.tolist()

raw_train_cols.insert(20, raw_train_cols.pop(raw_train_cols.index('HeartDisease')))

raw_train = raw_train.reindex(columns = raw_train_cols)

raw_train['HeartDisease'] = raw_train['HeartDisease'].astype(int)

# Divide datset
raw_train_X = raw_train.iloc[:, :-1]
raw_train_y = raw_train.iloc[:, -1]
raw_train_X.shape
raw_train_y.shape

raw_train_y.head()

X_train, X_val, y_train, y_val = train_test_split(raw_train_X, raw_train_y, test_size=0.3)

#### MODEL BUILDING

#### LR

model_lr = LogisticRegression(solver='liblinear')
model_lr.fit(X_train, y_train)
y_pred_proba = model_lr.predict_proba(X_val)[:, 1]

clf = CustomThreshold(model_lr, 0.50)
y_pred = clf.predict(X_val)

# Accuracy
lr_accuracy = accuracy_score(y_val, y_pred)
lr_f1_score = f1_score(y_val, y_pred)
lr_roc_auc = roc_auc_score(y_val, y_pred_proba)
lr_cf_matrix = confusion_matrix(y_val, y_pred)

print('LogisticRegression - accuracy score: {} and f1_score: {} and roc_auc: {}'.format(lr_accuracy, lr_f1_score, lr_roc_auc))


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                lr_cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     lr_cf_matrix.flatten()/np.sum(lr_cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(lr_cf_matrix, annot=labels, fmt='', cmap='Blues')




