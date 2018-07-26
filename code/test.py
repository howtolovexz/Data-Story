import scipy
from scipy.io import arff
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
import numpy as np

data, meta = scipy.io.arff.loadarff('../../data/test/yeast-train.arff')
df_train = pd.DataFrame(data)
str_df = df_train.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
X_train = df_train.iloc[:, 0:103]
X_train = X_train.astype('float64')
y_train = str_df
y_train = y_train.astype('int')

data, meta = scipy.io.arff.loadarff('../../data/test/yeast-test.arff')
df_test = pd.DataFrame(data)
str_df = df_test.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
X_test = df_test.iloc[:, 0:103]
X_test = X_test.astype('float64')
y_test = str_df
y_test = y_test.astype('int')

# ======================= Binary Relevance =================================
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

# ======================= Classifier Chains =================================
# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

# ======================= Label Powerset =================================
# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))