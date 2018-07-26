from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from math import sqrt
import pandas as pd
import gensim
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import datetime as dt
import scipy
from scipy.io import arff


def excludeLink(df):
    df = df.replace(regex=r"http\S+", value='')
    return df


def read_corpus(tokenized_text, tokens_only=False):
    for i, line in enumerate(tokenized_text):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(line, [i])


def newOutput(df):
    df_temp = pd.DataFrame(columns=['label'])
    for index, row in df.iterrows():
        strTemp = ''
        for column in row:
            strTemp = strTemp + str(int(column))
        df_temp = df_temp.append({'label': int(strTemp, 2)}, ignore_index=True)
    return df_temp


def plotRidgeCoef(gammas, coefs, errs):
    # Display results
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(gammas, coefs)
    ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(gammas, errs)
    ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')

    # horizontal line
    plt.hlines(y=min(errs), xmin=0, xmax=max(gammas), color='red', zorder=1, linestyles=':', label=str(min(errs)))

    # vertical line
    plt.vlines(x=gammas[errs.index(min(errs))], ymin=0, ymax=max(errs), color='orange', zorder=2,
               linestyles=':', label=str(gammas[errs.index(min(errs))]))
    plt.legend()
    plt.show()


colnames = ['index', 'created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
            'fact(text)',
            'statistic(numeric)', 'analytic', 'opinion&emotional', 'advertisment', 'goal', 'team_performance',
            'player_performance', 'result', 'music', 'food', 'incident', 'image', 'video']
df = pd.read_csv('../../data/SampledData/roundof16.csv', names=colnames, header=0, sep=',')
df = df.fillna(value=0)  # filled blank with zero
df = excludeLink(df)  # remove link from the text

tokenized_text = []
tokenizer = RegexpTokenizer("\w+|%|-")
# eng_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Start pre-processing
tokenized_text = []
for text in df.text:
    # Lowercase
    # lower_case = text.lower()

    # Tokenize
    tokens = tokenizer.tokenize(text)

    # Re-initial token list in each round
    filtered_tokens = []

    # Remove stop word but include the negative helping verb
    for word in tokens:
        # if not word in eng_stopwords:
        # Lemmatize
        lemmatized = lemmatizer.lemmatize(word, pos="v")
        filtered_tokens.append(lemmatized)

    # Append each tokenized tweet in the list
    tokenized_text.append(filtered_tokens)

iter_tag_doc = list(read_corpus(tokenized_text))

# =============================================================================
# train data by doc2vec
# =============================================================================
# Create a Doc2Vec model
model = gensim.models.Doc2Vec(vector_size=100, min_count=0
                              , alpha=0.025, min_alpha=0.025
                              , seed=0, workers=4)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print('number of vocabulary : ' + str(len(model.wv.vocab)))

# Train the doc2vec model
model.train(iter_tag_doc, total_examples=len(tokenized_text), epochs=100)
print('model trained')

# created list of vector
docvecs = []
for tag in range(len(iter_tag_doc)):
    inferred_vector = model.infer_vector(iter_tag_doc[tag].words)
    docvecs.append(inferred_vector)
print('completed list')

# ================================= Model Part ========================================
df_output = df.iloc[:, 9:21]
# df_output = newOutput(df_output)
df_input = pd.DataFrame(docvecs)
df_all = pd.concat([df_input, df_output], axis=1, join='inner')

df_input_train = df_input.iloc[0:700, :]
df_input_test = df_input.iloc[700:1000, :]

df_output_train = df_output.iloc[0:700, :]
df_output_test = df_output.iloc[700:1000, :]

# ================================== Ridge Regression =================================
num_gamma = 160
gammas = np.logspace(-5, 10, num_gamma)
errs = []
coefs = []
for gamma in gammas:
    ridge = linear_model.Ridge(alpha=gamma, fit_intercept=False)
    ridge.fit(df_input_train, df_output_train)
    ridge.coef_

    df_predicted = ridge.predict(df_input_test)
    result = df_output_test.sub(df_predicted)

    rms = sqrt(mean_squared_error(df_output_test, df_predicted))
    coefs.append(ridge.coef_)
    errs.append(rms)

coefs2 = []
for i in range(0, num_gamma):
    coef_temp = coefs[i][0]
    coefs2.append(coef_temp)

plotRidgeCoef(gammas, coefs2, errs)

# ==================================== SVM ==============================================
X, y = df_input_train, df_output_train
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
clf.fit(X, y)
proba = clf.predict_proba(df_input_test)
rms = sqrt(mean_squared_error(df_output_test, proba))
print(rms)

# # ==================================== Multi-Label ======================================
# # ==================================== Binary Relevance =================================
# # initialize binary relevance multi-label classifier
# # with a gaussian naive bayes base classifier
# classifier = BinaryRelevance(GaussianNB())
#
# # train
# classifier.fit(X, y)
#
# # predict
# predictions = classifier.predict(df_input_test)
# accuracy_binary = accuracy_score(df_output_test, predictions)
#
# # ==================================== Classifier Chains =================================
# # initialize classifier chains multi-label classifier
# # with a gaussian naive bayes base classifier
# classifier = ClassifierChain(GaussianNB())
#
# # train
# classifier.fit(X, y)
#
# # predict
# predictions = classifier.predict(df_input_test)
#
# accuracy_chain = accuracy_score(df_output_test, predictions)
#
#
# # ==================================== Label Powerset =================================
# # initialize Label Powerset multi-label classifier
# # with a gaussian naive bayes base classifier
# classifier = LabelPowerset(GaussianNB())
#
# # train
# classifier.fit(X, y)
#
# # predict
# predictions = classifier.predict(df_input_test)
#
# accuracy_powerset = accuracy_score(df_output_test, predictions)


# ==================================== LightGBM ========================================
# train_data = lgb.Dataset(X, label=y)
# test_data = lgb.Dataset(df_input_test, label=df_output_test)
#
# param = {'num_leaves':31, 'num_trees':100, 'objective':'binary', 'num_class': 12}
# param['metric'] = 'auc'
#
# num_round = 10
# bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
# ypred = bst.predict(test_data)

temp = pd.DataFrame()
for column in df_output_train:
    train_data = lgb.Dataset(df_input_train, label=df_output_train[column])
    test_data = lgb.Dataset(df_input_test, label=df_output_test[column])

    param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
    param['metric'] = 'auc'

    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    ypred = bst.predict(df_input_test)
    temp[column] = pd.Series(ypred)

temp.index = np.arange(700, len(temp)+700)
errs = df_output_test - temp