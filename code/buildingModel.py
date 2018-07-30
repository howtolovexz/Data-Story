from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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


# colnames = ['index', 'created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
#             'fact(text)',
#             'statistic(numeric)', 'analytic', 'opinion&emotional', 'advertisment', 'goal', 'team_performance',
#             'player_performance', 'result', 'music', 'food', 'incident', 'image', 'video']
# df = pd.read_csv('../../data/SampledData/roundof16.csv', names=colnames, header=0, sep=',')
colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
            'fact', 'statistic', 'analysis', 'opinion', 'unrelated']
df = pd.read_csv('../../data/SampledData/worldcupLabelled2.csv', names=colnames, header=0, sep=',')
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
print('doc2vec model trained')

# created list of vector
docvecs = []
for tag in range(len(iter_tag_doc)):
    inferred_vector = model.infer_vector(iter_tag_doc[tag].words)
    docvecs.append(inferred_vector)
print('completed list')

# ================================= Model Part ========================================
# y = df.iloc[:, 9:21]
# x = pd.DataFrame(docvecs)
# df_all = pd.concat([x, y], axis=1, join='inner')
#
# xTrain = x.iloc[0:700, :]
# xTest = x.iloc[700:1000, :]
#
# yTrain = y.iloc[0:700, :]
# yTest = y.iloc[700:1000, :]

y = df.iloc[:, 9:14]
x = pd.DataFrame(docvecs)
df_all = pd.concat([x, y], axis=1, join='inner')

xTrain = x.iloc[0:200, :]
xTest = x.iloc[200:300, :]

yTrain = y.iloc[0:200, :]
yTest = y.iloc[200:300, :]

# ================================== Ridge Regression =================================
def buildRidgeRegression(xTrain, yTrain, xTest, yTest, nGamma = 160):
    gammas = np.logspace(-5, 10, nGamma)
    errs = []
    coefs = []
    for gamma in gammas:
        ridge = linear_model.Ridge(alpha=gamma, fit_intercept=False)
        ridge.fit(xTrain, yTrain)
        ridge.coef_

        yPredict = ridge.predict(xTest)

        rms = sqrt(mean_squared_error(yTest, yPredict))
        coefs.append(ridge.coef_)
        errs.append(rms)

    coefs2 = []
    for i in range(0, nGamma):
        coef_temp = coefs[i][0]
        coefs2.append(coef_temp)

    # plotRidgeCoef(gammas, coefs2, errs)
    return yPredict

def buildRidgeRegressionClassifier(xTrain, yTrain, xTest, yTest):
    ridge = linear_model.RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
    ridge.fit(xTrain, yTrain)
    ridge.coef_

    yPredict = ridge.predict(xTest)
    return yPredict

def buildRidgeRegressionCV(xTrain, yTrain, xTest, yTest, cv):
    ridge = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv)
    ridge.fit(xTrain, yTrain)
    ridge.coef_

    yPredict = ridge.predict(xTest)

    return yPredict

RRPredict = buildRidgeRegression(xTrain, yTrain, xTest, yTest)
RRCVPredict = buildRidgeRegressionCV(xTrain, yTrain, xTest, yTest, 10)

RRRMSE = sqrt(mean_squared_error(yTest, RRPredict)) # 0.3324989491864065
RRCVRMSE = sqrt(mean_squared_error(yTest, RRCVPredict)) # 0.20725988226476166
print('RRRMSE = ' + str(RRRMSE))
print('RRCVRMSE = ' + str(RRCVRMSE))
# RRCPredict = buildRidgeRegressionClassifier(xTrain, yTrain, xTest, yTest)
# ==================================== SVM ==============================================
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
clf.fit(xTrain, yTrain)
yPredict = clf.predict_proba(xTest)
SVMRMSE = sqrt(mean_squared_error(yTest, yPredict)) # 0.20538432907524162
print('SVMRMSE = ' + str(SVMRMSE))

# ==================================== Multi-Label ======================================
# ==================================== Binary Relevance =================================
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(xTrain, yTrain)

# predict
yPredict = classifier.predict(xTest)
accuracy_binary = accuracy_score(yTest, yPredict) # 0.14000000000000001
yPredict = pd.DataFrame(yPredict.todense())
BRRMSE = sqrt(mean_squared_error(yTest, yPredict)) # 0.46157941413754094
print('BRRMSE = ' + str(BRRMSE))
print('BR ACC = ' + str(accuracy_binary))

# ==================================== Classifier Chains =================================
# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(xTrain, yTrain)

# predict
yPredict = classifier.predict(xTest)
accuracy_chain = accuracy_score(yTest, yPredict) # 0.00666666666667
yPredict = pd.DataFrame(yPredict.todense())
CCRMSE = sqrt(mean_squared_error(yTest, yPredict)) # 0.5416025603090641
print('CCRMSE = ' + str(CCRMSE))
print('CC ACC = ' + str(accuracy_chain))


# ==================================== Label Powerset =================================
# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
xTrain = np.ascontiguousarray(xTrain)
yTrain = np.ascontiguousarray(yTrain)
classifier.fit(xTrain, yTrain)

# predict
yPredict = classifier.predict(xTest)
accuracy_powerset = accuracy_score(yTest, yPredict) # 0.186666666667
yPredict = pd.DataFrame(yPredict.todense())
LPRMSE = sqrt(mean_squared_error(yTest, yPredict)) # 0.3257470047615344
print('LPRMSE = ' + str(LPRMSE))
print('LP ACC = ' + str(accuracy_powerset))

# ==================================== LightGBM ========================================
yPredict = pd.DataFrame()
for column in yTrain:
    train_data = lgb.Dataset(xTrain, label=yTrain[column])
    test_data = lgb.Dataset(xTest, label=yTest[column])

    param = {'num_leaves':50, 'num_trees':100, 'objective':'binary'}
    param['metric'] = 'auc'

    num_round = 15
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    ypred = bst.predict(xTest)
    yPredict[column] = pd.Series(ypred)

yPredict.index = np.arange(700, len(yPredict)+700)
LGBMRMSE = sqrt(mean_squared_error(yTest, yPredict)) # 0.3257470047615344
print('LGBMRMSE = ' + str(LGBMRMSE))