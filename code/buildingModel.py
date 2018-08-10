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
df = pd.read_csv('../../data/SampledData/ver 2/traintest.csv', names=colnames, header=0, sep=',')
# colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
#             'statistic', 'non statistic']
# df = pd.read_csv('../../data/SampledData/ver 2/2classes.csv', names=colnames, header=0, sep=',')
df = df.fillna(value=0)  # filled blank with zero
df = excludeLink(df)  # remove link from the text


def cleaningData(df):
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
    return tokenized_text


tokenized_text = cleaningData(df)
iter_tag_doc = list(read_corpus(tokenized_text))


# =============================================================================
# train data by doc2vec
# =============================================================================
def createDoc2VecModel(iter_tag_doc):
    # Create a Doc2Vec model
    model = gensim.models.Doc2Vec(vector_size=200, min_count=0
                                  , alpha=0.025, min_alpha=0.025
                                  , seed=0, workers=4)

    # Build a set of vocabulary
    model.build_vocab(iter_tag_doc)
    print('number of vocabulary : ' + str(len(model.wv.vocab)))

    # Train the doc2vec model
    model.train(iter_tag_doc, total_examples=len(tokenized_text), epochs=150)
    print('doc2vec model trained')

    # created list of vector
    docvecs = []
    for tag in range(len(iter_tag_doc)):
        inferred_vector = model.infer_vector(iter_tag_doc[tag].words)
        docvecs.append(inferred_vector)
    print('completed list')
    return docvecs


docvecs = createDoc2VecModel(iter_tag_doc)
# ================================= Model Part ========================================
y = df.iloc[:, 8:14]
x = pd.DataFrame(docvecs)
df_all = pd.concat([x, y], axis=1, join='inner')

xTrain = x.iloc[0:300, :]
xTest = x.iloc[300:500, :]

yTrain = y.iloc[0:300, :]
yTest = y.iloc[300:500, :]


# ================================== Ridge Regression =================================
def buildRidgeRegression(xTrain, yTrain, xTest, yTest, nGamma=160):
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


def buildRidgeRegressionClassifier(xTrain, yTrain):
    ridge = linear_model.RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
    ridge.fit(xTrain, yTrain)
    ridge.coef_

    return ridge


def buildRidgeRegressionCV(xTrain, yTrain, cv):
    ridge = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv)
    ridge.fit(xTrain, yTrain)
    ridge.coef_

    return ridge


# def probToBinary(yPredict):
#     for i in range(0, len(yPredict)):
#         if yPredict[i][0] >= .5:  # setting threshold to .5
#             yPredict[i][0] = 1
#             yPredict[i][1] = 0
#         else:
#             yPredict[i][0] = 0
#             yPredict[i][1] = 1
#     return yPredict

def probToBinary(yPredict):
    for i in range(0, len(yPredict)):
        for j in range(0, yPredict.shape[1]):
            if yPredict[i][j] >= 0.5:
                yPredict[i][j] = 1
            else:
                yPredict[i][j] = 0
    return yPredict


# RRPredict = buildRidgeRegression(xTrain, yTrain, xTest, yTest)
# RRRMSE = sqrt(mean_squared_error(yTest, RRPredict)) # 0.3324989491864065
# print('RRRMSE = ' + str(RRRMSE))

RRCVModel = buildRidgeRegressionCV(xTrain, yTrain, 15)
RRCVPredict = RRCVModel.predict(xTest)
RRCVPredict = probToBinary(RRCVPredict)
RRCVRMSE = sqrt(mean_squared_error(yTest, RRCVPredict))  # 0.20725988226476166
RRCVACC = accuracy_score(yTest, RRCVPredict)  # 0.14000000000000001
print('RRCVRMSE = ' + str(RRCVRMSE))
print('RRCVACC = ' + str(RRCVACC))


# ==================================== SVM ==============================================
def buildSVMClassifier(xTrain, yTrain):
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
    clf.fit(xTrain, yTrain)

    return clf


SVMModel = buildSVMClassifier(xTrain, yTrain)
SVMPredict = SVMModel.predict(xTest)
SVMPredict = probToBinary(SVMPredict)
SVMRMSE = sqrt(mean_squared_error(yTest, SVMPredict))  # 0.20538432907524162
SVMACC = accuracy_score(yTest, SVMPredict)
print('SVMRMSE = ' + str(SVMRMSE))
print('SVMACC = ' + str(SVMACC))


# ==================================== Multi-Label ======================================
# ==================================== Binary Relevance =================================
def buildBRClassifier(xTrain, yTrain):
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())

    # train
    classifier.fit(xTrain, yTrain)
    return classifier


BRModel = buildBRClassifier(xTrain, yTrain)
BRPredict = BRModel.predict(xTest)
BRACC = accuracy_score(yTest, BRPredict)  # 0.14000000000000001
BRPredict = pd.DataFrame(BRPredict.todense())
BRRMSE = sqrt(mean_squared_error(yTest, BRPredict))  # 0.46157941413754094
print('BRRMSE = ' + str(BRRMSE))
print('BR ACC = ' + str(BRACC))


# ==================================== Classifier Chains =================================
def buildCCClassifier(xTrain, yTrain):
    # initialize classifier chains multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = ClassifierChain(GaussianNB())

    # train
    classifier.fit(xTrain, yTrain)
    return classifier


CCModel = buildCCClassifier(xTrain, yTrain)
CCPredict = CCModel.predict(xTest)
CCACC = accuracy_score(yTest, CCPredict)  # 0.00666666666667
CCPredict = pd.DataFrame(CCPredict.todense())
CCRMSE = sqrt(mean_squared_error(yTest, CCPredict))  # 0.5416025603090641
print('CCRMSE = ' + str(CCRMSE))
print('CC ACC = ' + str(CCACC))


# ==================================== Label Powerset =================================
def buildLBClassifier(xTrain, yTrain):
    # initialize Label Powerset multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = LabelPowerset(GaussianNB())

    # train
    xTrain = np.ascontiguousarray(xTrain)
    yTrain = np.ascontiguousarray(yTrain)
    classifier.fit(xTrain, yTrain)

    return classifier


LPModel = buildLBClassifier(xTrain, yTrain)
LPPredict = LPModel.predict(xTest)
LPACC = accuracy_score(yTest, LPPredict)  # 0.186666666667
LPPredict = pd.DataFrame(LPPredict.todense())
LPRMSE = sqrt(mean_squared_error(yTest, LPPredict))  # 0.3257470047615344
print('LPRMSE = ' + str(LPRMSE))
print('LP ACC = ' + str(LPACC))


# ==================================== LightGBM ========================================
def buildLGBMClassifier(xTrain, yTrain, xTest, yTest):
    LGBMClassifier = []
    for column in yTrain:
        train_data = lgb.Dataset(xTrain, label=yTrain[column])
        test_data = lgb.Dataset(xTest, label=yTest[column])

        params = {}
        params['learning_rate'] = 0.001
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 10
        params['min_data'] = 50
        params['max_depth'] = 10

        num_round = 20
        classifier = lgb.train(params, train_data, num_round, valid_sets=[test_data])
        LGBMClassifier.append(classifier)
    return LGBMClassifier

# def buildLGBMClassifier(xTrain, yTrain, xTest, yTest):
#     LGBMClassifier = []
#     for column in yTrain:
#         train_data = lgb.Dataset(xTrain, label=yTrain[column])
#         test_data = lgb.Dataset(xTest, label=yTest[column])
#
#         params = {}
#         params['learning_rate'] = 0.001
#         params['boosting_type'] = 'gbdt'
#         params['objective'] = 'multiclass'
#         params['metric'] = 'multi_logloss'
#         params['sub_feature'] = 0.5
#         params['num_leaves'] = 10
#         params['min_data'] = 50
#         params['max_depth'] = 10
#         params['num_class'] = 2
#
#         num_round = 20
#         classifier = lgb.train(params, train_data, num_round, valid_sets=[test_data])
#         LGBMClassifier.append(classifier)
#     return LGBMClassifier

def predictLGBM(LGBMModel, xTest, yTrain):
    LGBMPredicts = pd.DataFrame()
    numColumn = 0
    count = 0
    columnName = []

    for column in yTrain:
        numColumn = numColumn + 1
        columnName.append(column)

    for model in LGBMModel:
        LGBMPredict = model.predict(xTest)
        for i in range(0, len(LGBMPredict)):
            if LGBMPredict[i] >= .5:  # setting threshold to .5
                LGBMPredict[i] = 1
            else:
                LGBMPredict[i] = 0
        LGBMPredicts[columnName[count]] = pd.Series(LGBMPredict)
        count = count + 1
    return LGBMPredicts


LGBMModel = buildLGBMClassifier(xTrain, yTrain, xTest, yTest)
LGBMPredict = predictLGBM(LGBMModel, xTest, yTrain)
LGBMPredict.index = np.arange(300, len(LGBMPredict)+300)
LGBMRMSE = sqrt(mean_squared_error(yTest, LGBMPredict))
LGBMACC = accuracy_score(yTest, LGBMPredict)
print('LGBMRMSE = ' + str(LGBMRMSE))
print('LGBM ACC = ' + str(LGBMACC))



# yPredict = pd.DataFrame()
# for column in yTrain:
#     train_data = lgb.Dataset(xTrain, label=yTrain[column])
#     test_data = lgb.Dataset(xTest, label=yTest[column])
#
#     params = {}
#     params['learning_rate'] = 0.001
#     params['boosting_type'] = 'gbdt'
#     params['objective'] = 'binary'
#     params['metric'] = 'binary_logloss'
#     params['sub_feature'] = 0.5
#     params['num_leaves'] = 10
#     params['min_data'] = 50
#     params['max_depth'] = 10
#
#     num_round = 20
#     bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
#     ypred = bst.predict(xTest)
#     for i in range(0, len(ypred)):
#         if ypred[i] >= .5:  # setting threshold to .5
#             ypred[i] = 1
#         else:
#             ypred[i] = 0
#     yPredict[column] = pd.Series(ypred)
#
# yPredict.index = np.arange(250, len(yPredict)+250)
# LGBMRMSE = sqrt(mean_squared_error(yTest, yPredict))
# accuracy_lgbm = accuracy_score(yTest, yPredict)
# print('LGBMRMSE = ' + str(LGBMRMSE))
# print('LGBM ACC = ' + str(accuracy_lgbm))


# colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
#             'statistic', 'non statistic']
# df_prediction = pd.read_csv('../../data/OriginalTweets/worldcup2018-07-04original.csv', names=colnames, header=0,
#                             sep=',')
# df_prediction = df_prediction.fillna(value=0)  # filled blank with zero
# df_prediction = excludeLink(df_prediction)  # remove link from the text
# df_prediction = df_prediction.iloc[0:2500, :]
#
# tokenized_text = cleaningData(df_prediction)
# iter_tag_doc = list(read_corpus(tokenized_text))
# docvecs = createDoc2VecModel(iter_tag_doc)
#
# x = pd.DataFrame(docvecs)
# yPredict = LPModel.predict(x)
# yPredict = pd.DataFrame(yPredict.todense())
# yPredict.to_csv('../../data/OriginalTweets/worldcup2018-07-04originalx.csv', encoding='utf-8', index=False,
#                 header=False)
