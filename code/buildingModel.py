from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
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
import csv
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

def plotBoxPlot(data, label, title):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.boxplot(data, labels=label)
    axes.set_title(title)
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


# =============================================================================
# cleaning data
# =============================================================================
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


# =============================================================================
# train data by doc2vec
# =============================================================================
def createDoc2VecModel(iter_tag_doc, vectorSize):
    # Create a Doc2Vec model
    model = gensim.models.Doc2Vec(vector_size=vectorSize, min_count=0
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


# ============================= Export doc2vec list ===========================
def exportDoc2VecToCSV(docvecs):
    fileName = '../../data/SampledData/ver 2/Labelled/doc2vec.csv'
    np.savetxt(fileName, docvecs, delimiter=",")


# =============================================================================
# Model section
# =============================================================================
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


def probToBinary(yPredict):
    for i in range(0, len(yPredict)):
        for j in range(0, yPredict.shape[1]):
            if yPredict[i][j] >= 0.5:
                yPredict[i][j] = 1
            else:
                yPredict[i][j] = 0
    return yPredict


# ==================================== SVM ==============================================
def buildSVMClassifier(xTrain, yTrain):
    clf = OneVsRestClassifier(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
                                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                                  tol=0.001, verbose=False))
    clf.fit(xTrain, yTrain)

    return clf


def buildSVMCVClassifier(xTrain, yTrain):
    svc = OneVsRestClassifier(SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
                                  decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
                                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                                  tol=0.001, verbose=False))
    Cs = np.logspace(-6, -1, 10)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)
    clf.fit(xTrain, yTrain)


# ==================================== Multi-Label ======================================
# ==================================== Binary Relevance =================================
def buildBRClassifier(xTrain, yTrain):
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())

    # train
    classifier.fit(xTrain, yTrain)
    return classifier


# ==================================== Classifier Chains =================================
def buildCCClassifier(xTrain, yTrain):
    # initialize classifier chains multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = ClassifierChain(GaussianNB())

    # train
    classifier.fit(xTrain, yTrain)
    return classifier


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


# ==================================== LightGBM ========================================
def buildLGBMClassifier(xTrain, yTrain, xTest, yTest):
    LGBMClassifier = []
    for column in yTrain:
        train_data = lgb.Dataset(xTrain, label=yTrain[column])
        test_data = lgb.Dataset(xTest, label=yTest[column])

        params = {}
        params['learning_rate'] = 0.001
        params['num_iterations'] = 100
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression_l2'
        params['metric'] = 'l2_root'
        params['num_leaves'] = 100
        params['max_depth'] = 7
        params['max_bin'] = 100
        num_round = 20
        classifier = lgb.train(params, train_data, num_round, valid_sets=[test_data])
        LGBMClassifier.append(classifier)
    return LGBMClassifier


def predictLGBM(LGBMModel, xTest, yTrain):
    LGBMPredicts = pd.DataFrame()
    LGBMPredictsBinary = pd.DataFrame()
    numColumn = 0
    count = 0
    columnName = []

    for column in yTrain:
        numColumn = numColumn + 1
        columnName.append(column)

    for model in LGBMModel:
        LGBMPredict = model.predict(xTest)
        print(LGBMPredict)
        LGBMPredictBinary = LGBMPredict
        for i in range(0, len(LGBMPredict)):
            if LGBMPredictBinary[i] >= .5:  # setting threshold to .5
                LGBMPredictBinary[i] = 1
            else:
                LGBMPredictBinary[i] = 0
        LGBMPredicts[columnName[count]] = pd.Series(LGBMPredict)
        LGBMPredictsBinary[columnName[count]] = pd.Series(LGBMPredictBinary)
        count = count + 1
    return LGBMPredicts, LGBMPredictsBinary


# ======================= Random Forest =========================
def buildRandomForestClassifier(xTrain, yTrain):
    clf = RandomForestClassifier(n_estimators=20, max_depth=2, min_samples_split=10, random_state=0, class_weight=None)
    clf.fit(xTrain, yTrain)

    return clf


# ======================================= Main Program =================================================
fileName = '../../data/SampledData/ver 2/Labelled/train2classes.csv'
# colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
#             'fact', 'statistic', 'analysis', 'opinion', 'unrelated', 'etc', 'score', 'goal', 'bet', 'match quality',
#             'world record', 'tournament record', 'team performance', 'personal performance', 'team record',
#             'personal record', 'number of attender', 'music', 'food']
colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
            'statistic', 'non statistic', 'etc', 'score', 'goal', 'bet', 'match quality',
            'world record', 'tournament record', 'team performance', 'personal performance', 'team record',
            'personal record', 'number of attender', 'music', 'food']
# colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status',
#             'fact', 'statistic', 'opinion', 'unrelated', 'etc', 'score', 'goal', 'bet', 'match quality',
#             'world record', 'tournament record', 'team performance', 'personal performance', 'team record',
#             'personal record', 'number of attender', 'music', 'food']
df = pd.read_csv(fileName, names=colnames, header=0, sep=',')
df = df.fillna(value=0)  # filled blank with zero
df = excludeLink(df)  # remove link from the text

# df.opinion = pd.to_numeric(df.opinion, errors='coerce')

# tokenized_text = cleaningData(df)
# iter_tag_doc = list(read_corpus(tokenized_text))
#
# vectorSize = 100
# docvecs = createDoc2VecModel(iter_tag_doc, vectorSize)
# exportDoc2VecToCSV(docvecs)

# ================================= Prepare train & test data ========================================
numRow = len(df)
numTrain = 500
y = df.iloc[:, 8:10]
# x = pd.DataFrame(docvecs)
x = pd.read_csv('../../data/SampledData/ver 2/Labelled/doc2vec.csv', sep=',', header=None)
df_all = pd.concat([x, y], axis=1, join='inner')

xTrain = x.iloc[0:numTrain, :]
xTest = x.iloc[numTrain:numRow, :]

yTrain = y.iloc[0:numTrain, :]
yTest = y.iloc[numTrain:numRow, :]

kf = KFold(n_splits=10, shuffle=True, random_state=0)
RRCVRMSEList = []
SVMRMSEList = []
BRRMSEList = []
CCRMSEList = []
LPRMSEList = []
LGBMRMSEList = []
RFRMSEList = []

RRCVACCList = []
SVMACCList = []
BRACCList = []
CCACCList = []
LPACCList = []
LGBMACCList = []
RFACCList = []
for train_index, test_index in kf.split(x):
    xTrain = x.iloc[train_index]
    xTest = x.iloc[test_index]
    yTrain = y.iloc[train_index]
    yTest = y.iloc[test_index]

    # ================================= Model Part ========================================
    RRCVModel = buildRidgeRegressionCV(xTrain, yTrain, 10)
    RRCVPredict = RRCVModel.predict(xTest)
    RRCVPredictBinary = probToBinary(RRCVPredict)
    RRCVRMSE = sqrt(mean_squared_error(yTest, RRCVPredict))
    RRCVACC = accuracy_score(yTest, RRCVPredictBinary)
    print('RRCVRMSE = ' + str(RRCVRMSE))
    print('RRCVACC = ' + str(RRCVACC))

    SVMModel = buildSVMClassifier(xTrain, yTrain)
    SVMPredict = SVMModel.predict(xTest)
    SVMPredictBinary = probToBinary(SVMPredict)
    SVMRMSE = sqrt(mean_squared_error(yTest, SVMPredict))
    SVMACC = accuracy_score(yTest, SVMPredictBinary)
    print('SVMRMSE = ' + str(SVMRMSE))
    print('SVMACC = ' + str(SVMACC))

    BRModel = buildBRClassifier(xTrain, yTrain)
    BRPredict = BRModel.predict(xTest)
    BRACC = accuracy_score(yTest, BRPredict)
    BRPredict = pd.DataFrame(BRPredict.todense())
    BRRMSE = sqrt(mean_squared_error(yTest, BRPredict))
    print('BRRMSE = ' + str(BRRMSE))
    print('BR ACC = ' + str(BRACC))

    CCModel = buildCCClassifier(xTrain, yTrain)
    CCPredict = CCModel.predict(xTest)
    CCACC = accuracy_score(yTest, CCPredict)
    CCPredict = pd.DataFrame(CCPredict.todense())
    CCRMSE = sqrt(mean_squared_error(yTest, CCPredict))
    print('CCRMSE = ' + str(CCRMSE))
    print('CC ACC = ' + str(CCACC))

    LPModel = buildLBClassifier(xTrain, yTrain)
    LPPredict = LPModel.predict(xTest)
    LPACC = accuracy_score(yTest, LPPredict)
    LPPredict = pd.DataFrame(LPPredict.todense())
    LPRMSE = sqrt(mean_squared_error(yTest, LPPredict))
    print('LPRMSE = ' + str(LPRMSE))
    print('LP ACC = ' + str(LPACC))

    LGBMModel = buildLGBMClassifier(xTrain, yTrain, xTest, yTest)
    LGBMPredict, LGBMPredictBinary = predictLGBM(LGBMModel, xTest, yTrain)
    LGBMPredict.index = np.arange(numTrain, len(LGBMPredict) + numTrain)
    LGBMRMSE = sqrt(mean_squared_error(yTest, LGBMPredict))
    LGBMACC = accuracy_score(yTest, LGBMPredict)
    print('LGBMRMSE = ' + str(LGBMRMSE))
    print('LGBM ACC = ' + str(LGBMACC))

    RFModel = buildRandomForestClassifier(xTrain, yTrain)
    RFPredict = RFModel.predict(xTest)
    RFRMSE = sqrt(mean_squared_error(yTest, RFPredict))
    RFACC = accuracy_score(yTest, RFPredict)
    print('RFRMSE = ' + str(RFRMSE))
    print('RF ACC = ' + str(RFACC))

    RRCVRMSEList.append(RRCVRMSE)
    SVMRMSEList.append(SVMRMSE)
    BRRMSEList.append(BRRMSE)
    CCRMSEList.append(CCRMSE)
    LPRMSEList.append(LPRMSE)
    LGBMRMSEList.append(LGBMRMSE)
    RFRMSEList.append(RFRMSE)

    RRCVACCList.append(RRCVACC)
    SVMACCList.append(SVMACC)
    BRACCList.append(BRACC)
    CCACCList.append(CCACC)
    LPACCList.append(LPACC)
    LGBMACCList.append(LGBMACC)
    RFACCList.append(RFACC)

# basic plot
ACCList = [RRCVACCList, SVMACCList, BRACCList, CCACCList, LPACCList, LGBMACCList, RFACCList]
RMSEList = [RRCVRMSEList, SVMRMSEList, BRRMSEList, CCRMSEList, LPRMSEList, LGBMRMSEList, RFRMSEList]
label = ['Ridge Regression', 'SVM', 'Binary Relevance', 'Classifier Chains', 'Label Powerset', 'LightGBM', 'Random Forest']
title = 'Models Accuracy'
plotBoxPlot(ACCList, label, title)
title = 'Models RMSE'
plotBoxPlot(RMSEList, label, title)