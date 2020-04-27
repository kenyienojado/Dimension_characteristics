# -*- encoding: utf-8 -*-

import os
from time import time
import argparse
import scipy
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, make_scorer
from sklearn.externals import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

import numpy as np
np.set_printoptions(threshold=np.inf)


__author__ = 'CMendezC'

# Goal: training, crossvalidation and testing transcription factor structural domain sentences

# Parameters:
# 1) --inputPath Path to read input files.
# 2) --inputTrainingData File to read training data.
# 4) --inputTrainingClasses File to read training classes.
# 3) --inputTestingData File to read testing data.
# 4) --inputTestingClasses File to read testing classes.
# 5) --outputModelPath Path to place output model.
# 6) --outputModelFile File to place output model.
# 7) --outputReportPath Path to place evaluation report.
# 8) --outputReportFile File to place evaluation report.
# 9) --classifier Classifier: BernoulliNB, SVM, kNN, MultinomialNB
# 10) --saveData Save matrices
# 11) --kernel Kernel
# 12) --reduction Feature selection or dimensionality reduction
# 13) --removeStopWords Remove most frequent words
# 14) --vectorizer Vectorizer: b=binary, f=frequency, t=tf-idf.


# Ouput:
# 1) Classification model and evaluation report.

# Execution:

# source activate python3
# python training-crossvalidation-testing-dom.py
# --inputPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset
# --inputTrainingData trainData.txt
# --inputTrainingClasses trainClasses.txt
# --inputTestingData testData.txt
# --inputTestingClasses testClasses.txt
# --outputModelPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset/models
# --outputModelFile SVM-lineal-model.mod
# --outputReportPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset/reports
# --outputReportFile SVM-linear.txt
# --classifier SVM
# --saveData
# --kernel linear
# --reduction SVD200
# --removeStopWords
# --vectorizer b
# --ngrinitial 2
# --ngrfinal 2
# python training-crossvalidation-testing-dom-v03.py --inputPath /home/storage/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset --inputTrainingData trainData.txt --inputTrainingClasses trainClasses.txt --inputTestingData testData.txt --inputTestingClasses testClasses.txt --outputModelPath /home/storage/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset/models --outputModelFile SVM-lineal-model.mod --outputReportPath /home/storage/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/structural-domain-dataset/reports --outputReportFile SVM-linear.txt --classifier BernoulliNB --saveData --vectorizer b --ngrinitial 2 --ngrfinal 2 --removeStopWords
# --reduction MI
# --removeStopWords

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = argparse.ArgumentParser(description='Training validation structural domain dataset.')
    parser.add_argument("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_argument("--inputTrainingData", dest="inputTrainingData",
                      help="File to read training data", metavar="FILE")
    parser.add_argument("--inputTrainingClasses", dest="inputTrainingClasses",
                      help="File to read training classes", metavar="FILE")
    parser.add_argument("--inputTestingData", dest="inputTestingData",
                      help="File to read testing data", metavar="FILE")
    parser.add_argument("--inputTestingClasses", dest="inputTestingClasses",
                      help="File to read testing classes", metavar="FILE")
    parser.add_argument("--outputModelPath", dest="outputModelPath",
                      help="Path to place output model", metavar="PATH")
    parser.add_argument("--outputModelFile", dest="outputModelFile",
                      help="File to place output model", metavar="FILE")
    parser.add_argument("--outputReportPath", dest="outputReportPath",
                      help="Path to place evaluation report", metavar="PATH")
    parser.add_argument("--outputReportFile", dest="outputReportFile",
                      help="File to place evaluation report", metavar="FILE")
    parser.add_argument("--classifier", dest="classifier",
                      help="Classifier", metavar="NAME",
                      choices=('BernoulliNB', 'SVM', 'kNN', 'MultinomialNB'), default='SVM')
    parser.add_argument("--saveData", dest="saveData", action='store_true',
                      help="Save matrices")
    parser.add_argument("--kernel", dest="kernel",
                      help="Kernel SVM", metavar="NAME",
                      choices=('linear', 'rbf', 'poly'), default='linear')
    parser.add_argument("--reduction", dest="reduction",
                      help="Feature selection or dimensionality reduction", metavar="NAME",
                      choices=('SVD200', 'SVD300', 'CHI2', 'MI'), default=None)
    parser.add_argument("--removeStopWords", default=False,
                      action="store_true", dest="removeStopWords",
                      help="Remove stop words")
    parser.add_argument("--ngrinitial", type=int,
                      dest="ngrinitial", default=1,
                      help="Initial n-gram", metavar="INTEGER")
    parser.add_argument("--ngrfinal", type=int,
                      dest="ngrfinal", default=1,
                      help="Final n-gram", metavar="INTEGER")
    parser.add_argument("--vectorizer", dest="vectorizer", required=True,
                      help="Vectorizer: b=binary, f=frequency, t=tf-idf", metavar="CHAR",
                      choices=('b', 'f', 't'), default='b')

    args = parser.parse_args()

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(args.inputPath))
    print("File to read training data: " + str(args.inputTrainingData))
    print("File to read training classes: " + str(args.inputTrainingClasses))
    print("File to read testing data: " + str(args.inputTestingData))
    print("File to read testing classes: " + str(args.inputTestingClasses))
    print("Path to place output model: " + str(args.outputModelPath))
    print("File to place output model: " + str(args.outputModelFile))
    print("Path to place evaluation report: " + str(args.outputReportPath))
    print("File to place evaluation report: " + str(args.outputReportFile))
    print("Classifier: " + str(args.classifier))
    print("Save matrices: " + str(args.saveData))
    print("Kernel: " + str(args.kernel))
    print("Reduction: " + str(args.reduction))
    print("Remove stop words: " + str(args.removeStopWords))
    print("Initial ngram: " + str(args.ngrinitial))
    print("Final ngram: " + str(args.ngrfinal))
    print("Vectorizer: " + str(args.vectorizer))



    # Start time
    t0 = time()

    if args.removeStopWords:
        pf = stopwords.words('english')
    else:
        pf = None

    y_train = []
    trainingData = []
    y_test = []
    testingData = []
    X_train = None
    X_test = None

    if args.saveData:
        print("Reading training data and true classes...")
        with open(os.path.join(args.inputPath, args.inputTrainingClasses), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                y_train.append(line)
        with open(os.path.join(args.inputPath, args.inputTrainingData), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                trainingData.append(line)
        print("   Done!")

        print("Reading testing data and true classes...")
        with open(os.path.join(args.inputPath, args.inputTestingClasses), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                y_test.append(line)
        with open(os.path.join(args.inputPath, args.inputTestingData), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                testingData.append(line)
        print("   Done!")

        # Create vectorizer
        print('Vectorization: {}'.format(args.vectorizer))
        if args.vectorizer == "b":
            # Binary vectorizer
            vectorizer = CountVectorizer(ngram_range=(args.ngrinitial, args.ngrfinal), binary=True, stop_words=pf)
        elif args.vectorizer == "f":
            # Frequency vectorizer
            vectorizer = CountVectorizer(ngram_range=(args.ngrinitial, args.ngrfinal), stop_words=pf)
        else:
            # Binary vectorizer
            vectorizer = TfidfVectorizer(ngram_range=(args.ngrinitial, args.ngrfinal), stop_words=pf)

        X_train = csr_matrix(vectorizer.fit_transform(trainingData), dtype='double')
        X_test = csr_matrix(vectorizer.transform(testingData), dtype='double')
        print("   Saving matrix and classes...")
        joblib.dump(X_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        joblib.dump(y_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        joblib.dump(X_test, os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
        joblib.dump(y_test, os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
        print("      Done!")

    else:
        print("   Loading matrix and classes...")
        X_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        y_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        X_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
        y_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
        print("      Done!")

    print("   Number of training classes: {}".format(len(y_train)))
    print("   Number of training class DOM: {}".format(y_train.count('DOM')))
    print("   Number of training class OTHER: {}".format(y_train.count('OTHER')))
    print("   Shape of training matrix: {}".format(X_train.shape))

    print("   Number of testing classes: {}".format(len(y_test)))
    print("   Number of testing class DOM: {}".format(y_test.count('DOM')))
    print("   Number of testing class OTHER: {}".format(y_test.count('OTHER')))
    print("   Shape of testing matrix: {}".format(X_test.shape))

    # Feature selection and dimensional reduction

    X_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
    y_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
    X_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
    y_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
    if args.reduction is not None:
        print('Performing dimensionality reduction or feature selection...', args.reduction)
        if args.reduction == 'SVD200':
            reduc = TruncatedSVD(n_components=200, random_state=42)
            X_train = reduc.fit_transform(X_train)
        if args.reduction == 'SVD300':
            reduc = TruncatedSVD(n_components=300, random_state=42)
            X_train = reduc.fit_transform(X_train)
        elif args.reduction == 'CHI2':
            nombres = vectorizer.get_feature_names()
            tablita= chi2(X_train, y_train)
            orden = np.argsort(tablita[0])
            reduc = SelectKBest(chi2, k= 3001)
            X_train = reduc.fit_transform(X_train, y_train)
            buenos_2 = [ (nombres[i],tablita[0][i]) for i in orden]
        elif args.reduction == 'MI':
            nombres = vectorizer.get_feature_names()
            tablita= mutual_info_classif(X_train, y_train)
            orden = np.argsort(tablita)
            reduc = SelectKBest(mutual_info_classif, k= 4001)
            X_train = reduc.fit_transform(X_train, y_train)
            buenos_2 = [ (nombres[i],tablita[i]) for i in orden]


        print("   Done!")
        print('     New shape of training matrix: ', X_train.shape)

    jobs = -1
    paramGrid = []
    nIter = 20
    crossV = 10
    # New performance scorer
    myScorer = make_scorer(f1_score, average='weighted')
    print("Defining randomized grid search...")
    if args.classifier == 'SVM':
        # SVM
        classifier = SVC()
        if args.kernel == 'rbf':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                        'gamma': scipy.stats.expon(scale=.1),
                        'kernel': ['rbf'], 'class_weight': ['balanced', None]}
        elif args.kernel == 'linear':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                        'kernel': ['linear'],
                        'class_weight': ['balanced', None]}
        elif args.kernel == 'poly':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                        'gamma': scipy.stats.expon(scale=.1), 'degree': [2, 3],
                        'kernel': ['poly'], 'class_weight': ['balanced', None]}
        myClassifier = model_selection.RandomizedSearchCV(classifier,
                    paramGrid, n_iter=nIter,
                    cv=crossV, n_jobs=jobs, verbose=3, scoring=myScorer)
    elif args.classifier == 'BernoulliNB':
        # BernoulliNB
        classifier = BernoulliNB()
        paramGrid = {'alpha': scipy.stats.expon(scale=1.0)}
        myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=nIter,
                                                          cv=crossV, n_jobs=jobs, verbose=3, scoring=myScorer)
    elif args.classifier == 'MultinomialNB':
        # MultinomialNB
        classifier = MultinomialNB()
        paramGrid = {'alpha': scipy.stats.expon(scale=1.0)}
        myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=nIter,
                                                          cv=crossV, n_jobs=jobs, verbose=3, scoring=myScorer)
    else:
        print("Bad classifier")
        exit()
    print("   Done!")

    print("Training...")
    myClassifier.fit(X_train, y_train)
    print("   Done!")

    print("Getting best model and hyperparameters")
    print('Best score {}: {}\n'.format(myScorer, myClassifier.best_score_))
    print('Best parameters:\n')
    best_parameters = myClassifier.best_estimator_.get_params()
    for param in sorted(best_parameters.keys()):
        print("\t%s: %r\n" % (param, best_parameters[param]))
    theBestClassifier = myClassifier.best_estimator_
    print(str(theBestClassifier) + '\n')
    print("FALTA ESCRIBIR EL MEJOR SCORE (\"myClassifier.best_score_\") A UN ARCHIVO")

    print("Testing (prediction in new data)...")
    if args.reduction is not None:
        X_test = reduc.transform(X_test)
    y_pred = myClassifier.predict(X_test)
    best_parameters = myClassifier.best_estimator_.get_params()
    print("   Done!")

    print("Saving report...")
    with open(os.path.join(args.outputReportPath, args.outputReportFile ), mode='w', encoding='utf8') as oFile:
        oFile.write('**********        EVALUATION REPORT     **********\n')
        oFile.write('Reduction: {}\n'.format(args.reduction))
        oFile.write('Classifier: {}\n'.format(args.classifier))
        oFile.write('Kernel: {}\n'.format(args.kernel))
        oFile.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
        oFile.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='weighted')))
        oFile.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='weighted')))
        oFile.write('F-score: {}\n'.format(f1_score(y_test, y_pred, average='weighted')))
        oFile.write('Confusion matrix: \n')
        oFile.write(str(confusion_matrix(y_test, y_pred)) + '\n')
        oFile.write('Classification report: \n')
        oFile.write(classification_report(y_test, y_pred) + '\n')
        oFile.write('Best parameters: \n')
        for param in sorted(best_parameters.keys()):
            oFile.write("\t%s: %r\n" % (param, best_parameters[param]))

    with open(os.path.join(args.outputReportPath, args.outputReportFile + "_SCORES"), mode='w', encoding='utf8') as oFile:
        oFile.write("#feature\tScore\n")
        for i in buenos_2:
            oFile.write(str(i[0]) + "\t" + str(i[1]) + "\n")

    print("   Done!")

    print("Training and testing done in: %fs" % (time() - t0))
