  # -*- encoding: utf-8 -*-

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from time import time
import argparse
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, make_scorer
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import scipy
import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, \
    OneSidedSelection, RandomUnderSampler, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold, ClusterCentroids
from imblearn.ensemble import EasyEnsemble, BalanceCascade
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import MDS

__author__ = 'CMendezC'

# Goal: training, crossvalidation and testing binding thrombin data set

# Parameters:
# 1) --inputPath Path to read input files.
# 2) --inputTrainingData File to read training data.
# 3) --inputTestingData File to read testing data.
# 4) --inputTestingClasses File to read testing classes.
# 5) --outputModelPath Path to place output model.
# 6) --outputModelFile File to place output model.
# 7) --outputReportPath Path to place evaluation report.
# 8) --outputReportFile File to place evaluation report.
# 9) --classifier Classifier: BernoulliNB, SVM, kNN.
# 10) --saveData Save matrices
# 11) --kernel Kernel
# 12) --reduction Feature selection or dimensionality reduction
# 13) --imbalanced Imbalanced method

# Ouput:
# 1) Classification model and evaluation report.

# Execution:

# python training-crossvalidation-testing-binding-thrombin.py
# --inputPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset
# --inputTrainingData thrombin.data
# --inputTestingData Thrombin.testset
# --inputTestingClasses Thrombin.testset.class
# --outputModelPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset/models
# --outputModelFile SVM-lineal-model.mod
# --outputReportPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset/reports
# --outputReportFile SVM-lineal.txt
# --classifier SVM
# --saveData
# --kernel linear
# --imbalanced RandomUS
# --CV
# --DimRed

# source activate python3
# python training-crossvalidation-testing-binding-thrombin.py --inputPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset --inputTrainingData thrombin.data --inputTestingData Thrombin.testset --inputTestingClasses Thrombin.testset.class --outputModelPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset/models --outputModelFile SVM-lineal-model.mod --outputReportPath /home/compu2/bionlp/lcg-bioinfoI-bionlp/clasificacion-automatica/binding-thrombin-dataset/reports --outputReportFile SVM-lineal.txt --classifier SVM --kernel linear --imbalanced RandomUS

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = argparse.ArgumentParser(description='Training validation Binding Thrombin Dataset.')
    parser.add_argument("--DimRed", dest="DimRed",
                      help="Dimensionality Reduction Method",
                      choices=('PCA', 'SVD', 'MDS'), metavar="NAME")
    parser.add_argument("--CV", dest="CV",
                      help="Cross Validation", metavar="INT")
    parser.add_argument("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_argument("--inputTrainingData", dest="inputTrainingData",
                      help="File to read training data", metavar="FILE")
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
                      choices=('CHI2', 'MI'), default=None)
    parser.add_argument("--imbalanced", dest="imbalanced",
                      choices=('RandomUS', 'Tomek', 'NCR',
                           'IHT', 'RandomOS', 'ADASYN', 'SMOTE_reg',
                           'SMOTE_svm', 'SMOTE_b1', 'SMOTE_b2', 'OSS',
                           'SMOTE+ENN'), default=None,
                      help="Undersampling: RandomUS, Tomek, Neighbourhood Cleanning Rule (NCR), "
                           "Instance Hardess Threshold (IHT), One Sided Selection (OSS). "
                           "Oversampling: RandomOS, ADACYN, SMOTE_reg, "
                           "SMOTE_svm, SMOTE_b1, SMOTE_b2. Combine: "
                           "SMOTE + ENN", metavar="TEXT")

    args = parser.parse_args()

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(args.inputPath))
    print("File to read training data: " + str(args.inputTrainingData))
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
    print("Imbalanced: " + str(args.imbalanced))
    print("CV: " + str(args.CV))
    print("Dimensionality Reduction: " + str(args.DimRed))

    # Start time
    t0 = time()

    print("Reading training data and true classes...")
    X_train = None
    if args.saveData:
        y_train = []
        trainingData = []
        with open(os.path.join(args.inputPath, args.inputTrainingData), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                listLine = line.split(',')
                y_train.append(listLine[0])
                trainingData.append(listLine[1:])
        # X_train = np.matrix(trainingData)
        X_train = csr_matrix(trainingData, dtype='double')
        print("   Saving matrix and classes...")
        joblib.dump(X_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        joblib.dump(y_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        print("      Done!")
    else:
        print("   Loading matrix and classes...")
        X_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        y_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        print("      Done!")

    print("   Number of training classes: {}".format(len(y_train)))
    print("   Number of training class A: {}".format(y_train.count('A')))
    print("   Number of training class I: {}".format(y_train.count('I')))
    print("   Shape of training matrix: {}".format(X_train.shape))
    if args.DimRed == 'PCA':
      it=[100,150,200,250,300]
      print("PCA")
    if args.DimRed == 'SVD':
      it=[100,150,200,250,300]
      print("SVD")
    if args.DimRed == 'MDS':
      it=[100,150,200,250,300]
      print("MDS")
    if args.reduction == 'CHI2':
      it=[200,450,700,2500,8500]
      print("CHI2")
    if args.reduction == 'MI':
      it=[200,450,700,2500,8500]
      print("MI")
    print(it)

    # Feature selection and dimensional reduction
    for i in it:
      print(i)
      X_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
      y_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))     
      if args.reduction is not None:
          print('Performing feature selection...', args.reduction)
          if args.reduction == 'CHI2':
            reduc = SelectKBest(chi2, k= i)
            X_train = reduc.fit_transform(X_train, y_train)
          if args.reduction == 'MI':
            reduc = SelectKBest(mutual_info_classif, k= i)
            X_train = reduc.fit_transform(X_train, y_train)
      if args.DimRed is not None:
          print('Performing dimensionality reduction...', args.DimRed)
          if args.DimRed == 'SVD':
            svd = TruncatedSVD(n_components=i, n_iter=7, random_state=42)
            X_train = svd.fit_transform(X_train.toarray())
          if args.DimRed == 'PCA':
            pca = PCA(n_components=i)
            X_train = pca.fit_transform(X_train.toarray())
          if args.DimRed == 'MDS':
            embedding = MDS(n_components=i)
            X_train = embedding.fit_transform(X_train.toarray())
        
      print("   Done!")
      print('     New shape of training matrix: ', X_train.shape)

      if args.imbalanced != None:
          t1 = time()
          # Combination over and under sampling
          jobs = 15
          if args.imbalanced == "SMOTE+ENN":
              sm = SMOTEENN(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "SMOTE+Tomek":
              sm = SMOTETomek(random_state=42, n_jobs=jobs)
          # Over sampling
          elif args.imbalanced == "SMOTE_reg":
              sm = SMOTE(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "SMOTE_svm":
              sm = SMOTE(random_state=42, n_jobs=jobs, kind='svm')
          elif args.imbalanced == "SMOTE_b1":
              sm = SMOTE(random_state=42, n_jobs=jobs, kind='borderline1')
          elif args.imbalanced == "SMOTE_b2":
              sm = SMOTE(random_state=42, n_jobs=jobs, kind='borderline2')
          elif args.imbalanced == "RandomOS":
              sm = RandomOverSampler(random_state=42)
          # Under sampling
          elif args.imbalanced == "ENN":
              sm = EditedNearestNeighbours(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "Tomek":
              sm = TomekLinks(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "OSS":
              sm = OneSidedSelection(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "RandomUS":
              sm = RandomUnderSampler(random_state=42)
          elif args.imbalanced == "NCR":
              sm = NeighbourhoodCleaningRule(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "IHT":
              sm = InstanceHardnessThreshold(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "ClusterC":
              sm = ClusterCentroids(random_state=42, n_jobs=jobs)
          elif args.imbalanced == "Balanced":
              sm = BalanceCascade(random_state=42)
          elif args.imbalanced == "Easy":
              sm = EasyEnsemble(random_state=42, n_subsets=3)
          elif args.imbalanced == "ADASYN":
              sm = ADASYN(random_state=42, n_jobs=jobs)

          # Apply transformation
          X_train, y_train = sm.fit_sample(X_train, y_train)

          print("  After transformtion with {}".format(args.imbalanced))
          print("   Number of training classes: {}".format(len(y_train)))
          print("   Number of training class A: {}".format(list(y_train).count('A')))
          print("   Number of training class I: {}".format(list(y_train).count('I')))
          print("   Shape of training matrix: {}".format(X_train.shape))
          print("      Data transformation done in : %fs" % (time() - t1))

      print("Reading testing data and true classes...")
      X_test = None
      if args.saveData:
          y_test = []
          testingData = []
          with open(os.path.join(args.inputPath, args.inputTestingData), encoding='utf8', mode='r') \
                  as iFile:
              for line in iFile:
                  line = line.strip('\r\n')
                  listLine = line.split(',')
                  testingData.append(listLine[1:])
          X_test = csr_matrix(testingData, dtype='double')
          with open(os.path.join(args.inputPath, args.inputTestingClasses), encoding='utf8', mode='r') \
                  as iFile:
              for line in iFile:
                  line = line.strip('\r\n')
                  y_test.append(line)
          print("   Saving matrix and classes...")
          joblib.dump(X_test, os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
          joblib.dump(y_test, os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
          print("      Done!")
      else:
          print("   Loading matrix and classes...")
          X_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
          y_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
          print("      Done!")

      print("   Number of testing classes: {}".format(len(y_test)))
      print("   Number of testing class A: {}".format(y_test.count('A')))
      print("   Number of testing class I: {}".format(y_test.count('I')))
      print("   Shape of testing matrix: {}".format(X_test.shape))

      jobs = -1
      paramGrid = []
      nIter = 20
      crossV = int(args.CV)
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
                      cv=crossV, n_jobs=jobs, verbose=3)
      
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
      # elif args.classifier == 'kNN':
      #     # kNN
      #     k_range = list(range(1, 7, 2))
      #     classifier = KNeighborsClassifier()
      #     paramGrid = {'n_neighbors ': k_range}
      #     myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=3,
      #                                                       cv=crossV, n_jobs=jobs, verbose=3)
      else:
          print("Bad classifier")
          exit()
      print("   Done!")

      print("Training...")
      
      myClassifier.fit(np.absolute(X_train), y_train)
      print("   Done!")

      print("Testing (prediction in new data)...")
      if args.reduction is not None:
          X_test = reduc.transform(X_test)
      if args.DimRed == 'SVD':
            svd = TruncatedSVD(n_components=i, n_iter=7, random_state=42)
            X_test = svd.fit_transform(X_test.toarray())
      elif args.DimRed == 'PCA':
            pca = PCA(n_components=i)
            X_test = pca.fit_transform(X_test.toarray())
      elif args.DimRed == 'MDS':
            embedding = MDS(n_components=i)
            X_test = embedding.fit_transform(X_test.toarray())
      print(X_test.shape,X_train.shape)
      y_pred = myClassifier.predict(X_test)
      best_parameters = myClassifier.best_estimator_.get_params()
      print("   Done!")

      print("Saving report...")
      with open(os.path.join(args.outputReportPath, args.outputReportFile + str(i) ), mode='w', encoding='utf8') as oFile:
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
      
      print("   Done!")

      print("Training and testing done in: %fs" % (time() - t0))

