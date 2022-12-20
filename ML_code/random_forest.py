from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import matplotlib

import os
import csv
import numpy as np
import pandas as pd
import copy as cp
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
# If you're working in Jupyter Notebook, include the following so that plots will display:
#get_ipython().run_line_magic('matplotlib', 'inline')

#from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, ReducedGraphs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors

import scipy.cluster.hierarchy as sch

ds_home_path = os.environ.get('DEGRADER_STUDIO_HOME')

# The following function runs Random Forest classifier by Stratified Shulled Split of train-test set.
# n_splits is number of splits to make, producing n_splits of cross validation
# n_features is number of featurers to show in feature importance plot
def train_random_forest(X, y, n_splits, TEST_SIZE, n_features):
    sss = StratifiedShuffleSplit(n_splits, test_size=TEST_SIZE, random_state=0)
    sss.get_n_splits(X, y)
    scores = []
    imp = []
    rf_list = []

    a = np.array([[0,0],[0,0]])
    # Random forest classification model
    rf = RandomForestClassifier(random_state = 0)
    probs=[]
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf = RandomForestClassifier(random_state = 0)
        rf_list.append(rf)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        scores.append(accuracy_score(y_test, pred))
        imp.append(rf.feature_importances_)
        confMatrix = confusion_matrix(y_test, pred)
        a = a + confMatrix

    # get accuracy of each prediction

    scores_mean = np.average(scores)
    scores_stdev = np.std(scores)
    scores_var = np.var(scores)

    print('scores average')
    print(scores_mean)
    print('scores stdev')
    print(scores_stdev)

    feaImp = np.array(imp)
    feaImp_mean = np.mean(feaImp, axis = 0)
    feaImp_var = np.var(feaImp, axis = 0)
    feaImp_std = np.std(feaImp, axis = 0)

    print('confusion matrix')
    confMatrix_avg = a/100
    print(confMatrix_avg)

    ds_home_path = os.environ.get('DEGRADER_STUDIO_HOME')

    # plot confusion matrix
    matplotlib.use('SVG')
    ax = sns.heatmap(confMatrix_avg, annot=True, cmap='Blues', fmt='.3g')
    #title = "Confusion Matrix\n" + "Mean score"  $(scores_mean) + "\n" + "stdev" + $(scores_stdev) + "\n"
    #ax.set_title('Confusion Matrix\n Mean score $(scores_mean)\n stdev $(scores_stdev)\n')
    ax.set_title('\n Confusion Matrix\n \n')
    ax.set_xlabel('\nPredicted degradation class')
    ax.set_ylabel('True degradation class')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['1-non-degrader','2-degrader'])
    ax.yaxis.set_ticklabels(['1-non-degrader','2-degrader'])

    ## Display the visualization of the Confusion Matrix.
    #plt.show()
    out_conf_matrix = ds_home_path + "/data/analysis/ML_features/"  + "confusion_matrix.svg"
    ax.figure.savefig(out_conf_matrix, bbox_inches='tight', facecolor = 'white')
    #plt.clf()

    # from feature importance array, sort top 10 features and make a dataframe for plotting
    dffeaImp_mean = pd.DataFrame(feaImp_mean)
    dffeaImp_std = pd.DataFrame(feaImp_std)
    dfX_header = pd.DataFrame(list(X.columns))
    e = pd.concat([dfX_header,dffeaImp_mean,dffeaImp_std], axis = 1)
    e.columns =['Feature', 'Importance', 'STDEV']
    f = e.nlargest(n=n_features, columns=['Importance'])

    print()
    print("Feature importance")
    print(f)
    # plot horizontal bar graph with steev error bars
    # https://www.easytweaks.com/bar-plot-python-pandas-dataframe-example/
    fig = f.plot(kind='barh', x='Feature', xerr='STDEV', title='Important principle components')
    out_feature_imp = ds_home_path + "/data/analysis/ML_features/"  + "feature_importance.svg"
    fig.figure.savefig(out_feature_imp, bbox_inches='tight', facecolor = 'white')
