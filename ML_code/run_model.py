#!/usr/bin/env python
# coding: utf-8


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

import os
import sys
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

from degrader_workflows.models.random_forest import train_random_forest

ds_home_path = os.environ.get('DEGRADER_STUDIO_HOME')
if ds_home_path is None:
    print('''Please set environmental variable DEGRADER_STUDIO_HOME
    Example: export DEGRADER_STUDIO_HOME=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio/''')
    quit()

pythonpath = os.environ.get('PYTHONPATH')
if pythonpath is None:
    print('''Please set environmental variable PYTHONPATH as
    export PYTHONPATH="$(realpath ./src)"''')
    quit()
    

if len(sys.argv) - 1 != 2:
    print('''No data files specified. Please specify 1.train-test and 2. prospective data matrix.
    Please place the data file cpontaining in $degrader_studio_path/data/analysis/
    Exiting.
    ''')
    quit()


dataFiles = sys.argv[1:]
#print(dataFiles)
print("The data files to read are ")
print("The train-test matrix ",dataFiles[0])
print("the prospectives matrix ", dataFiles[1])

print()
print("Reading data")
dfX_final = pd.read_csv(ds_home_path + "/data/analysis/ML_features/" + dataFiles[0])
y = dfX_final['Classification_binary_label']
dfX_final.drop('Classification_binary_label', axis=1, inplace=True)

dfX_target_final = pd.read_csv(ds_home_path + "/data/analysis/ML_features/" + dataFiles[1])

#print("dfX_target RDK features shape ",dfRDKitFeature_target.shape)
#print("dfX_target HBF_sol_MD shape ",dfSol_HBF_target.shape)
print("Test-train data shape is",dfX_final.shape)
print("Prospectives data shape is",dfX_target_final.shape)

dfX_final_keys = list(dfX_final.keys())
dfX_target_final_keys = list(dfX_target_final.keys())
set_diff = set(dfX_final_keys) - set(dfX_target_final_keys)
list_diff = list(set_diff)
if len(list_diff) != 0:
    print("the feaytures of the train-test set and the prospective set do not mmatch. Exiting...")
    quit()

print()
print("Scaling data")
dfX_final_scaled = (dfX_final-dfX_final.mean()) / (dfX_final.std(ddof=0))
dfX_target_final_scaled = (dfX_target_final-dfX_final.mean()) / (dfX_final.std(ddof=0))

dfX_final_scaled.fillna(0, inplace=True)
dfX_target_final_scaled.fillna(0, inplace=True)

print("Test-train scaled data shape is ",dfX_final_scaled.shape)
print("Prospectives scaled data shape is ",dfX_target_final_scaled.shape)

dfX_final_scaled.replace([np.inf, -np.inf], 0, inplace=True)
dfX_final_scaled.replace([np.inf, -np.inf], 0, inplace=True)
dfX_target_final_scaled.replace([np.inf, -np.inf], 0, inplace=True)
dfX_target_final_scaled.replace([np.inf, -np.inf], 0, inplace=True)

#dfX_final_scaled.dropna(axis=1, inplace=True)
#dfX_target_final_scaled.dropna(axis=1, inplace=True)

print()
print("Running PCA on test-train data")
pca = PCA(n_components=min(len(dfX_final_scaled), len(dfX_final_scaled.columns)), svd_solver='full')

pca.fit(dfX_final_scaled)
X_pca = pca.transform(dfX_final_scaled)
var_ratio_cumsum_95 = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]+1
pca_components_all = pd.DataFrame(X_pca)
pca_components_imp = pca_components_all.iloc[: , :var_ratio_cumsum_95]

dfFeatures_PCAs = pd.DataFrame(pca.components_,columns=dfX_final_scaled.columns)
#dfFeatures_PCAs.to_excel('/Users/mihirdate/CDD_data/dfFeatures_rdkit_MOE_sol_MD.xlsx')  

print("PCs explaining 95th variance for the test-train set are ",var_ratio_cumsum_95)

#global scores
#global rf_list

print()
print("Training Random Forest model...")
# Run Random Forest
# RandomForest(X, y, n_splits, TEST_SIZE, n_features)
rf = train_random_forest(pca_components_imp, y, 100, 0.3, len(pca_components_imp.columns)) 

#print("printing length of rf_list",len(rf[0]))
#print("printing length of scores",len(rf[1]))

print()
print("Running PCA on prospectives data")
dfX_target_final_scaled_pca = pca.transform(dfX_target_final_scaled)
var_ratio_cumsum_95_target = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]+1

pca_components_target_all = pd.DataFrame(dfX_target_final_scaled_pca)
pca_components_target_imp = pca_components_target_all.iloc[: , :var_ratio_cumsum_95]

#print("printing pca_components_target_all ")
#pca_components_target_all.to_csv('/Users/mihirdate/CDD_data/pca_components_target_all.csv',sep= ',')


print("PCs explaining 95th variance for prospectives are ",var_ratio_cumsum_95_target)
print()
print("Running predictions...")
#prediction = [rf.predict(pca_components_target_imp) for rf in rf_list]
prediction = [rf.predict(pca_components_target_imp) for rf in rf[0]]

dfPrediction_proba = pd.DataFrame()
i = 0
#for rf in rf_list:
for rf in rf[0]:
    prediction_proba_i = []
    prediction = rf.predict(pca_components_target_imp)
    prediction_proba_i.append(rf.predict_proba(pca_components_target_imp))
    i = i + 1
    arr_i = np.array(prediction_proba_i)
    arr_i = arr_i[0, :, :]
    dfPrediction_proba_i = pd.DataFrame(arr_i)
    dfPrediction_proba_i.columns =['P(class 1)', 'P(class 2)']
    dfPrediction_proba_d = dfPrediction_proba_i.drop(['P(class 1)'], axis=1)
    dfPrediction_proba = pd.concat([dfPrediction_proba,dfPrediction_proba_d], axis = 1)
    
#print(dfPrediction_proba.shape)


dfPrediction_proba_mean = dfPrediction_proba.mean(axis=1)
dfPrediction_proba_stdev = dfPrediction_proba.std(axis=1)
dfPredictions = pd.concat([dfPrediction_proba_mean,dfPrediction_proba_stdev], axis = 1)
dfPredictions.columns = ['Mean', 'STDEV']


pd.options.display.float_format = '{:,.4f}'.format
dfPredictions.index = list(dfX_target_final['Molecule Name'])
dfPredictions.reset_index(inplace=True)
dfPredictions = dfPredictions.rename(columns = {'index':'Molecule Name'})
dfPredictions

print()
out_predictions = ds_home_path + "/data/analysis/ML_features/"  + "RF_predictions.csv"
print("Saving prediction files", out_predictions)
dfPredictions.to_csv(out_predictions, sep=',', float_format='%.4f')
print()
print("DONE!")
