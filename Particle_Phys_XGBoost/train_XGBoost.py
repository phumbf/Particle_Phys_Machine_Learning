'''
Python script to train the XGBoost classifier. The classifier is trained to separate 
signal (represented with Monte Carlo simulation in the training data) from so-called 
combinatorial background which is caused by the accidental reconstruction of random combinations
of particle tracks. The combinatorial background proxy is taken from the data itself

Different variables are used to train the classifier which exploit Kinematic and topological
behaviour of the particle decay

Details regarding the training, motivation and tests for overtraining can be found in Chapter 5 
of my thesis, https://cds.cern.ch/record/2756320/files/CERN-THESIS-2020-306.pdf
'''

from rep.estimators import XGBoostClassifier
from rep.estimators import SklearnClassifier # This must be imported first
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams.update({'figure.autolayout': True})
from rep.metaml import ClassifiersFactory
from rep.report.metrics import RocAuc
import numpy as np
import pandas as pd
from root_pandas import read_root
from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from ROOT import TFile, TLorentzVector, TVector3, RooRealVar, RooGaussian, RooDataSet, RooExponential
from ROOT import RooArgSet, TCanvas, RooAddPdf, RooPolynomial, RooArgList, RooCBShape, RooFit
from ROOT import TFile, TH1D, TCanvas
import sys
import cPickle as pickle

# The variables defined in the data and Monte Carlo simulation which separate signal from background
vs = ['B0_DIRA_OWNPV',
'B0_ENDVERTEX_CHI2',
'B0_IPCHI2_OWNPV',
'Jpsi_IPCHI2_OWNPV',
'Min_PT',
'B0_VTXISODCHI2TWOTRACK_PROB_2DOF',
'B0_TAU',
'B0_PT',
'minIPCHI2_piplus_piminus',
'minIPCHI2_muplus_muminus',
'minPT_muplus_muminus',
'absmuplus_PT_minus_muminus_PT'
]

print 'Usage: python TRAIN_XGBOOST.py SignalFile BackgroundFile'

#the signal input file
signalfile = sys.argv[2]
#the background input file
print 'signalfile is', signalfile
backgroundfile = sys.argv[1]
print 'backgroundfile is', backgroundfile

# Add any other variables present in the command line
vs += sys.argv[3:]

#Will be cutting on the B0_M for the 'data'
extraVars = ['B0_M']

#Will be cutting on the BKGCAT for the MC

# Read ROOT files into Pandas data frame format (http://pandas.pydata.org/pandas-docs/stable/10min.html)
# The data file contains all data, including the signal region and upper sideband we want to train on, but will deal with this later

data = read_root(backgroundfile, 'DecayTree', columns = vs + extraVars)
#DATA NOT BEING WEIGHTED SO SET WEIGHT TO ONE
data['mva_weight'] = 1

#Our "signal" is MC, and add the extravars 
#ADD THE WEIGHTS
mc = read_root(signalfile, 'DecayTree', columns = vs + extraVars + ['mva_weight'])

#Our "background" is that from the upper sideband
data = data.query('B0_M > 5800')

# Create a 'SIGNAL' column in each data frame, which has the value 0 for data and 1 for MC
data['SIGNAL'] = 0
mc['SIGNAL'] = 1

# Print how many MC and data events we have for a sanity check
print len(mc), len(data)

# Merge the total MC and data samples into one data frame (merged)
merged = pd.concat((mc, data))

# Split the merged data into a training (75%) and test (25%) sample,
# where X is all the variables for the classifier apart from the true SIGNAL category, and y is the true SIGNAL category
# It is VERY important that this split is randomised at each iteration of an optimisation step (for variables, classifier hyperparameters, etc)
# to avoid overtraining, although the random_state can be set for reproducibility

X_train, X_test, y_train, y_test = train_test_split(merged, merged['SIGNAL'], test_size=0.25)#, random_state=42)

#APPLYING WEIGHTS
weights_train = pd.Series(X_train['mva_weight'])
X_train = X_train.drop('mva_weight',1)

weights_test = pd.Series(X_test['mva_weight'])
X_test = X_test.drop('mva_weight',1)

# We now need to remove the signal region from the data, as we want to use this to calculate signal and background
# yields, but obviously don't want to train on it (as its currently labelled as background anyway, SIGNAL == 0)

X_train = X_train.drop('SIGNAL', 1)
X_test = X_test.drop('SIGNAL', 1)

##EXPORTING PARTITIONS for usage elsewhere
np.savetxt(r'./X_train.txt',X_train.values,fmt='%10.6f')
np.savetxt(r'./X_test.txt',X_test.values,fmt='%10.6f')
np.savetxt(r'./y_train.txt',y_train.values,fmt='%10.6f')
np.savetxt(r'./y_test.txt',y_test.values,fmt='%10.6f')

# Drop all other variables we might have acquired
for exV in extraVars:
    X_train = X_train.drop(exV, 1)
    X_test = X_test.drop(exV, 1)

# The dataframes for training and testing should now only have the variables we want to train the classifier on
# so keep track of these (this is shorthand for the list of variable names in the data frame)

X_trainCols =  list(X_train)
X_testCols =  list(X_test)

# The variables have very different scales, so to help the classifier, re-scale these to be more comparable
# It's important to ensure that both the training AND the evaluation/validation data are transformed in the same
# way
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler so that this can be re-used when applying the training
with open('scalerpipimumu.pkl','wb') as fid:
    pickle.dump(scaler, fid)

# This procedure renames the columns, so re-insert these (this isn't neccessary and is probably wrong)
X_train = pd.DataFrame(X_train, columns = X_trainCols)
X_test = pd.DataFrame(X_test, columns = X_testCols)

# Make a map of classifier names to classifier constructors, with some vaguely sensible arguments
print 'Training with:'

for v in X_trainCols:
    print v

classifiers = ClassifiersFactory()
classifiers.add_classifier('xgboost', XGBoostClassifier(max_depth = 4, n_estimators = 100, eta = 0.3, nthreads = 4))

classifiers.fit(X_train, y_train,sample_weight=weights_train)

trainPred = classifiers.predict_proba(X_train)
testPred = classifiers.predict_proba(X_test)

#Save output of trained classifier
with open('xgboostClassifierpipimumu.pkl','wb') as fid:
    pickle.dump(classifiers, fid)

#Print the ROC curve Area under the curve - gauge of performance
print 'ROC AUC:'
for key in trainPred:
        print key, 'Train:', roc_auc_score(y_train, trainPred[key][:, 1]), 'Test:', roc_auc_score(y_test, testPred[key][:, 1])

report = classifiers.test_on(X_test, y_test)

report.feature_importance().plot()
plt.savefig('featureImportances.pdf')
plt.clf()

learning_curve = report.learning_curve(RocAuc(), metric_label='ROC AUC', steps=1)
learning_curve.plot()
plt.savefig('learningCurve.pdf')
plt.clf()

report.features_correlation_matrix_by_class(features=X_trainCols).plot(new_plot=True, show_legend=False, figsize=(15, 5))
plt.savefig('correlations.pdf')
plt.clf()

plt.subplot(1, 2, 1)
report.roc(physics_notion=True).plot(new_plot=True, ylim=(0.5, 1), xlim=(0.5, 1), figsize=(12 , 8))
plt.savefig('rocs.pdf')
plt.clf()

plt.subplot(1, 2, 1)
report.roc(physics_notion=True).plot(new_plot=True, ylim=(0.97, 1), xlim=(0.5, 1), figsize=(12 , 8))
plt.savefig('rocszoom1.pdf')
plt.clf()

