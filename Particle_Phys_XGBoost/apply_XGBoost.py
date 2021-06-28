'''
Python script to apply the now trained XGBoost classifier.
'''

from rep.estimators import XGBoostClassifier, TMVAClassifier, SklearnClassifier # This must be imported first
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from root_pandas import read_root
import sys
import cPickle as pickle
from variables import vs 

classifier1 = pickle.load(open('./xgboostClassifierpipimumu.pkl', 'rb'))
scaler1 = pickle.load(open('./scalerpipimumu.pkl', 'rb'))

def getDataset(fileName, treeName, weighted):
    data = read_root(fileName, treeName, columns = vs )
    
    data['IDX'] =  np.arange(0, len(data))
    
    print 'Made dataset with variables:', list(data)

    return data.drop('IDX', 1), data['IDX'].as_matrix()

def applyClassifier(fileToClassify, treeToClassify, isWeighted):

    data, ordering = getDataset(fileToClassify, treeToClassify, isWeighted)
    
    toClassify = data

    toClassifyNames = list(toClassify)

    toClassify = scaler1.transform(toClassify, copy = True)

    toClassify = pd.DataFrame(toClassify, columns = toClassifyNames)
    
    pred1 = pd.DataFrame(classifier1['xgboost'].predict_proba(toClassify)[:,1], columns = ['PRED'])

    pred1['IDX'] = ordering

    predsOrdered = pred1.sort_values('IDX')['PRED'].as_matrix()
    
    return predsOrdered

if __name__ == '__main__':

    addroot = '.root'
    fileToClassify = sys.argv[1]+addroot if len(sys.argv) > 2 else None
    treeToClassify = sys.argv[2] if len(sys.argv) > 2 else None
    isWeighted = int(sys.argv[4]) if len(sys.argv) > 3 else 0

    if fileToClassify is None or treeToClassify is None:
        exit(1)

    preds = applyClassifier(fileToClassify, treeToClassify, isWeighted)

    with open(fileToClassify.split('/')[-1][:-5] + '-' + treeToClassify + '-Classified.txt', 'w') as f:
        for i in xrange(len(preds)):
            f.write(str(preds[i]) + '\n')

