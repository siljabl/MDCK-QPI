import numpy as np  # numerics
import math as math  # math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import random
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from catboost import (CatBoostRegressor, Pool, sum_models,to_classifier)
from skimage.feature import multiscale_basic_features


class MlModel:
    """
    Really just a wrapper to keep the feature computing function and
    the classifier at the same place, and pickled at the same time.
    """

    def __init__(self,clf,**kwargs):
        self.clf = clf(**kwargs)
        self.thresholds = []
        self.fpr = []
        self.tpr = []
        self.th = []

    def setComputeFeatures(self,fctn):
        """
        Set the function computing the features. Receives an image.
        """
        self.computeFeatures = fctn

    def predict_proba(self, arr):
        probs = []
        if (np.array(arr).ndim ==3): # stack
            for i in arr:
                features = (self.computeFeatures(i))
                prob = self.clf.predict_proba(features)
                prob = prob.reshape((len(arr[0]),len(arr[0][0]),-1))
                probs.append(prob)
        return np.array(probs)

    def predict(self,arr):
        probs = self.predict_proba(arr)
        mask = []
        for p in probs:
            for t in range(len(self.thresholds)):
                if (t==0):
                    mask.append(np.array(p[:,:,t+1]>self.thresholds[t],dtype=np.uint8))
                else:
                    mask[-1][p[:,:,t+1]>self.thresholds[t]] = t+1
        return np.array(mask)
            


    
def detect_bioM_FL_MlMod_lowRam(stack, model, return_prob=False):
    mask = []
    prob = []
    N=0
    thresholds = model.thresholds
    for i in stack:
        features = (model.computeFeatures(i))
        prob_i = model.clf.predict_proba(features)
        prob_i = prob_i.reshape((len(stack[0]),len(stack[0,0]),-1))
        prob.append(np.array(prob_i, dtype=np.float16))
        #print(prob.shape)
        for t in range(len(thresholds)):
            
            prob_i_tmp = prob_i[:,:,t+1]
            if (t==0):
                mask.append(np.array(prob_i_tmp>thresholds[t],dtype=np.uint8))
            else:
                mask[-1][prob_i_tmp>thresholds[t]]=t+1
        #print(mask[-1].shape)
        N = N+1
    mask = np.array(mask)
    prob = np.array(prob)
    
    if return_prob:
        return mask, prob

    else:
        return mask