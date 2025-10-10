import matplotlib.pyplot as plt  # plotting
import numpy as np  # numerics
import math as math  # math
#import skvideo.io  # video opener
import skimage.io as io
import skimage.io
from tqdm import tqdm  # fancy progress bar
import os  as os # folder and file manipulation
import gc # garbage collector
import time
import sys
import colorama as cl
from skimage import transform
from skimage.feature import multiscale_basic_features

def compute_features(arr):
    features = multiscale_basic_features(arr).reshape(len(arr)*len(arr[0]),-1)
    """
    features1 = multiscale_basic_features(arr, sigma_max = 1024,edges=False,texture=False).reshape(len(arr)*len(arr[0]),-1)
    features2 = multiscale_basic_features(arr, sigma_max = 32,edges=True,texture=True, intensity=False).reshape(len(arr)*len(arr[0]),-1)
    features = np.concatenate((features1,features2),axis=-1)
    #features.reshape(len(arr)*len(arr[0]),-1)
    """
    return features

def compute_features_no_intensity(arr):
    features = multiscale_basic_features(arr,intensity=False).reshape(len(arr)*len(arr[0]),-1)
    return features

def compute_features_extended_no_intensity(arr):
    features = multiscale_basic_features(arr,intensity=False,sigma_max=512).reshape(len(arr)*len(arr[0]),-1)
    return features

def compute_features_extended(arr):
    features = multiscale_basic_features(arr,sigma_min=0.1,sigma_max = 512).reshape(len(arr)*len(arr[0]),-1)
    return features

def compute_features_shrink(arr):
    features = multiscale_basic_features(arr,sigma_min=0.1).reshape(len(arr)*len(arr[0]),-1)
    return features

def compute_features_st(arr):
    ret = []
    for i in arr:
        ret.append(multiscale_basic_features(i).reshape(len(i)*len(i[0]),-1))
    return ret


def detect_bioM_ML(stack, clf, threshold):
    features = []
    for i in stack:
        features.append(compute_features(i))
    features = np.concatenate(features)
    prob = clf.predict_proba(features)
    prob = prob.reshape((len(stack),len(stack[0]),len(stack[0,0]),-1))
    prob = prob[:,:,:,1]
    prob = prob>threshold
    return prob

def detect_bioM_ML_lowRam(stack, clf, threshold):
    mask = []
    for i in stack:
        features = (compute_features(i))
        prob = clf.predict_proba(features)
        prob = prob.reshape((len(stack[0]),len(stack[0,0]),-1))
        #print(prob.shape)
        prob = prob[:,:,1]
        mask.append(prob>threshold)
        #print(mask[-1].shape)
    mask = np.array(mask)
    
    return mask

def detect_bioM_FL_ML_lowRam(stack, clf, thresholds):
    mask = []
    for i in stack:
        features = (compute_features(i))
        prob = clf.predict_proba(features)
        prob = prob.reshape((len(stack[0]),len(stack[0,0]),-1))
        #print(prob.shape)
        for t in range(len(thresholds)):
            
            prob_tamp = prob[:,:,t+1]
            if (t==0):
                mask.append(np.array(prob_tamp>thresholds[t],dtype=np.uint8))
            else:
                mask[-1][prob_tamp>thresholds[t]]=t+1
        #print(mask[-1].shape)
    mask = np.array(mask)
    
    return mask