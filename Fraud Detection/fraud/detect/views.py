import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from django.shortcuts import render,HttpResponse
import sys
import os

ml_model_path = "C:\\Users\\sreek\\Downloads"
sys.path.append(ml_model_path)

# Now you can import ml_model.py
from fradetmod1 import predict

def home(request):
    return render(request,'home.html')
def details(request):
    context={}
    if request.method=="POST":
        fname=request.POST['fullname']
        words=len(fname.split())
        nums1=0
        for i in fname:
            if ord(i)<65:
                nums1+=1
        nums2=0
        uname=request.POST['username']
        for i in uname:
            if ord(i)<65:
                nums2+=1
        followers=request.POST['followers']
        desc=request.POST['description']
        follows=request.POST['follows']
        posts=request.POST['posts']
        prof=request.POST['profile']
        if prof=='yes':
            prof=1
        else:
            prof=0
        priv=request.POST['private']
        if priv=='yes':
            priv=1
        else:
            priv=0
        res=predict([prof,nums2/len(uname),words,nums1/len(fname),int(fname==uname),len(desc),0,priv,posts,followers,follows])
        if res==0:
            context={'success':True,'grand':True}
        else:
            context={'success':True}
    return render(request,'details.html',context)
