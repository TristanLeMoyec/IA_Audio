# -*- coding: utf-8 -*-
"""
Main script

In this script you should create the main structure of your speaker detector

Created on Mon Oct 24 20:40:40 2022

@author: ValBaron10
"""

from joblib import load

# Get input signals that are different from the ones used during training



# Compute the features on these signals


FEATURES = # Compute the features
    
# Load the scaler and SVM model to test the class of your source
scaler = load("SCALER")
model = load("SVM_MODEL")


# Get the class of your source
prediction = model.predict(FEATURES)


# Analyze the results
    
