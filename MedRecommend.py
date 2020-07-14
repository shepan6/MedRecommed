#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:44:59 2020

@author: alexshepherd
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

class MedRecommend:
    """
    The goal of this medical advice recommendation system is to rate which 
    medical advice should be given to the following patients.
    
    Recommendation system based on Andrew Ng's lecture series
    https://www.youtube.com/watch?v=9siFuMMHNIA
    
    Additional helpful source:
    https://towardsdatascience.com/recommender-systems-in-python-from-scratch-643c8fc4f704
    """
    
    def __init__(self, lr = 0.05, lambda_ = 0.1):
        
        # latent features represents features among patients
        # loading fixed data csv files
        
        # shape (advice, illnesses) 
        self.advice_features = pd.read_csv(
            "medical_advice_features.csv", index_col=0)
        
        # shape (advice, patients)
        self.advice_recommend = pd.read_csv(
            "medical_advice_helpfulness.csv", index_col=0)
        
        # shape: (illness, advice)
        self.X = self.advice_features.values.T
        # Normalising values
        self.X = self.X / np.max(self.X)
        
        # adding intercept X as each patient is a linear model.
        self.X = np.vstack((np.atleast_2d(
            np.ones(self.X.shape[0])), self.X))
        
        # Initializing theta which represents patients preferences
        # shape: (patients, illnesses + intercept)
        # add intercept to self.theta
        
        """ in reality self.theta shoudl contain two matrices,
        1. Relationship between user and latent features
        2. Relationship between advice and latent features.
        
        Currently, recommender system is simplified, suh that (2) is given
        and we need to optimise (1) such that min E(pred, true).
        """
        self.theta = np.random.randn(self.advice_recommend.shape[1],
                                      self.advice_features.shape[1] + 1)
        
        #self.theta[:,0] = 0
        
        # learning rate
        self.lr = lr
        # regularisation constant
        self.lambda_ = lambda_
        
        # fill 0s as NaN values
        # NaN values will denote where a patient has not rated that advice.
        self.y = self.advice_recommend#.replace(0,np.nan)
        
        self.y_idx = np.where(~self.y.isna())
        self.y  = self.y.values
    
    
    def content_based(self):
        """
        content-based recommender systems aim to select top X advice which are
        most similar to a particular advice.

        Returns
        -------
        None.

        """
        pass
        
    def forward(self):
        """
        
        Forward propagation to drive prediction.

        Returns
        -------
        y : Matrix (self.theta.shape[0], self.advice_features.shape[1])
        (advice, patient)
        Model predictions

        """
        
        y = np.dot(self.theta.T, self.X) 
        
        return y
        
    
    
    def backward(self, debug = False):
        """
        
        Back propagation with theta/parameter updates
        
        Parameters
        ----------
        debug : Boolean 
        this will return main components involved in the function to
        check if the components of the function are correct

        Returns
        -------
        None.

        """
        
        
        
        # shape should be (1, self.theta.shape[1])
        dE_db = np.sum(
            np.dot((self.forward() - self.y), self.X)[self.y_idx]
            , axis = 1)
        dE_dW = dE_db + (self.lambda_ * self.theta)
        
        dE_dW = dE_dW[:,1:]
        
        dE_dTheta = np.hstack((dE_dW, np.atleast_2d(dE_db).T))
        self.theta -= (dE_dTheta * self.lr)
        
        if debug:
            return dE_db, dE_dW, dE_dTheta
        
    
    def MSE(self, ypred, ytrue, debug = False):
        """
        Error function: Mean Squared Error. L2 regularisation is applied.

        Returns
        -------
        mse : Float.
        Mean squared error between model predictions and true values.

        """
        
        try:
            total_deviation = np.sum(
                np.square(
                    ypred[self.y_idx] - ytrue[self.y_idx])
                )
        except:
            total_deviation = np.sum(np.square(ypred - ytrue))
                
        reg_constant = (0.5 * self.lambda_) * np.sum(
            np.square(self.theta[:-1]))
        
        
        mse = 0.5 * total_deviation + reg_constant
        mse = np.mean(mse)
        
        if debug:
            return [total_deviation, reg_constant, mse]
        
        return mse
        
        
    def train(self, epochs = 100):
        
        self.metrics = []
        
        for e in tqdm(range(epochs)):
            self.backward()
            print("Epoch {}: MSE = {}".format(e+1, 
                                              self.MSE(self.forward(),self.y)))
        
        
        
m = MedRecommend()
m.train()

