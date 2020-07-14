#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:06:13 2020

@author: alexshepherd
"""

import unittest
from MedRecommend import MedRecommend

class TestMedRecommend(unittest.TestCase):
    
    def test_forward(self):
        # testing if forward function returns
        # 1) returns appropriate shape
        # 2) returns correctly computed values
        
        m = MedRecommend()
        m.X = np.arange(8).reshape((4,2))
        m.theta = np.arange(12).reshape((4,3))
        
        true_result = np.array([[ 84, 102],
                                [ 96, 118],
                                [108, 134]])
        
        self.assertEqual(m.forward(), true_result)
        self.assertEqual(m.forward().shape, (3,2))
        
        
    def test_backward(self):
        
        m = MedRecommend()
        m.X = np.arange(8).reshape((4,2))
        m.theta = np.arange(12).reshape((4,3))
        m.y = np.array([[0.1, np.nan],
                        [np.nan, 0.3],
                        [0.9, 0.4]])
        
        dE_db, dE_dW, dE_dTheta = m.backward(debug = True)
        
        
    
    def test_MSE(self):
        
    
if __name__ == "__main__":
    unittest.main() 

    

    