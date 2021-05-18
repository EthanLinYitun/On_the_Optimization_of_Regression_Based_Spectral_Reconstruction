# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:35:29 2020

@author: Yi-Tun Lin
"""

import numpy as np
from copy import copy
from data import *
import time
from pandas import read_csv
from numpy.linalg import inv

def mrae(gt, rec):
    return np.mean(np.abs(gt['spec'] - rec['spec']) / gt['spec'], axis=1)

def ange(gt, rec):
    inner_product = np.sum(gt['spec'] * rec['spec'], axis=1 )   # DIM_DATA,
    normalized_inner_product = inner_product / np.linalg.norm(gt['spec'], axis=1) / np.linalg.norm(rec['spec'], axis=1)
    
    return np.arccos(normalized_inner_product) * 180 / np.pi

def rmse(gt, rec):
    return np.sqrt(np.mean((gt['spec'] - rec['spec'])**2, axis=1))