# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

with open("data\coeff_6j.json","r") as f:
    matrix =json.load(f)

mat = np.array(matrix)

def bla(i,v):
    j_range=np.array([0,1,2,3,4,5,6,7,8,9])
    return np.sum((j_range*2+1)*mat[0+j_range,2]*v) -np.sum((j_range*2+1)*mat[i+j_range,2]*v)

v = np.array([0,1718,1484,1935,2359,2042,2529,2263,2653,820])

v_ph =  [0.0,3048.034558910189,1587.600833503105,3810.22027903737,2207.5925704900233,2081.445622386741,2442.459971393591,1813.5643644876611,2530.317414007536,  2576.9788836926123]

for i in range(10):
    print("v_ph[",i,"] = ",bla(i,v_ph))