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


v_ph =  [0.0,
835.1899683945763,
-3185.4111714357314,
-3021.2070076849946,
-3246.1314949081807,
-3572.23027100757,
-3798.6444745360795,
-3611.657207870087,
-3703.6451813795948,
-2945.762378376896]

v_pp = [ 0.0,
2347.776417580249,
1209.3173029705035,
2861.603516028554,
1970.508805778896,
2628.128782861728,
2908.8628518770433,
2193.8015428222006,
3146.4348697177725,
821.1315181113247]

for i in range(10):
    print("v_ph[",i,"] = ",bla(i,v_ph))