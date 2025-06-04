# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:55:48 2025

@author: patri
"""

def set_pvalue(pvalue):
    if pvalue < 0.0001:
        starpval = "****"
    elif pvalue < 0.001:
        starpval = "***"
    elif pvalue < 0.01:
        starpval = "**"
    elif pvalue < 0.05:
        starpval = "*"
    else:
        starpval = "ns"
    return starpval