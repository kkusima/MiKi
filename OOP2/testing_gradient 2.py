#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:33:06 2022

@author: klkusima
"""

import numpy as np
import numdifftools as nd
fun = lambda x: np.sum(x**3)
dfun = nd.Gradient(fun)
check = nd.Derivative(fun)
print(dfun([1,2,3]), [ 3.,  12.,  27.])
print([check(1),check(2),check(3)], [ 3.,  12.,  27.])

print('checking:')
print(check(1))
print(dfun(1))