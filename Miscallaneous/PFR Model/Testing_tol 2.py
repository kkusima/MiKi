import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import *

MKM = MKModel('Atomic.csv','Stoich.csv','Param.csv')
MKM.set_rxnconditions(Pr=[(1.0e-4*1.0e-5), (1.0e-4*0.1), 0])
MKM_init_coverages = np.empty([5,4])