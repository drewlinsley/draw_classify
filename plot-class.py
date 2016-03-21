#!/usr/bin/env python 

from __future__ import division, print_function

import logging
import argparse
import numpy as np
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle

from pandas import DataFrame

from mpl_toolkits.mplot3d import Axes3D

from blocks.main_loop import MainLoop
from blocks.log.log import TrainingLogBase


FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


model_file = 'new_class_test-20160320-145543/new_class_test_log'
with open(model_file,"rb") as f:
    model = pickle.load(f)



if isinstance(model, MainLoop):
    log = model.log
elif isinstance(model, TrainingLogBase):
    log = model
else: 
    print("Don't know how to handle unpickled %s" % type(model))
    exit(1)

df = DataFrame.from_dict(log, orient='index')
#df = df.iloc[[0]+log.status._epoch_ends]
names = list(df)

nsp = int(np.ceil(np.sqrt(len(df.columns))))
for num in range(3,len(names)):
    plt.subplot(nsp,nsp,num+1)
    plt.plot(df[names[num]])
    plt.title(names[num])
plt.show()

