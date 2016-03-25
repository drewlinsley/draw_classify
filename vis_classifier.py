from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import ipdb
import time
import cPickle as pickle

from argparse import ArgumentParser
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model

try:
    from blocks.extras import Plot
except ImportError:
    pass


import draw.datasets as datasets
from draw.draw import *
from draw.samplecheckpoint import SampleCheckpoint
from draw.partsonlycheckpoint import PartsOnlyCheckpoint

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE, MDS
from sklearn import cross_validation


from PIL import Image

import unittest

from draw.attention import *

import SVRT_analysis_helper_functions 
from draw.datasets import package_sketch_images



dataset = 'sketch_uint8_shuffle'
data_dir = '/Users/drewlinsley/Desktop/res_results_problem_4'

#Get shuffled data
test_X, train_X, test_y, train_y = package_sketch_images.import_sketch(data_dir)
image_size = (int(np.sqrt(test_X.shape[1])),int(np.sqrt(test_X.shape[1])))
channels = 1
target_categories = np.unique(train_y).shape[0]

#Initialize vars and theano function
num_estimates = 20
num_training = 6 #per evaluation
num_testing = 1 #per evaluation
batch_size = num_estimates * num_training + num_estimates * num_testing #get a bunch just to be sure

rows = 10
cols = 20
N_iter = 64;

#Load old model
#model_file = 'new_test-20160313-125114/new_test_model'
#model_file = 'new_class_test-20160321-160949/new_class_test_model'
model_file = 'new_class_test-16_it/new_class_test_model'
with open(model_file,"rb") as f:
    model = pickle.load(f)
draw = model.get_top_bricks()[0]

#Get attention model extractions
images = T.matrix('images')

probs, h_enc, c_enc, center_y, center_x, delta = draw.reconstruct(images)
do_sample = theano.function(inputs=[images],outputs=[probs, h_enc, c_enc, center_y, center_x, delta],allow_input_downcast=True)

#rec,kl,cy,cx = draw.reconstruct(images)
#do_sample = theano.function(inputs=[images],outputs=[cy,cx],allow_input_downcast=True)
tprobs, th_enc, tc_enc, tcenter_y, tcenter_x, tdelta = do_sample(test_X)
correct_ims = tprobs.argmax(axis=1) == test_y.flatten()
acc = np.sum(correct_ims) / test_X.shape[0]
incorrect_ims = ~correct_ims

#Plot everything
from matplotlib.collections import LineCollection
from SVRT_analysis_helper_functions import scale_range
import matplotlib as mpl
mymap = plt.get_cmap("Reds")
# get the colors from the color map
test_image_res = test_X#[0:batch_size,:].reshape(batch_size,image_size[0],image_size[1])

#img_grid(test_image.reshape(batch_size,1,image_size[0],image_size[1]),rows,cols)
from SVRT_analysis_helper_functions import plot_ims
#Make a folder
subdir = "classification_trace_errors"
if not os.path.exists(subdir):
    os.makedirs(subdir)
plot_ims(test_image_res,tcenter_x,tcenter_y,tdelta,incorrect_ims,out_dir)
#Make a folder
subdir = "classification_trace_correct"
if not os.path.exists(subdir):
    os.makedirs(subdir)
plot_ims(test_image_res,tcenter_x,tcenter_y,tdelta,correct_ims,out_dir)








#Train control classifiers with x/y/deltas, test it on those same values. Compare this to features extracted from the same images
from sklearn import svm
from sklearn import metrics
from sklearn import decomposition 
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

#Extract attention features
h_enc, y_arr, x_arr, delta = atten_features(N_iter,batch_size,image_size,test_image,do_sample)
tr_enc, tr_y_arr, tr_x_arr, tr_delta = atten_features(N_iter,batch_size,image_size,train_image,do_sample)
tr_rho, tr_phi = conv_coors(tr_x_arr,tr_y_arr)
te_rho, te_phi = conv_coors(x_arr,y_arr)
#Globally normalize everything (preserve shape but rescale)
sd_tr_rho = (tr_rho - np.mean(tr_rho)) / np.std(tr_rho)
sd_tr_phi = (tr_phi - np.mean(tr_phi)) / np.std(tr_phi)
sd_tr_delta = (tr_delta - np.mean(tr_delta)) / np.std(tr_delta)
sd_tr_enc = (tr_enc - np.mean(tr_enc)) / np.std(tr_enc)
te_rho, te_phi = conv_coors(x_arr,y_arr)
sd_te_rho = (te_rho - np.mean(te_rho)) / np.std(te_rho)
sd_te_phi = (te_phi - np.mean(te_phi)) / np.std(te_phi)
sd_te_delta = (delta - np.mean(delta)) / np.std(delta)
sd_te_enc = (h_enc - np.mean(h_enc)) / np.std(h_enc)

tr_data = np.transpose(np.concatenate([sd_tr_rho,sd_tr_phi,sd_tr_delta,sd_tr_enc]))
te_data = np.transpose(np.concatenate([sd_te_rho,sd_te_phi,sd_te_delta,sd_te_enc]))

clf = linear_model.BayesianRidge()
clf.fit(tr_data,train_labels.ravel())


clf = linear_model.RidgeCV(alphas=[1e-4,1e-3, 0.1, 1.0, 10.0, 100])
clf.fit(tr_data,train_labels.ravel())
clf.coef_
clf.intercept_

in_labels = train_labels.ravel()



#Fit SVM on attention features
#clf = svm.SVC()
#clf.fit(tr_data, train_labels.ravel())  
#y_hat = clf.predict(te_data)
#acc = metrics.accuracy_score(test_labels.ravel(),y_hat)
#y_score = clf.fit(tr_data, train_labels.ravel()).decision_function(te_data)
acc = svm_trainer(tr_data,te_data,train_labels.ravel(),test_labels.ravel(),
	num_estimates,num_training,num_testing)

#Extract HOG features
from skimage.feature import hog
from skimage import data, color, exposure
orientations = 4
pixels_per_cell=[4, 4]
tr_hf = hog_features(train_image,orientations,pixels_per_cell)
te_hf = hog_features(test_image,orientations,pixels_per_cell)


#Fit SVM on HOG features
clf = svm.SVC()
clf.fit(tr_hf, train_labels.ravel())  
hog_y_hat = clf.predict(te_hf)
#hog_acc = metrics.accuracy_score(test_labels.ravel(),hog_y_hat)
hog_y_score = clf.fit(tr_hf, train_labels.ravel()).decision_function(te_hf)
hog_acc = svm_trainer(tr_hf,te_hf,train_labels.ravel(),test_labels.ravel(),
	num_estimates,num_training,num_testing)


#Fit SVM on Pixel data
px_tr = train_image.reshape(batch_size,np.prod(image_size))
px_te = test_image.reshape(batch_size,np.prod(image_size))
clf = svm.SVC()
clf.fit(px_tr, train_labels.ravel())  

px_y_hat = clf.predict(px_te)
px_acc = metrics.accuracy_score(test_labels.ravel(),px_y_hat)
px_y_score = clf.fit(px_tr, train_labels.ravel()).decision_function(px_te)
px_acc = svm_trainer(px_tr,px_te,train_labels.ravel(),test_labels.ravel(),
	num_estimates,num_training,num_testing)


######
#Plot PR curves

#Attention
precision, recall, _ = metrics.precision_recall_curve(test_labels.ravel(),
    y_score.ravel())
average_precision = metrics.average_precision_score(test_labels.ravel(), y_score.ravel())

#HOG
hog_precision, hog_recall, _ = metrics.precision_recall_curve(test_labels.ravel(),
    hog_y_score.ravel())
hog_average_precision = metrics.average_precision_score(test_labels.ravel(), hog_y_score.ravel())

#Pixel data
px_precision, px_recall, _ = metrics.precision_recall_curve(test_labels.ravel(),
    px_y_score.ravel())
px_average_precision = metrics.average_precision_score(test_labels.ravel(), px_y_score.ravel())

plt.clf()
plt.plot(recall, precision, label='Attention PR',color='r',linewidth=2)
plt.plot(hog_precision, hog_recall, label='HOG PR',color='g',linewidth=2)
plt.plot(px_precision, px_recall, label='Pixel PR',color='b',linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: Attention AUC={0:0.2f}, HOG AUC={0:0.2f}, Pixel AUC={0:0.2f}'.format(average_precision,hog_average_precision,px_average_precision))
plt.legend(loc="lower left")
plt.show()




# Plot DR data
from sklearn.manifold import TSNE, MDS

x_recons, h_enc, c_enc, z, kl, i_dec, h_dec, c_dec, center_y, center_x, delta = do_sample(test_image.reshape(batch_size,np.prod(image_size)))

in_data = np.sum(h_dec,axis=0).reshape(140,256)
in_labels = test_labels.ravel()


X_embedded = TSNE(n_components=2, init='pca', method = 'exact', verbose=0).fit_transform(in_data)
colors = plt.cm.Set1(in_labels/10.)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
        c=in_labels, marker=".", color=colors)
plt.show()




model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
att_coors = model.fit_transform(tr_data) 

clf = MDS(n_components=2, max_iter=100000)
att_coors = clf.fit_transform(me)


plot_embedding(att_coors,in_labels)
plt.show()

plt.scatter(att_coors[:,0],att_coors[:,1],c=test_labels.ravel())



#pca = decomposition.PCA()
#pipe = Pipeline(steps=[('pca', pca), ('svm', clf)])
#n_components = [20, 40, 64,100,200]
#Cs = np.logspace(-4, 4, 3)
#estimator = GridSearchCV(pipe,
#                         dict(pca__n_components=n_components,
#                              svm__C=Cs))
#clf = estimator.fit(tr_hf, train_labels.ravel())
from scipy.io import savemat
out_data = {}
out_data['attention_tr'] = tr_data
out_data['attention_te'] = te_data
out_data['hog_tr'] = tr_hf
out_data['hog_te'] = te_hf
out_data['px_tr'] = px_tr
out_data['px_te'] = px_te
out_data['tr_label'] = train_labels.ravel()
out_data['te_label'] = test_labels.ravel()
savemat('xy_data',out_data)




l = draw.writer.z_trafo.apply(h_enc)

wcenter_y  = l[:,0]
wcenter_x  = l[:,1]
log_delta = l[:,2]
wdelta = T.exp(log_delta)

# normalize coordinates
wcenter_x = (wcenter_x+1.)/2. * image_size[0]
wcenter_y = (wcenter_y+1.)/2. * image_size[1]
wdelta = (max(image_size[0], image_size[0])-1) / (r_size-1) * wdelta

#c_update, wcenter_y, wcenter_x, wdelta = draw.writer.zoomer.nn2att(l)
do_sample = theano.function(inputs=[images],outputs=[probs, wcenter_y, wcenter_x, wdelta],allow_input_downcast=True)



