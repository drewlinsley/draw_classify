#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import multiprocessing as mp
import theano
from fuel.datasets import IterableDataset, IndexableDataset
import commands
import re

def process(fname):
    image = imread(fname, as_grey=True)
    imagethr = np.where(image > np.mean(image),0.,1.0)
    return imagethr.ravel().astype(np.int64)

def assign_datastream(X,y):
	n_labels = np.unique(y).shape[0]
	y = np.eye(n_labels)[y]
	# Reassign dataset
	dataset = IndexableDataset({'features': X.astype(np.float64),'targets': y.astype(np.uint8)},sources=('features','targets'))
	return dataset

def import_sketch(data_dir):


	# make graphics inline
	#get_ipython().magic(u'matplotlib inline')

	find_string = u'find ' + data_dir + ' -name "*.jpg"'
	file_string = commands.getoutput(find_string)
	files = re.split('\n',file_string)

	#files = get_ipython().getoutput(u'find ' + data_dir + ' -name "*.jpg"')
	#len(files)

	#outpath = '/Users/drewlinsley/Documents/draw/draw/datasets'
	#datasource = 'sketch_uint8_shuffle'

	#plt.figure(figsize=(12,3))
	#image = imread(files[0], as_grey=True)
	#imagethr = np.where(image > np.mean(image),0.,1.0)

	#plt.subplot(1,3,1)
	#plt.imshow(imagethr, cmap=cm.gray);
	#imdilated = morphology.dilation(imagethr, np.ones((16,16)))
	#plt.subplot(1,3,2)
	#plt.imshow(imdilated, cmap=cm.gray);

	#im1 = resize(imdilated,[56,56])
	#plt.subplot(1,3,3)
	#plt.imshow(im1, cmap=cm.gray);
	#plt.show()


	NUM_PROCESSES = 8
	pool = mp.Pool(NUM_PROCESSES)
	results = pool.map(process, files, chunksize=100)
	pool.close()
	pool.join()


	y = np.array(map(lambda f: f.split('_')[-2], files))
	y = y.reshape(-1,1)
	y = y.astype(np.int64)
	#y.reshape(-1,1)


	X = np.array(results)
	N, image_size = X.shape
	D = int(np.sqrt(image_size))
	N, image_size, D


	num_els = y.shape[0]
	test_size = int(num_els * (.1/2)) #/2 because +/- types
	pos_test_id = np.asarray(range(0,test_size))
	neg_test_id = np.asarray(range(num_els - test_size,num_els))
	train_id = np.asarray(range(test_size, num_els - test_size))


	test_y = y[np.hstack((pos_test_id,neg_test_id))]
	test_X = X[np.hstack((pos_test_id,neg_test_id))]
	N_test = test_y.shape[0]
	np.sum(test_y)


	train_y = y[train_id]
	train_X = X[train_id]
	N_train = train_y.shape[0]
	np.sum(train_y)


	import random
	test_s = random.sample(xrange(test_y.shape[0]),test_y.shape[0])
	train_s = random.sample(xrange(train_y.shape[0]),train_y.shape[0])
	test_X=test_X[test_s]
	train_X=train_X[train_s]
	test_y=test_y[test_s]
	train_y=train_y[train_s]


	train_y.dtype

	return test_X, train_X, test_y, train_y

	#import fuel
	#datasource_dir = os.path.join(outpath, datasource)
	#get_ipython().system(u'mkdir -p {datasource_dir}')
	#datasource_fname = os.path.join(datasource_dir , datasource+'.hdf5')
	#datasource_fname


	# In[132]:

	#import h5py
	#fp = h5py.File(datasource_fname, mode='w')
	#image_features = fp.create_dataset('features', (N, image_size), dtype='uint8')


	# In[133]:

	# image_features[...] = np.vstack((train_X,test_X))


	# # In[134]:

	# targets = fp.create_dataset('targets', (N, 1), dtype='uint8')


	# # In[135]:

	# targets[...] = np.vstack((train_y,test_y)).reshape(-1,1)


	# # In[136]:

	# from fuel.datasets.hdf5 import H5PYDataset
	# split_dict = {
	#     'train': {'features': (0, N_train), 'targets': (0, N_train)},
	#     'test': {'features': (N_train, N), 'targets': (N_train, N)}
	# }
	# fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)


	# # In[137]:

	# fp.flush()
	# fp.close()


	# # In[138]:

	# get_ipython().system(u'ls -l {datasource_fname}')


	# # In[139]:

	# #!aws s3 cp {datasource_fname} s3://udidraw/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers


	# # #Look at training

	# # In[140]:

	# train_set = H5PYDataset(datasource_fname, which_sets=('train',))


	# # In[141]:

	# train_set.num_examples


	# # In[142]:

	# train_set.provides_sources


	# # In[143]:

	# handle = train_set.open()
	# data = train_set.get_data(handle, slice(0, 16))
	# data[0].shape,data[1].shape


	# # In[144]:

	# data[1]


	# # In[145]:

	# plt.figure(figsize=(12,12))
	# for i in range(16):
	#     plt.subplot(4,4,i+1)
	#     plt.imshow(data[0][i].reshape(D,D), cmap=cm.gray)
	#     plt.title(data[1][i][0]);


	# # In[146]:

	# train_set.close(handle)

