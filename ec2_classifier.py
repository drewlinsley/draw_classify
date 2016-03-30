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


def main(dataset,model_dir,model_file):

	#dataset = 'sketch_uint8_shuffle'
	#data_dir = '/Users/drewlinsley/Desktop/res_results_problem_4'
	data_dir = os.path.join('/Users/drewlinsley/Desktop/',dataset)
    #data_dir = '/home/ubuntu/svrt_data/'+dataset

	#Get shuffled data
	test_X, train_X, test_y, train_y = package_sketch_images.import_sketch(data_dir)
	image_size = (int(np.sqrt(test_X.shape[1])),int(np.sqrt(test_X.shape[1])))
	channels = 1
	target_categories = np.unique(train_y).shape[0]

	#Load old model
	#model_file = 'new_class_test-16_it/new_class_test_model'
	model_pointer = os.path.join(model_dir,model_file)
	with open(model_pointer,"rb") as f:
	    model = pickle.load(f)
	draw = model.get_top_bricks()[0]

	#Get attention model extractions
	images = T.matrix('images')

	probs, h_enc, c_enc, center_y, center_x, delta = draw.reconstruct(images)
	do_sample = theano.function(inputs=[images],outputs=[probs, h_enc, c_enc, center_y, center_x, delta],allow_input_downcast=True)
	tprobs, th_enc, tc_enc, tcenter_y, tcenter_x, tdelta = do_sample(test_X)
	correct_ims = tprobs.argmax(axis=1) == test_y.flatten()
	acc = np.sum(correct_ims) / test_X.shape[0]
	incorrect_ims = ~correct_ims

	#Save everything
	outfile = os.path.join(model_dir,'classification_data')
	np.savez(outfile,tprobs=tprobs, correct_ims=correct_ims, test_y=test_y, acc=acc)

	#Plot everything
	max_ims = 100
	from matplotlib.collections import LineCollection
	from SVRT_analysis_helper_functions import scale_range
	from SVRT_analysis_helper_functions import plot_ims
	# get the colors from the color map
	test_image_res = test_X#[0:batch_size,:].reshape(batch_size,image_size[0],image_size[1])
	#Make a folder
	subdir = os.path.join(model_dir,"classification_trace_errors")
	if not os.path.exists(subdir):
	    os.makedirs(subdir)
	plot_ims(test_image_res,tcenter_x,tcenter_y,tdelta,incorrect_ims,image_size,max_ims,subdir)
	#Make a folder
	subdir = os.path.join(model_dir,"classification_trace_correct")
	if not os.path.exists(subdir):
	    os.makedirs(subdir)
	plot_ims(test_image_res,tcenter_x,tcenter_y,tdelta,correct_ims,image_size,max_ims,subdir)
	#Which vars to save? tprobs, correct_ims, test_y, acc
	print("finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, dest="dataset",
                default=False, help="directory pointer to images")
    parser.add_argument("--model_dir", type=str, dest="model_dir",
                default=False, help="directory pointer to model")
    parser.add_argument("--model_file", type=str, dest="model_file",
                default=False, help="name of model file")
    args = parser.parse_args()
    main(**vars(args))