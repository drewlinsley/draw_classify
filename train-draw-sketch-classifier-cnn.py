#!/usr/bin/env python

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
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity, Softmax, Rectifier, Logistic
from blocks.bricks.bn import (BatchNormalization, BatchNormalizedMLP,
                              SpatialBatchNormalization)
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate, SquaredError
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
from blocks.bricks.conv import (MaxPooling, Convolutional, ConvolutionalSequence, Flattener, ConvolutionalTranspose)

import draw.datasets as datasets
from draw.draw_cnn import *
from draw.samplecheckpoint import SampleCheckpoint
from draw.partsonlycheckpoint import PartsOnlyCheckpoint

from draw.datasets import package_sketch_images
sys.setrecursionlimit(100000)

#----------------------------------------------------------------------------
name = 'new_class_test'
epochs = 50
batch_size = 200
learning_rate = 3e-4
attention = '16,16'
n_iter = 8
enc_dim = 1024
dec_dim = 1024
z_dim = 100
oldmodel = None
#dataset = 'sketch'
dataset = 'sketch_uint8_shuffle'
data_dir = '/Users/drewlinsley/Documents/ubuntu_shared/res_results_problem_4'
#data_dir = '/home/ubuntu/res_results_problem_4'

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)

# train_ind = np.arange(data_train.num_examples)
# test_ind = np.arange(data_test.num_examples)
# rng = np.random.RandomState(seed=1)
# rng.shuffle(train_ind)
# rng.shuffle(test_ind)

# train_stream  = Flatten(DataStream.default_stream(
#     data_train,  iteration_scheme=ShuffledScheme(train_ind, batch_size)))
# test_stream  = Flatten(DataStream.default_stream(
#     data_test,  iteration_scheme=ShuffledScheme(test_ind, batch_size)))

#Get shuffled data
test_X, train_X, test_y, train_y = package_sketch_images.import_sketch(data_dir)
data_test = package_sketch_images.assign_datastream(test_X,test_y)
data_train = package_sketch_images.assign_datastream(train_X,train_y)
image_size = (int(np.sqrt(test_X.shape[1])),int(np.sqrt(test_X.shape[1])))
channels = 1
target_categories = np.unique(train_y).shape[0]

train_ind = np.arange(data_train.num_examples)
test_ind = np.arange(data_test.num_examples)
rng = np.random.RandomState(seed=1)
rng.shuffle(train_ind)
rng.shuffle(test_ind)

#####
#Comparisons to humans:
#Is there a neural signature for changes in read delta parameter (glimpse size)?
#Do machines/humans make similar mistakes?
#Learning time::: compare this somehow...
#####

#Convert datasets into fuel
#valid_stream = Flatten(DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
test_stream  = Flatten(DataStream.default_stream(data_test,  iteration_scheme=ShuffledScheme(test_ind, batch_size)))
train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=ShuffledScheme(train_ind, batch_size)))

if name is None:
    name = dataset

img_height, img_width = image_size
x_dim = channels * img_height * img_width

rnninits = {
    #'weights_init': Orthogonal(),
    'weights_init': IsotropicGaussian(0.01),
    'biases_init': Constant(0.),
}
inits = {
    #'weights_init': Orthogonal(),
    'weights_init': IsotropicGaussian(0.01),
    'biases_init': Constant(0.),
}

# Configure attention mechanism
if attention != "":
    read_N, write_N = attention.split(',')

    read_N = int(read_N)
    write_N = int(write_N)
    read_dim = 2 * channels * read_N ** 2

    reader = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                             channels=channels, width=img_width, height=img_height,
                             N=read_N, **inits)
    writer = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                             channels=channels, width=img_width, height=img_height,
                             N=write_N, **inits)
    attention_tag = "r%d-w%d" % (read_N, write_N)
else:
    read_dim = 2*x_dim

    reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

    attention_tag = "full"

#----------------------------------------------------------------------

if name is None:
    name = dataset

# Learning rate
def lr_tag(value):
    """ Convert a float into a short tag-usable string representation. E.g.:
        0.1   -> 11
        0.01  -> 12
        0.001 -> 13
        0.005 -> 53
    """
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, -exp)

lr_str = lr_tag(learning_rate)

subdir = name + "-cnn-" + time.strftime("%Y%m%d-%H%M%S");
longname = "%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % (dataset, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)
pickle_file = subdir + "/" + longname + ".pkl"

print("\nRunning experiment %s" % longname)
print("               dataset: %s" % dataset)
print("          subdirectory: %s" % subdir)
print("         learning rate: %g" % learning_rate)
print("             attention: %s" % attention)
print("          n_iterations: %d" % n_iter)
print("     encoder dimension: %d" % enc_dim)
print("           z dimension: %d" % z_dim)
print("     decoder dimension: %d" % dec_dim)
print("            batch size: %d" % batch_size)
print("                epochs: %d" % epochs)
print()

#----------------------------------------------------------------------

#encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
#encoder_mlp = MLP([Identity()], [(260), 4*enc_dim], name="MLP_enc", **inits)
#classifier_mlp = MLP([Rectifier(), Softmax()], [dec_dim, z_dim, 1], name="classifier", **inits) 

#q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

#draw = DrawClassifierModel(
#            n_iter, 
#            reader=reader,
#            encoder_mlp=encoder_mlp,
#            encoder_rnn=encoder_rnn,
#            sampler = q_sampler,
#            classifier=classifier_mlp)
#draw.initialize()


#///// 1. Convolution
act = Rectifier()
pool_layer = MaxPooling(
            pooling_size=(2, 2),
            step=(2,2),
            padding=(0,0))

encoder_cnn = ConvolutionalSequence(
    [
        Convolutional(
                    filter_size=(3, 3),
                    num_filters=64,
                    border_mode='half',
                    step=(1,1),
                    name='C1'),
        SpatialBatchNormalization(name='batch_norm1'),
        act,
        pool_layer,
        Convolutional(
                    filter_size=(3, 3),
                    num_filters=128,
                    border_mode='half',
                    step=(1,1),
                    name='C2'),
        SpatialBatchNormalization(name='batch_norm2'),
        act,
        pool_layer,
        Convolutional(
                    filter_size=(3, 3),
                    num_filters=256,
                    border_mode='half',
                    step=(1,1),
                    name='C3'),
        act,
        SpatialBatchNormalization(name='batch_norm3'),
        pool_layer
    ],
    num_channels=1,
    image_size=(read_N, read_N),
    name='encoder_cnn',
    **inits)

dummy_cnn = encoder_cnn
dummy_cnn.initialize()
cnn_output_dim = np.prod(dummy_cnn.get_dim('output')) #Take product now so that you can flatten later
cnn_mlp = MLP([Identity()], [cnn_output_dim, read_N ** 2],name="CNN_encoder", **inits) #convert CNN feature maps to encoder_mlp dimensions

#///// 2. Deconvolution
act = Rectifier()
pool_layer = MaxPooling(
            pooling_size=(2, 2),
            step=(2,2),
            padding=(0,0))

decoder_cnn = ConvolutionalSequence(
    [
        ConvolutionalTranspose(
                    filter_size=(3, 3),
                    num_filters=256,
                    border_mode='half',
                    step=(1,1),
                    name='C1'),
        SpatialBatchNormalization(name='batch_norm1'),
        act,
        #pool_layer,
        ConvolutionalTranspose(
                    filter_size=(3, 3),
                    num_filters=128,
                    border_mode='half',
                    step=(1,1),
                    name='C2'),
        SpatialBatchNormalization(name='batch_norm2'),
        act,
        #pool_layer,
        ConvolutionalTranspose(
                    filter_size=(3, 3),
                    num_filters=64,
                    border_mode='half',
                    step=(1,1),
                    name='C3'),
        SpatialBatchNormalization(name='batch_norm3'),
        Logistic(),
        #pool_layer
    ],
    num_channels=batch_size,
    image_size=4*enc_dim,
    name='decoder_cnn',
    **inits)

#/////
flattener = Flattener()
#///// LSTM  and classifier stuff
encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
encoder_mlp = MLP([Identity()], [(read_dim+enc_dim), 4*enc_dim], name="LSTM_encoder", **inits) #260 read_dim+dec_dim
#decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
#decoder_mlp = MLP([Identity()], [             dec_dim, 4*dec_dim], name="MLP_dec", **inits)
classifier_mlp = MLP([Identity(),Softmax()], [4*dec_dim, z_dim, target_categories], name="classifier", **inits) 

q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

draw = DrawClassifierModel(
            n_iter, 
            reader=reader,
            writer=writer,
            encoder_cnn=encoder_cnn,
            cnn_mlp=cnn_mlp,
            encoder_mlp=encoder_mlp,
            encoder_rnn=encoder_rnn,
            sampler = q_sampler,
            classifier=classifier_mlp,
            flattener=flattener)
draw.initialize()


#------------------------------------------------------------------------
x = tensor.matrix('features')
#y = tensor.ivector('targets')
y = tensor.imatrix('targets')

#y = theano.tensor.extra_ops.to_one_hot(tensor.lmatrix('targets'),2)
probs, h_enc, c_enc, center_y, center_x, delta = draw.reconstruct(x)
#probs, h_enc, c_enc, center_y, center_x, delta = draw.reconstruct(x)
#trim_probs = probs[-1,:] #Only take information from the last iteration
trim_probs = probs #Only take information from the last iteration
labels = y #tensor.lt(y, .5)
#trim_probs = probs.argmax(axis=1) #Only take information from the last iteration

#Apply a max to probs (get position of max index)
#Do the same for labels/dont use one hot

#cost = BinaryCrossEntropy().apply(labels, trim_probs)
#cost = SquaredError().apply(labels,trim_probs)
#cost = AbsoluteError().apply(tensor.concatenate([center_y, center_x, deltaY, deltaX]), tensor.concatenate([orig_y, orig_x, orig_dy, orig_dx]))
cost = (CategoricalCrossEntropy().apply(labels, trim_probs).copy(name='cost'))
#cost = tensor.nnet.categorical_crossentropy(trim_probs, labels)
#error_rate = tensor.neq(labels, trim_probs).mean(dtype=theano.config.floatX)
#error_rate = tensor.neq(labels.argmax(axis=1), trim_probs.argmax(axis=1)).mean(dtype=theano.config.floatX)


error_rate = tensor.neq(y.argmax(axis=1), trim_probs.argmax(axis=1)).mean(dtype=theano.config.floatX)
#error_rate = tensor.neq(y.argmax(axis=1), tensor.lt(trim_probs, .5).argmax(axis=1)).mean(dtype=theano.config.floatX)
#error_rate = (MisclassificationRate().apply(labels, trim_probs).copy(name='error_rate'))
cost.name = "BCE"
error_rate.name = "error_rate"



guesses = labels.argmax(axis=1) #tensor.lt(y, .5)#T.sum(y)#.argmax(axis=0)
ps = trim_probs
guesses.name = "guesses"
ps.name = "probs_shape"
#------------------------------------------------------------
cg = ComputationGraph([cost])
params = VariableFilter(roles=[PARAMETER])(cg.variables)


algorithm = GradientDescent(
    cost=cost, 
    parameters=params,
    step_rule=CompositeRule([
        StepClipping(10.), 
        Adam(learning_rate),
    ])
    #step_rule=RMSProp(learning_rate),
    #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
)


#------------------------------------------------------------------------
# Setup monitors
#monitors = [cost,error_rate,guesses,ps]
monitors = [cost,error_rate]
#monitors = [cost]
train_monitors = monitors[:]
train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
train_monitors += [aggregation.mean(algorithm.total_step_norm)]
# Live plotting...


#------------------------------------------------------------

if not os.path.exists(subdir):
    os.makedirs(subdir)


main_loop = MainLoop(
    model=Model(cost),
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=[
        Timing(),
        FinishAfter(after_n_epochs=epochs),
        TrainingDataMonitoring(
            train_monitors, 
            prefix="train",
            after_epoch=True),
#            DataStreamMonitoring(
#                monitors,
#                valid_stream,
##                updates=scan_updates,
#                prefix="valid"),
        DataStreamMonitoring(
            monitors,
            test_stream,
#                updates=scan_updates, 
            prefix="test"),
        #Checkpoint(name, before_training=True, after_epoch=True, save_separately=['log', 'model']),
        PartsOnlyCheckpoint("{}/{}".format(subdir,name), before_training=True, after_epoch=True, save_separately=['log', 'model']),
        ProgressBar(),
        Printing()])

if oldmodel is not None:
    print("Initializing parameters with old model %s"%oldmodel)
    with open(oldmodel, "rb") as f:
        oldmodel = pickle.load(f)
        main_loop.model.set_parameter_values(oldmodel.get_parameter_values())
    del oldmodel

main_loop.run()

#-----------------------------------------------------------------------------
