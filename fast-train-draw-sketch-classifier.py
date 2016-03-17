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
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity, Softmax, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy, MisclassificationRate
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

sys.setrecursionlimit(100000)

#----------------------------------------------------------------------------
name = 'class_test'
epochs = 100
batch_size = 100
learning_rate = 3e-4
attention = '2,5'
n_iter = 16
enc_dim = 256
dec_dim = 256
z_dim = 100
oldmodel = None
dataset = 'sketch'

image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset)

train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size)))
valid_stream = Flatten(DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
test_stream  = Flatten(DataStream.default_stream(data_test,  iteration_scheme=SequentialScheme(data_test.num_examples, batch_size)))

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

subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S");
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

encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
encoder_mlp = MLP([Identity()], [(260), 4*enc_dim], name="MLP_enc", **inits)
decoder_mlp = MLP([Identity()], [             dec_dim, 4*dec_dim], name="MLP_dec", **inits)
classifier_mlp = MLP([Rectifier(), Softmax()], [dec_dim, z_dim, 1], name="classifier", **inits) 

q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

#draw = DrawClassifierModel(
#            n_iter, 
#            reader=reader,
#            encoder_mlp=encoder_mlp,
#            encoder_rnn=encoder_rnn,
#            sampler = q_sampler,
#            decoder_mlp=decoder_mlp,
#            decoder_rnn=decoder_rnn,
#            classifier=classifier_mlp)
#draw.initialize()
draw = LimDrawClassifierModel(
            n_iter, 
            reader=reader,
            encoder_mlp=encoder_mlp,
            encoder_rnn=encoder_rnn,
            sampler = q_sampler,
            classifier=classifier_mlp)
draw.initialize()

#------------------------------------------------------------------------
x = tensor.matrix(u'features')
y = tensor.lmatrix(u'targets')
#y = theano.tensor.extra_ops.to_one_hot(tensor.lmatrix(u'targets'),2)

#probs, h_enc, c_enc, i_dec, h_dec, c_dec, center_y, center_x, delta = draw.reconstruct(x)
probs, h_enc, c_enc, center_y, center_x, delta = draw.reconstruct(x)

trim_probs = probs[-1,:,:] #Only take information from the last iteration
labels = y.flatten()
#cost = BinaryCrossEntropy().apply(labels, trim_probs)
cost = CategoricalCrossEntropy().apply(y, trim_probs)
error_rate = MisclassificationRate().apply(labels, trim_probs)
cost.name = "CCE"

#------------------------------------------------------------
cg = ComputationGraph([cost])
params = VariableFilter(roles=[PARAMETER])(cg.variables)

algorithm = GradientDescent(
    cost=cost, 
    parameters=params,
    step_rule=CompositeRule([
        StepClipping(10.), 
        Adam(learning_rate),
    ]),
on_unused_sources='ignore',    
    #step_rule=RMSProp(learning_rate),
    #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
)

#------------------------------------------------------------------------
# Setup monitors
monitors = [cost,error_rate]
train_monitors = monitors[:]
#train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
#train_monitors += [aggregation.mean(algorithm.total_step_norm)]
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
        #Checkpoint(name, before_training=False, after_epoch=True, save_separately=['log', 'model']),
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
