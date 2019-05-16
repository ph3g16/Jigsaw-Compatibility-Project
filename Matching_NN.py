# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:30:52 2019

Neural network to evaluate image matching.

Structure:
    0) imports and global parameters
    1) data input
    2) pre-processing
    3) network structure
    4) define optimisation algorithm
    5) training
    6) validation
    7) output
    8) load/save trained model

@author: Peter
"""

""" 0) imports and global parameters """
import os
import numpy
import math
import bloscpack
from timeit import default_timer as timer # start = timer(), end = timer(), time_elapsed = end - start
from random import shuffle
import tensorflow as tf
import tensorflow.contrib.slim as slim

piece_height = 28
piece_width = 28
colour_channels = 3
data_length_of_example = piece_width*piece_height*colour_channels*2 + 1 # x2 since there are two pieces in each example pair, +1 for the label
binary_filepath = "Binaries/"

""" 1) data input """
# data comes in large chunks referred to here as sub_epochs
# if better mixing is required then use Image_Prep.prepare_data_epoch (or a variation of this) to resample data for every new epoch

def load_sub_epoch(file_name):
    stack = bloscpack.unpack_ndarray_from_file(binary_filepath + file_name)
    num_examples = int( len(stack) / (data_length_of_example) )
    sub_epoch = []
    for example in range(num_examples):
        sub_epoch.append(DataPoint(example, stack))
    return(sub_epoch)

class DataPoint:
    # class DataPoint associates an input with an output (e.g. associating a picture with a label)
    def __init__(self, example_number, arr):
        example_start = example_number * data_length_of_example
        self.label = arr[example_start]
        self.image = arr[example_start+1: example_start+data_length_of_example]

""" 2) pre-processing --- this happens within the scope of the learning algorithm and will be applied to both training and test data """
# ideally the random elements in this process should be fixed/skipped when using the algorithm in evaluation mode

# two ways of doing the pre-processing.... one is to do it when assembling the data ... other option is to apply as you go using tf.map_fn

# placeholder variables
data = tf.placeholder(tf.float32, [None, data_length_of_example-1]) # Number of examples, number of inputs per example
target = tf.placeholder(tf.int32, [None]) # Number of examples, number of outputs

reshape = tf.reshape(data, [-1, piece_height, piece_width*2, colour_channels])
 
# randomly change the orientation of the puzzle-pair
reshape = tf.map_fn(lambda x: tf.image.random_flip_up_down(tf.image.random_flip_left_right(x)), reshape)

# At this point we split the image into two streams.
# "distorted_image" is pre-processed to highlight structural features within the image
# "float_image" is just a zero to one floating point representation of the raw RGB pixel values
# distorted image sensitizes the network to colour gradiants whereas the float image sensitizes the network to actual colours
distorted_image = tf.map_fn(lambda x: tf.image.per_image_standardization(tf.image.random_contrast(tf.image.random_brightness(x, max_delta=63), lower=0.2, upper=1.8)), reshape) 
float_image = reshape / 256

""" 3) network structure """
# set variables and scopes for tensorflow graph

# slicey slicey
conv1 = float_image[:,:,piece_width-1:piece_width+3,:]
conv3 = distorted_image[:,:,piece_width-13:piece_width+15,:]
conv7 = distorted_image[:,:,piece_width-13:piece_width+15,:]

# network layers  
with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.005),
                      padding = "VALID"): # padding = VALID implies no padding
    # conv colour differences
    conv1 = tf.transpose(conv1, perm=[0, 1, 3, 2])
    conv1 = slim.conv2d(conv1, 64, [1, 1], stride=1, scope="conv_flat")
    conv1 = slim.fully_connected(conv1, 64, scope="conv_flat_1")
    # conv 3x3
    conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, scope="conv_3x3")
    conv3 = slim.conv2d(conv3, 64, [2, 2], stride=2, scope="conv_3x3_1")
    conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, scope="conv_3x3_2")
    # conv 7x7
    conv7 = slim.conv2d(conv7, 64, [7, 7], stride=1, scope="conv_7x7")
    
#    cross_conv = slim.conv2d(distorted_image, 64, [4, 4], scope="cross_conv_1")
#    cross_conv = slim.max_pool2d(cross_conv, [2, 2], stride=2, padding="VALID")
#    cross_conv = slim.conv2d(cross_conv, 64, [4, 4], scope="cross_conv_2")
    
conv1 = tf.layers.flatten(conv1)
conv3 = tf.layers.flatten(conv3)
conv7 = tf.layers.flatten(conv7)
#cross_conv = tf.layers.flatten(cross_conv)
#meld = tf.concat([conv1, cross_conv], axis=1)
meld = tf.concat([conv1, conv3], axis=1)
    
with slim.arg_scope([slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.005)):    
    dense = slim.fully_connected(conv3, 512, scope="fully_connected_1")
    dense = slim.fully_connected(dense, 128, scope="fully_connected_2")
    logits = slim.fully_connected(dense, 2, activation_fn=tf.nn.tanh, scope="logits")
    output = tf.nn.softmax(logits)[:,1] # this outputs the probability that a pair matches, use [:,0] to obtain the probability that a pair does not match

""" 4) define optimisation algorithm """
# backpropogation variables
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits, name="loss_function")

#cross_entropy = tf.reduce_sum(tf.abs(target - tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
# translate the network output into a 1 or 0 prediction, use this to produce an output or to perform validation
#pred_rounded = tf.round(prediction)

prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
mistakes =tf.not_equal(target, prediction)
sum_mistakes = tf.reduce_sum(tf.cast(mistakes, tf.int32))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
false_negatives = tf.reduce_sum(tf.minimum(target, tf.cast(mistakes, tf.int32))) # logical AND, triggers if it is both a mistake and the original label was 1=matching
true_positives = tf.reduce_sum(tf.minimum(target, prediction))
evaluation_summary = tf.stack([sum_mistakes, false_negatives, true_positives])

# initialize session
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

""" 5) training """
# run training
def train_network(num_epochs=5, identifier="Training_", batch_size=500):

    for epoch in range(num_epochs):
        start = timer()
        total_batches = 0
        print("Epoch ",str(epoch+1))
        for file in os.listdir(binary_filepath):
            if file.startswith(identifier):
                print("Sub-epoch " + file)
                train_set = load_sub_epoch(file)
                shuffle(train_set)
                no_of_batches = math.floor(len(train_set) / batch_size)
                ptr = 0
                for j in range(no_of_batches): # cycle through all batches within training data
                    batch = train_set[ptr:ptr+batch_size]
                    inp = [element.image for element in batch]
                    out = [element.label for element in batch]
                    ptr+=batch_size
                    # run an accuracy check every 20 batches. This yields an imperfect evaluation which is useful to gauge how well the network is performing when tweaking and testing.
                    if j % 20 == 19: print("Batch {} error: {:1.3f}".format(j+1, sess.run(error,{data: inp, target: out})))
                    sess.run(minimize,{data: inp, target: out}) # run backpropogation
                total_batches += no_of_batches
        time_to_complete = timer() - start
        print("Epoch completed in {:.1f} seconds. {} batches were completed at a cost of {:.3f} seconds per batch".format(time_to_complete, total_batches, time_to_complete/total_batches))

# this function is experimental and serves no significant purpose
# running the function prior to training seems to create conditions where ReLU neurons can be used as logits. Interesting!
def false_training(category=1, identifier="Training_", batch_size=500):   
    for file in os.listdir(binary_filepath):
        if file.startswith(identifier):
            train_set = load_sub_epoch(file)
            shuffle(train_set)
            batch = []
            a = 0
            b = 0
            while b < batch_size:
                subject = train_set[a]
                if subject.label == category:
                    batch.append(train_set[a])
                    b += 1
                a += 1
            inp = [element.image for element in batch]
            out = [element.label for element in batch]
            sess.run(minimize,{data: inp, target: out}) # run backpropogation
            return("false training completed for category {}".format(category))

""" 6) validation """
# evaluate network results
def evaluate_network(identifier="Evaluation_", batch_size=500):
    total_incorrect = 0
    total_examples = 0
    total_false_negatives = 0
    total_true_positives = 0
    for file in os.listdir(binary_filepath):
        if file.startswith(identifier):
            test_set = load_sub_epoch(file)
            len_test_set = len(test_set)
            total_examples += len_test_set
            no_of_batches = math.floor(len_test_set / batch_size)
            ptr = 0
            # in this case batches are used to avoid memory overrun, rather than to combine gradients
            for j in range(no_of_batches): # cycle through all batches within training data
                    batch = test_set[ptr:ptr+batch_size]
                    inp = [element.image for element in batch]
                    out = [element.label for element in batch]
                    ptr+=batch_size
                    results = sess.run(evaluation_summary,{data: inp, target: out})
                    total_incorrect += results[0]
                    total_false_negatives += results[1]
                    total_true_positives += results[2]
    accuracy = 100 * ((total_examples - total_incorrect)/total_examples)
    false_positives = total_incorrect - total_false_negatives
    precision = total_true_positives / (total_true_positives + false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)
    print("Accuracy: {:.1f}%, precision: {:.3f}, recall: {:.3f}".format(accuracy, precision, recall))
    return(total_examples, total_incorrect, total_false_negatives, total_true_positives)

""" 7) output """
# use the network to evaluate matchings on new data
def use_network(identifier, target_filepath, batch_size=500):
    results = []
    for file in os.listdir(target_filepath):
        if file.startswith(identifier):
            test_set = load_sub_epoch(file)
            len_test_set = len(test_set)
#            total_examples += len_test_set
            no_of_batches = math.floor(len_test_set / batch_size)
            ptr = 0
            # in this case batches are used to avoid memory overrun, rather than to combine gradients
            for j in range(no_of_batches): # cycle through all batches within training data
                    batch = test_set[ptr:ptr+batch_size]
                    inp = [element.image for element in batch]
                    out = [element.label for element in batch]
                    ptr+=batch_size
                    outcomes = sess.run(output,{data: inp, target: out})
                    results.append(outcomes)    
    return(results)

""" 8) load/save trained model """
def save_model(save_name="model", save_location="temp/"):
    
    tf.train.Saver().save(sess, save_location + save_name + ".ckpt")
    
def load_model(filename="model", filepath="temp/"):
    
    tf.train.Saver().restore(sess, filepath + filename + ".ckpt")
    
def revert_blank_state():
    
    # re-initialise all variables, reverting the network to its initial pre-trained state
    sess.run(init_op)
