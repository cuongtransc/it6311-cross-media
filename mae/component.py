
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import const

import math
import tensorflow as tf



def encode(images,texts):

  #img_hidden
  with tf.name_scope('img_hidden'):
    weights = tf.Variable(
        tf.truncated_normal([const.INP_IMAGE_DIM, const.HIDDEN_IMG_DIM],
                            stddev=1.0 / math.sqrt(float(const.HIDDEN_IMG_DIM))),
        name='weights')

    biases = tf.Variable(tf.zeros([const.HIDDEN_IMG_DIM]),
                         name='biases')
    hidden_img = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Text_Hidden
  with tf.name_scope('txt_hidden'):


    weights = tf.Variable(
         tf.truncated_normal([const.INP_TXT_DIM, const.HIDDEN_TXT_DIM],
                             stddev=1.0 / math.sqrt(float(const.HIDDEN_TXT_DIM))),
         name='weights')
    biases = tf.Variable(tf.zeros([const.HIDDEN_TXT_DIM]),
                         name='biases')
    hidden_txt = tf.nn.relu(tf.matmul(texts, weights) + biases)
  # Linear
  with tf.name_scope('join_layer'):

    hidden_con = tf.concat([hidden_img,hidden_txt],axis=-1,name="hidden_con");
    weights = tf.Variable(
        tf.truncated_normal([const.HIDDEN_CON_DIM, const.SHARED_DIM],
                            stddev=1.0 / math.sqrt(float(const.SHARED_DIM))),
        name='weights')
    biases = tf.Variable(tf.zeros([const.SHARED_DIM]),
                         name='biases')
    representive = tf.matmul(hidden_con, weights) + biases
  return representive

def decode(representive):
    with tf.name_scope('img_join_d'):
        weights = tf.Variable(
            tf.truncated_normal([const.SHARED_DIM, const.HIDDEN_IMG_DIM],
                                stddev=1.0 / math.sqrt(float(const.HIDDEN_IMG_DIM))),
            name='weights')

        biases = tf.Variable(tf.zeros([const.HIDDEN_IMG_DIM]),
                             name='biases')
        hidden_img = tf.nn.relu(tf.matmul(representive, weights) + biases)


    with tf.name_scope("img_hidden_d"):
        weights = tf.Variable(
            tf.truncated_normal([const.HIDDEN_IMG_DIM, const.INP_IMAGE_DIM],
                                stddev=1.0/ math.sqrt(float(const.INP_IMAGE_DIM)),
                                name="weights")
        )
        biases = tf.Variable(tf.zeros([const.INP_IMAGE_DIM]), name="biases")

        img_decode = tf.nn.relu(tf.matmul(hidden_img,weights)+biases)

    with tf.name_scope("txt_join_d"):
        weights = tf.Variable(
            tf.truncated_normal([const.SHARED_DIM, const.HIDDEN_TXT_DIM],
                                stddev=1.0 / math.sqrt(float(const.HIDDEN_TXT_DIM))),
            name='weights')
        biases = tf.Variable(tf.zeros([const.HIDDEN_TXT_DIM]),
                             name='biases')
        hidden_txt = tf.nn.relu(tf.matmul(representive, weights) + biases)

    with tf.name_scope("txt_hidden_d"):
        weights = tf.Variable(
            tf.truncated_normal([const.HIDDEN_TXT_DIM, const.INP_TXT_DIM],
                                stddev=1.0 / math.sqrt(float(const.INP_TXT_DIM)),
                                name="weights")
        )
        biases = tf.Variable(tf.zeros([const.INP_TXT_DIM]), name="biases")

        txt_decode = tf.nn.relu(tf.matmul(hidden_txt, weights) + biases)

    return [img_decode,txt_decode]


def loss(decodes,true_inputs):
    #print (len(decodes),len(inputs))
    return tf.reduce_mean(tf.pow(tf.concat(decodes,axis=-1)-tf.concat(true_inputs,axis=-1),2))

def training(loss, learning_rate):

  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def compute_scores(sources,des):
    return tf.matmul(
        sources, des, transpose_b=True)


def eval(scores,matching_labels,true_labels):
    #print (scores.shape)
    #print (matching_labels.shape)
    #print (true_labels.shape)
    size = scores.shape[0]
    correct = 0
    for i in xrange(size):
        index = (-scores[i, :]).argsort()[0]
        if matching_labels[index] == true_labels[i]:
            correct += 1
    return correct
