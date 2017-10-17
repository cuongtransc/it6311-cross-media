#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import component
import fakedata
import  loaddata
import const

import time
import os
import argparse
import sys
import numpy as np
FLAGS = None


def placeholder_inputs(batch_size):

  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                             const.INP_IMAGE_DIM))
  texts_placeholder = tf.placeholder(tf.float32,shape=(None,const.INP_TXT_DIM))

  true_images_placeholder = tf.placeholder(tf.float32,shape=(None,const.INP_IMAGE_DIM))
  true_texts_placeholder = tf.placeholder(tf.float32,shape=(None,const.INP_TXT_DIM))

  return images_placeholder,texts_placeholder,true_images_placeholder,true_texts_placeholder


def fill_train_feed_dict(fake_data, images_pl, texts_pl, true_images_pl, true_texts_pl):
    datas = fake_data.next_minibatch(FLAGS.batch_size)
    imgs,txts = datas[0]
    true_imgs,true_txts = datas[1]
    #print ("SHAPE: ",true_imgs.shape,true_txts.shape)
    feed_dict = {
        images_pl: imgs,
        texts_pl: txts,
        true_images_pl: true_imgs,
        true_texts_pl : true_txts

    }
    return feed_dict

def fill_test_feed_dict(imgs_test_data, txts_test_data, images_pl, texts_pl):
    test_feed_dict = {
        images_pl: imgs_test_data,
        texts_pl: txts_test_data
    }
    return test_feed_dict

def do_eval_img2txt(sess,encoder,images_data,text_representives,matching_labels,true_labels,imgs_pl,txts_pl):

    img_representives = do_get_img_representive(sess,encoder,images_data,imgs_pl,txts_pl)
    scores = component.compute_scores(img_representives,text_representives)
    scores = sess.run(scores)
    #scoring = sess.run(scores, feed_dict=test_feed_dict)
    correct = component.eval(scores,matching_labels,true_labels)
    print ("Acc Img to txt: %.4f"%(correct*1.0/images_data.shape[0]))


def do_eval_txt2img(sess,encoder,texts_data,image_representives,matching_labels,true_labels,imgs_pl,txts_pl):

    txt_representives = do_get_txt_representive(sess,encoder,texts_data,imgs_pl,txts_pl)

    scores = component.compute_scores(txt_representives,image_representives)
    scores = sess.run(scores)
    #scoring = sess.run(scores)
    correct = component.eval(scores,matching_labels,true_labels)
    print ("Acc Txt to img: %.4f"%(correct*1.0/texts_data.shape[0]))

def do_get_img_representive(sess,encoder,images_data,imgs_pl,txts_pl):
    with tf.Graph().as_default():

        zeros_text = np.zeros((images_data.shape[0], const.INP_TXT_DIM),dtype=float)
        #img_representives = component.encode(imgs_pl, txts_pl)
        test_feed_dict = fill_test_feed_dict(images_data, zeros_text, imgs_pl, txts_pl)

        return sess.run(encoder,feed_dict=test_feed_dict)

def do_get_txt_representive(sess,encoder,texts_data,imgs_pl,txts_pl):
    with tf.Graph().as_default():

        zeros_imgs = np.zeros((texts_data.shape[0], const.INP_IMAGE_DIM),dtype=float)
        #txt_representives = component.encode(imgs_pl, txts_pl)

        test_feed_dict = fill_test_feed_dict(zeros_imgs, texts_data, imgs_pl, txts_pl)
        return sess.run(encoder,feed_dict=test_feed_dict)



def run_training():
  with tf.Graph().as_default():
    #fake_data = fakedata.FakeData()
    fake_data = loaddata.TrueData(const.PATH_FEATURES,const.PATH_LABELS)

    images_placeholder, texts_placesholder , true_images_placeholder, true_texts_placesholder= placeholder_inputs(
        FLAGS.batch_size * 3)

    representive = component.encode(images_placeholder,
                        texts_placesholder)

    decode_values = component.decode(representive)

    loss_f = component.loss(decode_values,[true_images_placeholder,true_texts_placesholder])

    train_op = component.training(loss_f, FLAGS.learning_rate)

    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      feed_dict = fill_train_feed_dict(fake_data,
                                       images_placeholder,
                                       texts_placesholder,
                                       true_images_placeholder,
                                       true_texts_placesholder)

      _, loss_value = sess.run([train_op, loss_f],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        all_imgs_data,true_tag_imgs = fake_data.get_all_train_images()
        all_tags_data,true_tags_indices = fake_data.get_all_txt_tags()

        test_img_data,true_test_img_data = fake_data.get_test_images()

        test_tag_data,true_test_tag_indices = fake_data.get_test_txt_tags()





        img_representive = do_get_img_representive(sess,representive,all_imgs_data,images_placeholder,texts_placesholder)
        txt_representive = do_get_txt_representive(sess,representive,all_tags_data,images_placeholder,texts_placesholder)

        do_eval_img2txt(sess,representive,test_img_data,txt_representive,true_tags_indices,true_test_img_data,images_placeholder,texts_placesholder)
        do_eval_txt2img(sess,representive,test_tag_data,img_representive,true_tag_imgs,true_test_tag_indices,images_placeholder,texts_placesholder)



      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        print ("Saving...")
        saver.save(sess, checkpoint_file, global_step=step)

def main(_):
    run_training()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=10000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--save_dir',
      type=str,
      default="save",
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  print (FLAGS.batch_size, FLAGS.learning_rate)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
