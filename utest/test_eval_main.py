"""
Unit test for datasets.voc07.
"""
import sys
sys.path.insert(0, '/home/autel/PycharmProjects/tf_ssd/')
sys.path.insert(1, '/home/autel/libs/tensorflow-models/models/research/slim')

import pytest
import tensorflow as tf
import utils.confUtil as confUtil
from absl import flags, app
from datasets.PascalDataset import PascalDataset
# from train_ssd_main import model_fn
import models.ssd_resnet_50 as ssd_resnet_50
from object_detection.core.post_processing import multiclass_non_max_suppression
from utils import evalUtil
import datasets.pascalUtils as pascalUtils

# Init global conf
FLAGS = confUtil.inputParam()

# confUtil.checkInputParam(flags)

def input_fn():
  gconf = confUtil.initParam(FLAGS)
  """
  Input function for estimator.
  :return:
  """
  features = {}
  labels = {}

  dt = PascalDataset(gconf.dataset_path, gconf.train_batch_size)
  itr = dt.get_itr()
  image, size, bbox_num, label_ids, bboxes = itr.get_next()


  features['image'] = tf.cast(image, dtype=tf.float32)

  labels['size'] = size
  labels['bbox_num'] = bbox_num
  labels['labels'] = label_ids
  labels['bboxes'] = bboxes

  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, itr.initializer)

  return features, labels

def model_fn(features, labels, mode, params, config):
  """
  Model function for estimator
  :param features:
  :param labels:
  :param mode:
  :param params:
  :param config:
  :return:
  """
  image = features['image']

  # Init network.
  ssdnet = ssd_resnet_50.init(params['class_num'], params['weight_decay'], params['is_training'])

  # Compute output.
  logits, locations, endpoints = ssdnet(image)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute SSD loss and put it to global loss.
    ssd_resnet_50.ssdLoss(logits, locations, labels, params['alpha'])
    total_loss = tf.losses.get_total_loss()

    # Create train op
    optimazer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimazer.minimize(total_loss, global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    plogits = tf.unstack(logits, axis=0)
    probs = tf.nn.softmax(plogits, axis=1)
    pbboxes = tf.unstack(locations, axis=0)

    # Remove all background bboxes
    pbboxes, probs = evalUtil.rmBackgroundBox(pbboxes, probs)

    # Apply non maximum suppression.
    pbboxes_list = multiclass_non_max_suppression(pbboxes,
                                                  probs,
                                                  params['score_thresh'],
                                                  params['iou_thresh'],
                                                  params['max_size_per_class'])

    eval_metrics = {}
    eval_metrics.update(evalUtil.get_evaluate_ops(probs, pbboxes_list, labels, categories=labels['category']))
    return eval_metrics

  if mode == tf.estimator.ModeKeys.PREDICT:
    return logits, locations


def input_fn_main(_):
  feats, labels = input_fn()
  inits = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
  with tf.Session() as ss:
    ss.run(inits)
    for _ in range(1):
      fv, lv = ss.run([feats, labels])

def model_fn_forward_main(_):
  feats, labels = input_fn()

  inits = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
  prob, locations = model_fn(feats, labels, tf.estimator.ModeKeys.PREDICT, params={
    'class_num' : 50,
    'weight_decay':0.9,
    'is_training': False,
    'alpha': 1
  }, config=None)

  init = tf.global_variables_initializer()
  with tf.Session() as ss:
    ss.run(inits)
    ss.run(init)

    for _ in range(1):
      prob, locations = ss.run([prob, locations])

      print(prob.shape)



def test_input_fn(): # Test passed.
  sys.argv = ['test_train_main.py']
  app.run(input_fn_main)

  assert 1

# def test_model_fn_forward():
if __name__ == '__main__':
  sys.argv = ['test_train_eval.py']
  app.run(model_fn_forward_main)