"""
Main script to train a ssd model.
"""

from datasets.PascalDataset import PascalDataset
import utils.confUtil as confUtil
import tensorflow as tf
import numpy as np
import models.ssd_resnet_50 as ssd_resnet_50
from tensorflow.contrib import slim

# Init global conf
flags = confUtil.inputParam()
confUtil.checkInputParam(flags)
gconf = confUtil.initParam(flags)


def input_fn():
  """
  Input function for estimator.
  :return:
  """
  features = {}
  labels = {}

  dt = PascalDataset(gconf.dataset_path, gconf.train_batch_size)
  image, size, bbox_num, label_ids, bboxes = dt.get_itr().get_next()

  features['image'] = image

  labels['size'] = size
  labels['bbox_num'] = bbox_num
  labels['labels'] = label_ids
  labels['bboxes'] = bboxes

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

  # Compute SSD loss and put it to global loss.
  ssd_resnet_50.ssdLoss(logits, locations, labels, params['alpha'])
  total_loss = tf.losses.get_total_loss()

  # Create train op
  optimazer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
  train_op = optimazer.minimize(total_loss, global_step=tf.train.get_or_create_global_step())

  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    pass # TODO

  if mode == tf.estimator.ModeKeys.PREDICT:
    prob_pred = tf.nn.softmax(logits, axis=4)
    predictions = {
      'prob': prob_pred,
      'location': locations
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions)




