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


  features['image'] = image

  labels['size'] = size
  labels['bbox_num'] = bbox_num
  labels['labels'] = label_ids
  labels['bboxes'] = bboxes

  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, itr.initializer)

  return features, labels

def main(_):
  feats, labels = input_fn()
  inits = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
  with tf.Session() as ss:
    ss.run(inits)
    for _ in range(1):
      fv, lv = ss.run([feats, labels])


def test_input_fn(): # Test passed.
  sys.argv = ['test_train_main.py']
  app.run(main)

  assert 1