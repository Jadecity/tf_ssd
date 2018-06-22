"""
Unit test for datasets.voc07.
"""
import sys
sys.path.insert(0, '/home/autel/PycharmProjects/tf_ssd/')

import pytest
from datasets import pascalUtils
from datasets.PascalDataset import PascalDataset
import tensorflow as tf
from utils.visualize import visulizeBBox
import matplotlib.pyplot as plt
import numpy as np

# @pytest.mark.skip()
def test_cvrt2tfrecord(): # Test passed
  json_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations_json'
  img_home = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/JPEGImages'
  label_file = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json'
  dest_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
  each_max = 1000

  pascalUtils.cvrt2tfrecord(json_path, img_home, label_file, dest_path, each_max, [300, 300])

  assert 1

def test_pascal_dataset(): # Test passed
  path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
  dataset = PascalDataset(path, 1)
  itr = dataset.get_itr()
  img, size, bbox_num, labels, bboxes = itr.get_next()

  with tf.Session() as ss:
    ss.run(itr.initializer)
    for i in range(1):
      img_val, size_val, bbox_num_val, labels_val, bboxes_val = ss.run([img, size, bbox_num, labels, bboxes])
      print('%d-th image: '%i , img_val.shape)
      print('%d-th image shape: '%i, size_val)
      print('%d-th image bbox_num: '%i , bbox_num_val)
      print('%d-th image labels: '%i, labels_val)
      print('%d-th image bboxes: '%i, bboxes_val)

      # Visualize
      shape = [size_val[0, 1], size_val[0, 0], size_val[0, 2]]
      img_val.shape = shape
      visulizeBBox(img_val, bboxes_val[0])
      plt.waitforbuttonpress()

  assert 1

if __name__ == '__main__':
  test_cvrt2tfrecord()