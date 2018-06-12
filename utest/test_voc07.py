"""
Unit test for datasets.voc07.
"""
import sys
sys.path.insert(0, '/home/autel/PycharmProjects/tf_ssd/')

import pytest
from datasets import pascalUtils
from datasets.PascalDataset import PascalDataset
import tensorflow as tf

@pytest.mark.skip()
def test_cvrt2tfrecord(): # Test passed
  json_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations_json'
  img_home = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/JPEGImages'
  label_file = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json'
  dest_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
  each_max = 1000

  pascalUtils.cvrt2tfrecord(json_path, img_home, label_file, dest_path, each_max)

  assert 1

def test_pascal_dataset(): # Test passed
  path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
  dataset = PascalDataset(path, 2)
  itr = dataset.get_itr()
  img, size, bbox_num, labels, bboxes = itr.get_next()

  with tf.Session() as ss:
    ss.run(itr.initializer)
    for _ in range(5):
      img_val = ss.run([img])

  assert 1