import tensorflow as tf
from utils.lossUtil import jaccardIndex, smoothL1
from utils import lossUtil
import numpy as np

def test_jaccardIndex(): # Test passed.
  gbbox = tf.constant([0.2, 0.3, 0.5, 0.5], dtype=tf.float32)
  anchors = np.array([[0.2, 0.3, 1, 1],
                      [0.2, 0.3, 0.5, 0.5]], dtype=np.float32)

  ji = jaccardIndex(gbbox, anchors)
  with tf.Session() as ss:
    ji_v = ss.run(ji)
    print(ji_v)

def test_smoothL1(): # Test passed.
  x = tf.constant([-1, 0.2, 0.5, 1, 2, 3], dtype=tf.float32)
  xs = smoothL1(x)
  with tf.Session() as ss:
    xsv = ss.run(xs)
    print(xsv)

def test_encode_bbox(): # Test passed.
  bbox = tf.constant([.2, .3, .5, .5], dtype=tf.float32)
  anchors = tf.constant([[.2, .3, .5, .5],
                         [.2, .2, .4, .3]])
  gb = lossUtil.encodeBBox(bbox, anchors)
  with tf.Session() as ss:
    out= ss.run(gb)
    print(out)

def test_decodeBBox(): # Test passed.
  # First encode bbox
  bbox = tf.constant([.2, .3, .5, .5], dtype=tf.float32)
  anchors = tf.constant([[.2, .3, .5, .5],
                         [.2, .2, .4, .3]])
  gb = lossUtil.encodeBBox(bbox, anchors)

  # Then decode bbox
  bbox_r = lossUtil.decodeBBox(gb, anchors)
  with tf.Session() as ss:
    gb_o, bbox_o = ss.run([gb, bbox_r])
    print(gb_o, "\n", bbox_o)

if __name__ == '__main__':
  # test_jaccardIndex()
  # test_smoothL1()
  # test_encode_bbox()
  test_decodeBBox()