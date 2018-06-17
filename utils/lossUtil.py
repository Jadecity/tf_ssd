import tensorflow as tf
from collections import namedtuple

def jaccardIndex(gbbox, bboxes):
  """
  Compute jaccard index between bbox1 and bbox2.
  :param gbbox: ground truth bouding box, [x, y, width, height]
  :param bboxes: bouding box 1, shape[width, height, anchor_num, 4], 4 is [x, y, width, height]
  :return: jaccard index, shape[width, height, anchor_num]
  """

  BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])
  gbbox = BBox(x=gbbox[0], y=gbbox[1], width=gbbox[2], height=gbbox[3])
  bboxes = BBox(x=bboxes[:, :, :, 0], y=bboxes[:, :, :, 1], width=bboxes[:, :, :, 2], height=bboxes[:, :, :, 3])

  xmin = tf.minimum(tf.subtract(gbbox.x, tf.divide(gbbox.width, 2.0)),
                    tf.subtract(bboxes.x, tf.divide(bboxes.width, 2.0)))
  xmax = tf.maximum(tf.add(gbbox.x, tf.divide(gbbox.width, 2.0)),
                    tf.add(bboxes.x, tf.divide(bboxes.width, 2.0)))
  ymin = tf.minimum(tf.subtract(gbbox.y, tf.divide(gbbox.height, 2.0)),
                    tf.subtract(bboxes.y, tf.divide(bboxes.height, 2.0)))
  ymax = tf.maximum(tf.add(gbbox.y, tf.divide(gbbox.height, 2.0)),
                    tf.add(bboxes.y, tf.divide(bboxes.height, 2.0)))

  width = tf.subtract(tf.subtract(xmax, xmin), tf.add(gbbox.width, bboxes.width))
  width = tf.minimum(tf.constant(0, dtype=tf.float32), width)
  height = tf.subtract(tf.subtract(ymax, ymin), tf.add(gbbox.height, bboxes.height))
  height = tf.minimum(tf.constant(0, dtype=tf.float32), height)

  intersect_area = tf.multiply(width, height)
  union_area = tf.subtract(tf.add(tf.multiply(gbbox.width, gbbox.height),
                                  tf.multiply(bboxes.width, bboxes.height)),
                           intersect_area)

  return tf.divide(intersect_area, union_area)

def smoothL1(x):
  """
  Compute l1 smooth for each element in tensor x.
  :param x: input tensor.
  :return: l1 smooth of x.
  """
  fx = tf.where(tf.less(tf.abs(x), 1.0),
                tf.multiply(tf.square(x), 0.5),
                tf.subtract(tf.abs(x), 0.5))
  return fx