import tensorflow as tf
from collections import namedtuple

def jaccardIndex(gbbox, anchors):
  """
  Compute jaccard index between bbox1 and bbox2.
  :param gbbox: ground truth bouding box, [x, y, width, height]
  :param anchors: bouding box 1, shape[anchor_num, 4], 4 is [x, y, width, height]
  :return: jaccard index, shape[anchor_num]
  """

  BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])
  gbbox = BBox(x=gbbox[0], y=gbbox[1], width=gbbox[2], height=gbbox[3])
  anchors = BBox(x=anchors[:, 0], y=anchors[:, 1], width=anchors[:, 2], height=anchors[:, 3])

  xmin = tf.minimum(tf.subtract(gbbox.x, tf.divide(gbbox.width, 2.0)),
                    tf.subtract(anchors.x, tf.divide(anchors.width, 2.0)))
  xmax = tf.maximum(tf.add(gbbox.x, tf.divide(gbbox.width, 2.0)),
                    tf.add(anchors.x, tf.divide(anchors.width, 2.0)))
  ymin = tf.minimum(tf.subtract(gbbox.y, tf.divide(gbbox.height, 2.0)),
                    tf.subtract(anchors.y, tf.divide(anchors.height, 2.0)))
  ymax = tf.maximum(tf.add(gbbox.y, tf.divide(gbbox.height, 2.0)),
                    tf.add(anchors.y, tf.divide(anchors.height, 2.0)))

  width = tf.subtract(tf.subtract(xmax, xmin), tf.add(gbbox.width, anchors.width))
  width = tf.minimum(tf.constant(0, dtype=tf.float32), width)
  height = tf.subtract(tf.subtract(ymax, ymin), tf.add(gbbox.height, anchors.height))
  height = tf.minimum(tf.constant(0, dtype=tf.float32), height)

  intersect_area = tf.multiply(width, height)
  union_area = tf.subtract(tf.add(tf.multiply(gbbox.width, gbbox.height),
                                  tf.multiply(anchors.width, anchors.height)),
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

def encodeBBox(bbox, anchors):
  """
  Encode ground truth bbox.
  :param bbox: shape[4]
  :param anchors: shape [anchor_num, 4]
  :return: encoded bbox, shape [anchor_num, 4]
  """
  anchor_num = anchors.get_shape()[0]
  shape = tf.stack([anchor_num, tf.constant(1, dtype=tf.int32)], axis=0)
  bbox = tf.expand_dims(bbox, axis=0)
  bbox = tf.tile(bbox, shape)

  bbox_center = tf.slice(bbox, [0, 0], [anchor_num, 2])
  bbox_size = tf.slice(bbox, [0, 2], [anchor_num, 2])
  anchor_center = tf.slice(anchors, [0,0], [anchor_num, 2])
  anchor_size = tf.slice(anchors, [0, 2], [anchor_num, 2])

  g_center = (bbox_center - anchor_center)/anchor_size
  g_size = tf.log(bbox_size/anchor_size)

  return tf.concat([g_center, g_size], axis=1)

def decodeBBox(bboxes, anchors):
  """
  Convert predicted bbox to original scale.
  :param bboxes: output of model, shape [anchor_num, 4]
  :param anchors: shape [anchor_num, 4]
  :return:
  """
  anchor_num = anchors.get_shape()[0]

  bbox_center = tf.slice(bboxes, [0, 0], [anchor_num, 2])
  bbox_size = tf.slice(bboxes, [0, 2], [anchor_num, 2])
  anchor_center = tf.slice(anchors, [0,0], [anchor_num, 2])
  anchor_size = tf.slice(anchors, [0, 2], [anchor_num, 2])

  pbbox_center = bbox_center * anchor_size + anchor_center
  pbbox_size = tf.exp(bbox_size) * anchor_size

  return tf.concat([pbbox_center, pbbox_size], axis=1)