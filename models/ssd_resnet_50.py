"""
Define ssd net using resnet_50 as backbone network.
"""
import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v2
from utils import anchorsUtil
from utils import confUtil, lossUtil
import numpy as np


def multibox_predict(input_layer, class_num, layer_name, weight_decay):
  """
  Compute predictions for each input layer.
  :param input_layer: Input feature layer
  :param class_num: number of output classes.
  :param anchor_num: number of anchors.
  :return: prediction p, and localizatoin l.
  """

  # Get anchors for each layer.
  layer_anchors = anchorsUtil.get_layer_anchors(layer_name)
  anchor_num = layer_anchors.shape[2]
  input_shape = [layer_anchors.shape[0], layer_anchors.shape[1]]

  with tf.variable_scope('pred/%s' % layer_name):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):

      pred = slim.conv2d(input_layer, anchor_num * (class_num + 4), kernel_size=[3, 3])

  # Reshape output tensor to extract each anchor prediction.
  pred = tf.reshape(pred, [-1, input_shape[0], input_shape[1], anchor_num, class_num + 4])
  pred_cls = tf.slice(pred, [0, 0, 0, 0, 0], [-1, input_shape[0], input_shape[1], anchor_num, class_num])
  pred_cls = tf.reshape(pred_cls, [-1, input_shape[0]*input_shape[1]*anchor_num, class_num])

  pred_pos = tf.slice(pred, [0, 0, 0, 0, class_num], [-1, input_shape[0], input_shape[1], anchor_num, 4])
  pred_pos = tf.reshape(pred_pos, [-1, input_shape[0] * input_shape[1] * anchor_num, 4])

  return pred_cls, pred_pos


def init(class_num, weight_decay, is_training):
  """
  Init and build net.
  :param class_num: number of class.
  :return: a callable object, e.g. network.
  """

  def predict(image):
    """
    Do predict using image.
    :param image: input image, shape [batch, w, h, c]
    :return: predicted raw logits.
    """

    # Load resnet50
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay,
                                                   use_batch_norm=False)):
      resnet, endpoints = resnet_v2.resnet_v2_50(inputs=image,
                                                 num_classes=class_num,
                                                 is_training=is_training)

    end_feats = {}
    net = endpoints['resnet_v2_50/block3']
    end_feats['resnet_v2_50/block3'] = net
    net = endpoints['resnet_v2_50/block4']
    end_feats['resnet_v2_50/block4'] = net

    # Create SSD conv layers.
    with tf.variable_scope('SSDNet'):
      with slim.arg_scope([slim.conv2d],
                          activation_fn=tf.nn.relu,
                          weights_regularizer=slim.l2_regularizer(weight_decay)):
        block = 'block-1'
        with tf.variable_scope(block):  # 14x14x512 -> 7x7x512
          net = slim.layers.conv2d(net, 256, kernel_size=[1, 1], activation_fn=tf.nn.relu)
          net = slim.layers.conv2d(net, 512, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        end_feats[block] = net

        block = 'block-2'
        with tf.variable_scope(block):  # 7x7x512 -> 4x4x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu)
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu)
        end_feats[block] = net

        block = 'block-3'
        with tf.variable_scope(block):  # 4x4x256 -> 2x2x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu)
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu)
        end_feats[block] = net

        block = 'block-4'
        with tf.variable_scope(block):  # 2x2x256 -> 1x1x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu)
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu, padding='VALID')
        end_feats[block] = net

    """
    Add classifier conv layers to each added feature map(including the last layer of backbone network).
    Prediction and localisations layers.
    """
    logits = []
    locations = []
    for layer_name in end_feats.keys():
      logit, location = multibox_predict(end_feats[layer_name], class_num, layer_name, weight_decay)

      logits.append(logit)
      locations.append(location)

    # Concat all feature layer outputs.
    logits = tf.concat(logits, axis=1)
    locations = tf.concat(locations, axis=1)

    return logits, locations, end_feats

  return predict


def posMask(bbox, anchors):
  """
  Create a boolean mask of shape [anchor_num].
  :param bboxes: ground truth bbox of shape[4]
  :param anchors: anchors of current layer , shape [anchor_num, 4]
  :return: a boolean mask of shape [anchor_num]
  """

  # Compute jaccard overlap.
  overlap = lossUtil.jaccardIndex(bbox, anchors)

  # Get positive and negtive mask accoding to overlap.
  pos_mask = tf.greater(overlap, 0.5)

  return pos_mask


def classLoss(logits, label, pos_mask):
  """
  Classification loss.
  :param logits: predicted logits, shape [anchor_num, class_num]
  :param label: scalar
  :param pos_mask: shape [anchor_num]
  :return: loss
  """
  neg_mask = tf.logical_not(pos_mask)

  # Loss for each postition.
  conf_loss = tf.log(tf.nn.softmax(logits, axis=1))


  pos_loss = tf.slice(conf_loss, [0, label], [-1, 1])
  pos_loss = tf.boolean_mask(pos_loss, pos_mask)
  pos_loss = tf.reduce_sum(pos_loss)

  neg_loss = tf.slice(conf_loss, [0, 0], [-1, 1])
  neg_loss = tf.boolean_mask(neg_loss, neg_mask)

  # Top-k negative loss.
  pos_num = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.int32))
  neg_num = tf.reduce_sum(tf.cast(neg_mask, dtype=tf.int32))
  neg_loss = tf.cond(tf.equal(tf.multiply(neg_num, pos_num), 0),
                     lambda: tf.constant(0, dtype=tf.float32),
                     lambda: tf.nn.top_k(neg_loss, tf.minimum(neg_num, tf.multiply(pos_num, 3)))[0])
  neg_loss = tf.reduce_sum(neg_loss)

  conf_loss = tf.negative(tf.add(pos_loss, neg_loss))

  return conf_loss


def locationLoss(location, gbbox, pos_mask):
  """
  Compute location loss.
  :param location: predicted location, shape [anchor_num, 4]
  :param gbbox: ground truth bbox, shape [anchor_num, 4]
  :param pos_mask: shape [anchor_num]
  :return: location loss
  """
  diff = tf.subtract(location, gbbox)
  loc_loss = lossUtil.smoothL1(diff)
  loc_loss = tf.boolean_mask(loc_loss, pos_mask)
  loc_loss = tf.reduce_sum(loc_loss)

  return loc_loss


def ssdLoss(logits, locations, labels, alpha, batch_size):
  """
  Compute SSD loss.
  :param logits: a tensor of raw prediction, shape [batch, total_anchor_num, class_num]
  :param locations: a tensor of location prediction, shape [batch, total_anchor_num, 4]
  :param labels: a dict,
            labels['bbox_num']: shape [batch, 1]
            labels['labels']: shape [batch, bbox_num],
            labels['bboxes']: shape [batch, bbox_num, 4]
  :param alpha: weight between classification loss and location loss.
  :return:
  """
  anchors = tf.constant(anchorsUtil.get_all_layer_anchors(), dtype=tf.float32)
  total_loss = tf.constant(0, dtype=tf.float32)

  for bi in range(batch_size):
    label = labels['labels'][bi]
    bboxes = labels['bboxes'][bi]

    """For each ground truth box, compute loss"""
    for bbox, cur_label in zip(tf.unstack(bboxes), tf.unstack(label)):
      # Transform bbox.
      g_bbox = lossUtil.encodeBBox(bbox, anchors)

      pos_mask = posMask(bbox, anchors)
      pos_num = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))


      cls_loss = classLoss(logits[bi], cur_label, pos_mask)
      loc_loss = locationLoss(locations[bi], g_bbox, pos_mask)

      total_loss += (cls_loss + alpha * loc_loss) / pos_num

  tf.losses.add_loss(total_loss)

  return total_loss