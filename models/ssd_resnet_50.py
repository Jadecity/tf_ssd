"""
Define ssd net using resnet_50 as backbone network.
"""
import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v2
from utils import anchors
from utils import confUtil, lossUtil
import numpy as np


def multibox_predict(input_layer, class_num, layer_name, batch_size, weight_decay):
  """
  Compute predictions for each input layer.
  :param input_layer: Input feature layer
  :param class_num: number of output classes.
  :param anchor_num: number of anchors.
  :return: prediction p, and localizatoin l.
  """

  # Get anchors for each layer.
  layer_anchors = anchors.get_layer_anchors(layer_name)
  anchor_num = layer_anchors.shape()[2]
  input_shape = [layer_anchors.shape()[0], layer_anchors.shape()[1]]

  with tf.variable_scope('pred'):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):

      pred = slim.conv2d(input_layer, anchor_num * (class_num + 4), kernel_size=[3, 3], name='pred_conv1')

  # Reshape output tensor to extract each anchor prediction.
  pred_cls = tf.slice(pred.outputs, [0, 0, 0, 0, 0], [batch_size,
                                                            input_shape[0],
                                                            input_shape[1],
                                                            anchor_num,
                                                            class_num])

  pred_pos = tf.slice(pred.outputs, [0, 0, 0, 0, class_num], [batch_size,
                                                              input_shape[0],
                                                              input_shape[1],
                                                              anchor_num,
                                                              4])

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
                                                   use_batch_norm=True)):
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
          net = slim.layers.conv2d(net, 256, kernel_size=[1, 1], activation_fn=tf.nn.relu, name='conv3')
          net = slim.layers.conv2d(net, 512, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu, name='conv4')
        endpoints[block] = net

        block = 'block-2'
        with tf.variable_scope(block):  # 7x7x512 -> 4x4x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu, name='conv5')
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.relu, name='conv6')
        endpoints[block] = net

        block = 'block-3'
        with tf.variable_scope(block):  # 4x4x256 -> 2x2x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu, name='conv7')
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu, name='conv8')
        endpoints[block] = net

        block = 'block-4'
        with tf.variable_scope(block):  # 2x2x256 -> 1x1x256
          net = slim.layers.conv2d(net, 128, kernel_size=[1, 1], activation_fn=tf.nn.relu, name='conv9')
          net = slim.layers.conv2d(net, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu, padding='VALID', name='conv10')
        endpoints[block] = net

    """
    Add classifier conv layers to each added feature map(including the last layer of backbone network).
    Prediction and localisations layers.
    """
    logits = {}
    locations = {}
    for layer_name in endpoints.keys():
      logit, location = multibox_predict(endpoints[layer_name], class_num, layer_name, weight_decay)

      logits[layer_name] = logit
      locations[layer_name] = location

    return logits, locations, endpoints

  return predict


def posMask(bbox, anchors):
  """
  Create a boolean mask of shape [w, h, anchor_num]
  :param bboxes: ground truth bbox of shape[4]
  :param anchors: anchors of current layer , shape [w, h, anchor_num]
  :return: a boolean mask of shape [w, h, anchor_num]
  """

  # Compute jaccard overlap.
  overlap = lossUtil.jaccardIndex(bbox, tf.constant(anchors, dtype=tf.float32))

  # Get positive and negtive mask accoding to overlap.
  pos_mask = tf.greater(overlap, 0.5)

  return pos_mask


def classLoss(logits, label, pos_mask):
  """
  Classification loss.
  :param logits: predicted logits, shape [w, h, anchor_num*class_num]
  :param label: shape [bbox_num]
  :param pos_mask: shape [w, h, anchor_num]
  :return: loss
  """
  neg_mask = tf.logical_not(pos_mask)

  # Loss for each postition.
  conf_loss = tf.log(tf.nn.softmax(logits, axis=3))
  cat_idx = tf.where(tf.greater(label, 0))
  cat_idx = tf.cast(cat_idx, dtype=tf.int32)
  cat_idx = tf.concat([[0, 0, 0], cat_idx[0]], axis=0)
  conf_loss = tf.slice(conf_loss, cat_idx,
                       [conf_loss.get_shape()[0],
                        conf_loss.get_shape()[1], conf_loss.get_shape()[2], 1])
  conf_loss = tf.squeeze(conf_loss, [3])

  pos_loss = tf.boolean_mask(conf_loss, pos_mask)
  neg_loss = tf.boolean_mask(conf_loss, neg_mask)

  # Top-k negative loss.
  pos_num = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.int32))
  neg_num = tf.reduce_sum(tf.cast(neg_mask, dtype=tf.int32))
  neg_loss = tf.cond(tf.equal(tf.multiply(neg_num, pos_num), 0),
                     lambda: tf.constant(0, dtype=tf.float32),
                     lambda: tf.nn.top_k(neg_loss, tf.minimum(neg_num, tf.multiply(pos_num, 3)))[0])

  pos_loss = tf.reduce_sum(pos_loss)
  neg_loss = tf.reduce_sum(neg_loss)
  conf_loss = tf.negative(tf.add(pos_loss, neg_loss))

  return conf_loss


def locationLoss(location, gbbox, pos_mask, layer_shape):
  """
  Compute location loss.
  :param location: predicted location, shape [w, h, anchor_num*4]
  :param gbbox: ground truth bbox, shape [4]
  :param pos_mask: shape [w, h, anchor_num]
  :param layer_shape: a tensor, 1-d, contains [w, h, anchor_num] of current layer.
  :return: location loss
  """

  for _ in range(3):
    gbbox = tf.expand_dims(gbbox, 0)

  gbbox = tf.tile(gbbox, layer_shape)

  diff = tf.subtract(location, gbbox)
  loc_loss = lossUtil.smoothL1(diff)
  loc_loss = tf.boolean_mask(loc_loss, pos_mask)
  loc_loss = tf.reduce_sum(loc_loss)

  return loc_loss


def ssdLoss(logits, locations, labels, alpha, bathch_size):
  """
  Compute SSD loss.
  :param logits: a dict of raw prediction, key is layer name, each of shape [batch, w, h, anchor_number*class_num]
  :param locations: a dict of location prediction, key is layer name, each of shape [batch, w, h, anchor_number*4]
  :param labels: a dict,
            labels['bbox_num']: bbox number
            labels['labels']: shape [batch, bbox_num],
            labels['bboxes']: shape [batch, bbox_num, 4]
  :param alpha: weight between classification loss and location loss.
  :return:
  """

  for bi in range(bathch_size):
    label = labels['labels'][bi]
    bboxes = labels['bboxes'][bi]
    """For each layer, compute ssd loss"""
    for layer_name in confUtil.endLayers():
      anchors = anchors.get_layer_anchors(layer_name)
      layer_shape = tf.constant([anchors.shape()[0:3]])
      """For each ground truth box, compute loss"""
      for bbox in bboxes:
        pos_mask = posMask(bbox, anchors)
        pos_num = tf.reduce_sum(pos_mask)
        cls_loss = classLoss(logits[bi], label, pos_mask, layer_shape)
        loc_loss = locationLoss(locations[bi], bbox, pos_mask, layer_shape)

        total_loss = (cls_loss + alpha * loc_loss) / pos_num
        tf.losses.add_loss(total_loss)

  return