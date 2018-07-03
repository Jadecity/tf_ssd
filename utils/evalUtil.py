import tensorflow as tf
from utils import lossUtil
from utils import anchorsUtil
from object_detection.metrics import coco_evaluation


def rmBackgroundBox(bboxes, probs):
  """
  Remove all background boxes.
  :param bboxes: tensor of shape [bbox_num, 4]
  :param probs: tensor of shape [bbox_num, class_num], class 0 is background class.
  :return: bboxes after background bboxes removed.
  """
  mask = tf.argmax(probs, axis=1)
  mask = tf.greater(mask, 0)
  bboxes = tf.boolean_mask(bboxes, mask)
  probs = tf.boolean_mask(probs, mask)

  return bboxes, probs


def get_evaluate_ops(probs, pbbox, glabel, categories):
  """
  :param probs: tensor of shape [1, anchor_num, class_num]
  :param pbbox: tensor of shape [1, anchor_num, 4]
  :param glabel: dict of tensor,
    'labels': shape [1, anchor_num]
    'bboxes': shape [1, anchor_num, 4]
  :param categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
  :return: a dict of tensor, representing metrics.
  """

  anchors = tf.constant(anchorsUtil.get_all_layer_anchors(), dtype=tf.float32)
  labels = tf.unstack(glabel['labels'], axis=0)
  bboxes = tf.unstack(glabel['bboxes'], axis=0)

  # Decode predicted bbox.
  pbboxes = lossUtil.decodeBBox(pbbox, anchors)
  pclasses = tf.argmax(probs, axis=1)

  # Compute mAP metrics.
  evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
  metric_ops = evaluator.get_estimator_eval_metric_ops(groundtruth_boxes=bboxes,
                                          groundtruth_classes=labels,
                                          detection_boxes=pbboxes,
                                          detection_scores=probs,
                                          detection_classes=pclasses)

  return metric_ops











