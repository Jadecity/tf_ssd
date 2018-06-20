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

def main(_):

  if gconf.mode == 'train':
    # Set run config.
    run_conf = tf.estimator.RunConfig()
    run_conf.model_dir = gconf.model_dir
    run_conf.keep_checkpoint_max = 5
    run_conf.save_checkpoints_steps = 1000
    run_conf.save_summary_steps = 100

    # Create estimator.
    ssd_detector = tf.estimator.Estimator(model_fn=model_fn,
                                          params={
                                            'class_num': gconf.class_num,
                                            'weight_decay': gconf.weight_decay,
                                            'is_training': gconf.is_train,
                                            'alpha': gconf.alpha,
                                            'learning_rate': gconf.learning_rate
                                          },
                                          config=run_conf)

    # Train model.
    ssd_detector.train(input_fn=input_fn,
                       max_steps=gconf.epoch_num*gconf.per_epoch_num)
    return

  if gconf.mode == 'eval':
    # Set run config.
    run_conf = tf.estimator.RunConfig()
    run_conf.model_dir = gconf.model_dir

    # Create estimator.
    ssd_detector = tf.estimator.Estimator(model_fn=model_fn,
                                          params={
                                            'class_num': gconf.class_num
                                          },
                                          config=run_conf)

    # Evaluate model.
    eval_metrics = ssd_detector.evaluate(input_fn=input_fn)

    # Print evaluation metrics.
    pass

    return

  if gconf.mode == 'pred':
    # Set run config.
    run_conf = tf.estimator.RunConfig()
    run_conf.model_dir = gconf.model_dir

    # Create estimator.
    ssd_detector = tf.estimator.Estimator(model_fn=model_fn,
                                          params={
                                            'class_num': gconf.class_num
                                          },
                                          config=run_conf)

    # Use model to predict.
    predictions = ssd_detector.predict(input_fn=input_fn)

    # Visualize prediction result.
    pass

    return


if __name__ == '__main__':
  tf.app.run()





