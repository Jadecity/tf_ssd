import models.ssd_resnet_50 as ssd_resnet_50
import tensorflow as tf
import numpy as np

def test_multibox_predict(): # Test passed.
  input_layer = tf.zeros([1, 10, 10, 256], dtype=tf.float32)
  class_num = 50
  layer_name = 'resnet_v2_50/block3'
  weight_decay = 0.9

  cls, pos = ssd_resnet_50.multibox_predict(input_layer, class_num, layer_name, weight_decay)

  init = tf.global_variables_initializer()
  with tf.Session() as ss:
    ss.run(init)

    clsv, posv = ss.run([cls, pos])

    print(clsv.shape)
    print(posv.shape)

def test_predict(): # Test passed.
  image = tf.zeros([1, 300, 300, 3], dtype=tf.float32)
  class_num = 50
  weight_decay = 0.9

  ssd = ssd_resnet_50.init(class_num, weight_decay, False)
  logits, locations, end_feats = ssd(image)

  init = tf.global_variables_initializer()
  with tf.Session() as ss:
    ss.run(init)

    out = ss.run(locations)
    print(out.shape)

def test_posMask(): # Test passed.
  bbox = tf.constant([0.5, 0.5, 0.2, 0.2], dtype=tf.float32)
  anchors = np.array([[0.5, 0.5, 0.1, 0.1],[0.5, 0.5, 0.17, 0.17]], dtype=np.float32)
  mask = ssd_resnet_50.posMask(bbox, anchors)
  print(mask.get_shape())
  with tf.Session() as ss:
    mask = ss.run(mask)

    print(mask)

def test_classLoss(): # Test passed.
  logits = tf.constant([[0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0]], dtype=tf.float32)
  label = 3
  pos_mask = tf.constant([False, True], dtype=tf.bool)
  cl = ssd_resnet_50.classLoss(logits, label, pos_mask)

  with tf.Session() as ss:
    neg_loss = ss.run(cl)

    print( neg_loss)
    return

def test_locationLoss(): # Test passed.
  loc = tf.constant([[0.5, 0.5, 0.2, 0.2],
                     [0.5, 0.5, 0.1, 0.2]], dtype=tf.float32)
  gbbox = tf.constant([0.5, 0.5, 0.1, 0.2], dtype=tf.float32)
  pos_mask = tf.constant([True, False], dtype=tf.bool)
  anchor_num = [2, 1]
  loss = ssd_resnet_50.locationLoss(loc, gbbox, pos_mask)
  with tf.Session() as ss:
    loss = ss.run(loss)

    print(loss)
    return

def test_ssdLoss(): # Test passed.
  logits = tf.constant([[[0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0]]], dtype=tf.float32)
  loc = tf.constant([[[0.5, 0.5, 0.2, 0.1],
                     [0.5, 0.5, 0.1, 0.2]]], dtype=tf.float32)
  labels = {'bbox_num': tf.constant([1], dtype=tf.float32),
            'labels': tf.constant([[3]]),
            'bboxes': tf.constant([[[0.5, 0.5, 0.2, 0.2]]])}
  alpha = 1
  batch_size = 1

  loss = ssd_resnet_50.ssdLoss(logits, loc, labels, alpha, batch_size)


  with tf.Session() as ss:
    loss = ss.run([loss])

    print(loss)
    return



if __name__ == '__main__':
    # test_multibox_predict()
  # test_predict( )
  # test_posMask()
  # test_classLoss()
  # test_locationLoss()
  test_ssdLoss()