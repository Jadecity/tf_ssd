import models.ssd_resnet_50 as ssd_resnet_50
import tensorflow as tf

def test_multibox_predict(): # Test passed.
  input_layer = tf.zeros([1, 38, 38, 256], dtype=tf.float32)
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

    for k in locations.keys():
      out = ss.run(locations[k])
      print(k, out.shape)


if __name__ == '__main__':
    # test_multibox_predict()
  test_predict()
