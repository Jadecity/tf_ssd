import tensorflow as tf
from utils import lossUtil
from utils import anchorsUtil
import utils.evalUtil as evalUtil

def test_rmBackgroundBox():# Test passed.
  bboxes = tf.constant([[0.1, 0.2, 0.4, 0.4],
                        [0.1, 0.2, 0.4, 0.5],
                        [0.1, 0.2, 0.4, 0.6]], dtype=tf.float32)
  probs = tf.constant([[0.1, 0.1, 0.4, 0.4],
                        [0.5, 0.1, 0.1, 0.3],
                        [0.1, 0.2, 0.4, 0.6]], dtype=tf.float32)
  bboxes, probs = evalUtil.rmBackgroundBox(bboxes, probs)
  with tf.Session() as ss:
    bboxes, probs = ss.run([bboxes, probs])
    print(bboxes, probs)

def test_get_evaluate_ops():
  probs = tf.constant([[0.1, 0.1, 0.4, 0.4],
                       [0.5, 0.1, 0.1, 0.3],
                       [0.1, 0.2, 0.4, 0.3]], dtype=tf.float32)
  pbbox = tf.constant([[0.2, 0.2, 0.5, 0.5],
                       [0.1, 0.1, 0.2, 0.2],
                       [0.5, 0.5, 0.3, 0.3]], dtype=tf.float32)
  glabel = {}
  glabel['labels'] = tf.constant([[2, 3, 2]], dtype=tf.int32)
  glabel['bboxes'] = tf.constant([[[0.2, 0.2, 0.5, 0.5],
                                   [0.1, 0.1, 0.1, 0.1],
                                   [0.5, 0.5, 0.3, 0.3]]], dtype=tf.float32)
  cat = [{'id': 1, 'name': 'dog'},
         {'id': 2, 'name': 'cat'}]

  metric_ops = evalUtil.get_evaluate_ops(probs, pbbox, glabel, cat)
  val_ops = []
  for k in metric_ops.keys():
    _, v = metric_ops[k]
    val_ops.append(v)

  with tf.Session() as ss:
    vals = ss.run(val_ops)
    print(vals)

if __name__ == '__main__':
  # test_rmBackgroundBox()
  test_get_evaluate_ops()