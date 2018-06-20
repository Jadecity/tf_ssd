"""
Script for reading voc 2007 dataset.
"""
import os.path
import tensorflow as tf
import glob
import datasets.pascalUtils as utils

class PascalDataset:
  def __init__(self, path, batchsize=1):
    """
    Init dataset object.
    :param path: path to tfrecords directory.
    :param batchsize: minibatch size.
    """

    self._dataset = self._createDataSet(path, batchsize)

  def _parse_func(self, example):
    """
    Parse tfrecord example.
    :param example: one example instance.
    :return: image, size, bbox_num, labels, bboxes
    """

    feature = {
      'image_name': tf.FixedLenFeature([], dtype=tf.string),
      'image': tf.FixedLenFeature([], dtype=tf.string),
      'size': tf.FixedLenFeature([3], tf.int64),
      'bbox_num': tf.FixedLenFeature([], tf.int64),
      'labels': tf.VarLenFeature(tf.int64),
      'bboxes': tf.VarLenFeature(tf.int64)
    }

    # Parse tfrecord example.
    example = tf.parse_single_example(serialized=example, features=feature)

    # Get attributes.
    image = tf.decode_raw(example['image'], tf.uint8)
    size = example['size']
    image = tf.reshape(image, size)
    labels = tf.sparse_tensor_to_dense(example['labels'])

    bbox_num = example['bbox_num']
    boxes_shape = tf.stack([bbox_num, 4])
    bboxes = tf.sparse_tensor_to_dense(example['bboxes'])
    bboxes = tf.reshape(bboxes, shape=boxes_shape)

    # TODO Preprocess batch data.
    # image, size, bbox_num, labels, bboxes = utils.preprocess(image, size,
    #                                                          bbox_num, labels,
    #                                                          bboxes)

    return image, size, bbox_num, labels, bboxes

  def _createDataSet(self, path, batchsize=1):
      """
      Create a tfrecrod dataset, all tfrecords should be in path directory.
      :param path: directory where tfrecords files live.
      :return: TFRecordDataset object.
      """

      if not os.path.exists(path):
          raise FileNotFoundError('Path %s not exist!' % path)

      rcd_files = glob.glob(os.path.join(path, '*.tfrecords'))
      if len(rcd_files) == 0:
          raise FileNotFoundError('No TFRecords file found in %s!' % path)

      # Create dataset.
      dataset = tf.data.TFRecordDataset(rcd_files)
      dataset = dataset.map(map_func=self._parse_func)
      padding_shape = ([None, None, None], [None], [],
                       tf.TensorShape([None]), tf.TensorShape([None, 4]))
      dataset = dataset.padded_batch(batchsize, padded_shapes=padding_shape)
      dataset = dataset.shuffle(buffer_size=200)
      dataset.prefetch(buffer_size=1000)

      return dataset

  def get_itr(self):
    return self._dataset.make_initializable_iterator()