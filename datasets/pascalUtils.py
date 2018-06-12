"""
Utils to manipulate pascal dataset.
"""
import os
import os.path as path
import xml.etree.ElementTree as ET
import glob
import json
import math
import tensorflow as tf
import numpy as np
import scipy


def _int64List_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# _bytes is used for string/char values
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def anno2json(ann_path, dest_json_path, label_id_file):
  """
  Convert all pascal annotations in ann_path to standard json files in dest_json_path.
  And save label id to label name dictionary at the same time.
  :param ann_path: path to pascal annotations.
  :param dest_json_path: path to destination json files.
  :param label_id_file: full file name of label name to id file.
  :return: None.
  """
  class_list = {}

  # Check folder existance.
  if not path.isdir(ann_path):
    print('Path %s not exists!' % ann_path)

  if not path.isdir(dest_json_path):
    print('Path %s not exists!' % dest_json_path)

  # Convert each annotation file to json.
  for ann_full_name in glob.glob(path.join(ann_path, '*.xml')):
    ann_file = ann_full_name[ann_full_name.rfind('/') + 1:]
    dest_file = open(path.join(dest_json_path, ann_file.replace('xml', 'json')), mode='w')

    json_obj = {}
    tree = ET.parse(path.join(ann_path, ann_file))
    root = tree.getroot()
    json_obj['imgname'] = root.find('filename').text

    size = root.find('size')
    json_obj['imgsize'] = {'width': int(size.find('width').text),
                           'height': int(size.find('height').text),
                           'channel': int(size.find('depth').text)
                           }

    objects = root.findall('object')
    bboxes = []
    for obj in objects:
      bndbox = obj.find('bndbox')
      bboxes.append({
        'label': obj.find('name').text,
        'x1': int(math.floor(float(bndbox.find('xmin').text))),
        'y1': int(math.floor(float(bndbox.find('ymin').text))),
        'x2': int(math.floor(float(bndbox.find('xmax').text))),
        'y2': int(math.floor(float(bndbox.find('ymax').text)))
      })
      class_list[obj.find('name').text] = 0

    json_obj['objects'] = bboxes
    json.dump(json_obj, dest_file)
    dest_file.close()

  # Write class labels to disk, just for once.
  keys = class_list.keys()

  for i, key in enumerate(keys):
    class_list[key] = i + 1

  with open(label_id_file, 'w') as class_label:
    json.dump(class_list, class_label)
    class_label.close()

  return


def cvrt2tfrecord(json_path, img_home, label_file,
                  dest_path, each_max):
  """
  Read json files in src_path and corresponding image file,
  convert them to tfrecord format saved in dest_path.
  :param json_path: directory contains json files.
  :param img_home: home prefix to image name in json files.
  :param label_file: json file contains class and labels.
  :param dest_path: destination directory to put tfrecords.
  :param each_max: max number of tfrecord in each tfrecords file.
  :param preprocessor: will preprocess inputs.
  :return: none.
  """

  # Load label name to label id dict.
  with open(label_file, 'r') as label_file:
    label_dict = json.load(label_file)

  cnt = 1
  tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
  writer = tf.python_io.TFRecordWriter(tfrecord_filename)

  # Process each json file.
  for file in os.listdir(json_path):
    content = open(os.path.join(json_path, file)).read()
    ori_rcd = json.loads(content)

    img = scipy.misc.imread(os.path.join(img_home, ori_rcd['imgname']))
    img_size = np.array([ori_rcd['imgsize']['width'],
                ori_rcd['imgsize']['height'],
                ori_rcd['imgsize']['channel']])

    # Get labels and bboxes.
    labels = []
    bboxes = []
    for obj in ori_rcd['objects']:
      labels.append(label_dict[obj['label']])
      object = [obj['x1'], obj['y1'], obj['x2'], obj['y2']]
      bboxes.append(object)

    labels = np.array(labels)
    bboxes = np.array(bboxes)

    # Precess each bbox as a example
    feature = {
      'image_name': _bytes_feature(tf.compat.as_bytes(ori_rcd['imgname'])),
      'image': _bytes_feature(img.tobytes()),
      'size': _int64List_feature(img_size),
      'bbox_num': _int64_feature(len(labels)),
      'labels': _int64List_feature(labels),
      'bboxes': _int64List_feature(bboxes)
    }

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example.
    example_str = example.SerializeToString()
    writer.write(example_str)

    print('Convert : %d' % cnt)
    cnt = cnt + 1

    # Write out records each cnt_max items
    if 0 == cnt % each_max:
      writer.close()
      tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
      writer = tf.python_io.TFRecordWriter(tfrecord_filename)

  writer.close()
  return

def preprocess(img, size, bbox_num, labels, bboxes):
  """
  Transform batch data according to SSD requirement.
  :param img: shape [batch, N-D]
  :param size:shape [batch , 3]
  :param bbox_num: shape [batch, 1]
  :param labels: shape [batch, bbox_num, 1]
  :param bboxes: shape [batch, bbox_num, 4]
  :return:
  img: shape[batch, w, h, c]
  size: [batch, 3]
  bbox_num: [batch, 1]
  labels: [batch, bbox_num, 1]
  bboxes:[batch, bbox_num, 4]
  """
  return img, size, bbox_num, labels, bboxes