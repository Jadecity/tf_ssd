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
from skimage.transform  import resize
import matplotlib.pyplot as plt


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
                  dest_path, each_max, dest_img_size):
  """
  Read json files in src_path and corresponding image file,
  convert them to tfrecord format saved in dest_path.
  :param json_path: directory contains json files.
  :param img_home: home prefix to image name in json files.
  :param label_file: json file contains class and labels.
  :param dest_path: destination directory to put tfrecords and image mean file.
  :param each_max: max number of tfrecord in each tfrecords file.
  :param dest_img_size: destination image size [width, height]
  :return: none.
  """

  # Load label name to label id dict.
  with open(label_file, 'r') as label_file:
    label_dict = json.load(label_file)

  cnt = 1
  tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
  writer = tf.python_io.TFRecordWriter(tfrecord_filename)

  dest_img_size = np.array(dest_img_size)
  dest_img_size = np.append(dest_img_size, 3)
  mean_img = np.zeros(dest_img_size, dtype=np.float32)

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

    # Preprocess raw data.
    img, img_size, bbox_num, labels, bboxes = preprocess(img, img_size, len(bboxes), labels, bboxes, dest_img_size)
    mean_img = np.add(mean_img, img)

    # from utils.visualize import visulizeBBox
    # visulizeBBox(img, bboxes)
    # plt.waitforbuttonpress()


    # Precess each bbox as a example
    feature = {
      'image_name': _bytes_feature(tf.compat.as_bytes(ori_rcd['imgname'])),
      'image': _bytes_feature(img.tobytes()),
      'size': _int64List_feature(img_size),
      'bbox_num': _int64_feature(bbox_num),
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

    # TODO
    break

  writer.close()

  # write mean image to file.
  mean_img /= cnt
  np.save(path.join(dest_path, 'mean_img.npy'), mean_img)
  return

class ResizePreprocessor:
  def __init__(self, dest_size):
    """
    This version, destination size should be square.
    :param conf:
    """

    self._dest_size = dest_size

  def __call__(self, img, size, bboxes):
    """
    Resize to dest size but keep all bounding boxes.
    Resize with ratio kept is first applied, then crop to dest size.
    Bbox positions are adjusted according to the new size.
    :param img: ndarray shape [w, h, c]
    :param size: ndarray, content [w, h, c]
    :param bboxes: ndarray, [Kx(x1, y1, x2, y2)]
    :return: new img, size, bboxes
    """
    w, h = size[0], size[1]
    wd, hd = self._dest_size[0], self._dest_size[1]
    ratio = np.float(w) / np.float(h)
    if ratio > 1:
      hm = hd
      wm = np.int(hm * ratio)
    else:
      wm = wd
      hm = np.int(wm / ratio)

    # Scale with ratio kept.
    # img_d = trans.resize(image=img, output_shape=([hm, wm]), preserve_range=True)
    img_d = resize(img, (hm, wm), preserve_range=True)

    # Scale bboxes
    w_s, h_s = wm / w, hm / h
    bboxes[:, 0] = (bboxes[:, 0] * w_s).astype(np.int)
    bboxes[:, 2] = (bboxes[:, 2] * w_s).astype(np.int)
    bboxes[:, 1] = (bboxes[:, 1] * h_s).astype(np.int)
    bboxes[:, 3] = (bboxes[:, 3] * h_s).astype(np.int)



    # Crop center area
    cx, cy = wm / 2, hm / 2
    x, y = np.int(cx - wd / 2), np.int(cy - hd / 2)
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 2])
    max_y = np.max(bboxes[:, 3])
    min_x, min_y = min(min_x, x), min(min_y, y)
    max_x = max(max_x, x + wd - 1, wm - 1)
    max_y = max(max_y, y + hd - 1, hm - 1)

    img_d = img_d[min_y:max_y + 1, min_x:max_x + 1, :]
    bboxes[:, 0] -= min_x
    bboxes[:, 2] -= min_x
    bboxes[:, 1] -= min_y
    bboxes[:, 3] -= min_y

    r_w, r_h = max_x - min_x + 1, max_y - min_y + 1

    if r_w != wd or r_h != hd:
      img_d, bboxes = self._rescale2dest(img_d, np.array([r_w, r_h]), bboxes)

    img_d = img_d.astype(np.uint8)
    #
    # from utils.visualize import visulizeBBox
    # visulizeBBox(img_d, bboxes)
    # plt.waitforbuttonpress()

    return img_d, self._dest_size, bboxes

  def _rescale2dest(self, img, size, bboxes):
    w, h = size[0], size[1]
    wd, hd = self._dest_size[0], self._dest_size[1]

    # Scale with ratio kept.
    # img_d = trans.resize(image=img, output_shape=([hd, wd]), preserve_range=True)
    img_d = resize(img, (hd, wd), preserve_range=True)
    # plt.imshow(img_d)
    # plt.draw()
    # plt.waitforbuttonpress()

    # Scale bboxes
    w_s, h_s = wd / w, hd / h
    bboxes[:, 0] = (bboxes[:, 0] * w_s).astype(np.int)
    bboxes[:, 2] = (bboxes[:, 2] * w_s).astype(np.int)
    bboxes[:, 1] = (bboxes[:, 1] * h_s).astype(np.int)
    bboxes[:, 3] = (bboxes[:, 3] * h_s).astype(np.int)

    return img_d, bboxes


def preprocess(img, size, bbox_num, labels, bboxes, dest_img_size):
  """
  Transform data according to SSD requirement.
  Crop and resize image, adjust bounding boxes location.
  :param img: shape [w, h, c]
  :param size:shape [3]
  :param bbox_num: integer
  :param labels: shape [bbox_num]
  :param bboxes: shape [bbox_num, 4]
  :param dest_img_size: desired image size, tuple or list (w, h)
  :return:
  img: shape[batch, w, h, c]
  size: [batch, 3]
  bbox_num: [batch, 1]
  labels: [batch, bbox_num, 1]
  bboxes:[batch, bbox_num, 4]
  """
  resizer = ResizePreprocessor(dest_img_size)
  img, size, bboxes = resizer(img, size, bboxes)

  return img, size, bbox_num, labels, bboxes

def getCategories(cat_file):
  """
  Load category dictionary from file.
  :return:
  """
  if not path.exists(cat_file):
    raise FileNotFoundError('%s not exists!' % cat_file)

  with open(cat_file) as cat:
    cat_lists = json.load(cat)

  return cat_lists['categories']