"""
Utilities for input parsing and configuration settings.
"""
from collections import namedtuple
from absl import flags

# Run script param help.
param_help = {
    'dataset_path': 'directory contains tfrecords',
    'mean_img': 'path to mean image file',
    'mode': 'train or eval',

    'log_step': 'log each log_step',
    'log_dir': 'log directory',

    'train_batch_size': 'batch size for training',
    'epoch_num': 'epoch number',
    'gpu_num': 'number of gpu device'
}


def arg_def(name, default_val):
  """
  For convinience when parsing param.
  :param name: param name.
  :param default_val: default value.
  :return: param name and its help description.
  """
  return name, default_val, param_help[name]


Param = namedtuple('ParamStruct', [
    'dataset_path',
    'mean_img',
    'mode',

    'log_step',
    'log_dir',

    'train_batch_size',
    'epoch_num',

    'class_num',
    'learning_rate',
    'weight_decay',
    'momentum',
    'input_size',
    'card',

    'is_train',
    'gpu_num'
])


def inputParam():
  """
  Run script param defination.
  :return:
  """
  flags.DEFINE_string(*arg_def('dataset_path', ''))
  flags.DEFINE_string(*arg_def('mean_img', ''))
  flags.DEFINE_string(*arg_def('mode', 'train'))

  flags.DEFINE_integer(*arg_def('log_step', 10))
  flags.DEFINE_string(*arg_def('log_dir', ''))

  flags.DEFINE_integer(*arg_def('train_batch_size', 32))
  flags.DEFINE_integer(*arg_def('epoch_num', 300))
  flags.DEFINE_integer(*arg_def('gpu_num', 1))

  return flags.FLAGS


def checkInputParam(FLAGS):
  if FLAGS.mode is 'train' and FLAGS.dataset_path is None:
    raise RuntimeError('You must specify --dataset_path for training.')

  if FLAGS.log_dir is None:
    raise RuntimeError('You must specify --log_dir for training.')

  if FLAGS.mean_img is None:
    raise RuntimeError('You must specify --mean_img for training.')

  return


def initParam(input_flag):
  """
  Complete config using input flags.
  :param input_flag: run script params.
  :return:
  """
  params = Param(
    dataset_path=input_flag.dataset_path,
    mean_img=input_flag.mean_img,
    mode=input_flag.mode,

    log_step=input_flag.log_step,
    log_dir=input_flag.log_dir,

    train_batch_size=input_flag.train_batch_size,
    epoch_num=input_flag.epoch_num,

    class_num=10,
    learning_rate=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    input_size=32,

    # Cardinality used in backbone ResNext
    card=10,

    is_train=(input_flag.mode == 'train'),
    gpu_num=input_flag.gpu_num
  )

  return params

