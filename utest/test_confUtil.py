"""
Unit test for utils.confUtil
"""
import sys
sys.path.insert(0, '/home/autel/PycharmProjects/tf_ssd/')

import pytest
from datasets import pascalUtils
from datasets.PascalDataset import PascalDataset
import tensorflow as tf
from utils import confUtil
import sys
from absl import flags, app


FLAGS = confUtil.inputParam()
def main(_):
  # sys.argv.append('--mode=train')
  confUtil.checkInputParam(FLAGS)

  # Init global conf
  gconf = confUtil.initParam(FLAGS)

  print(gconf)

def test_inputParam():
# if __name__ == '__main__':

  # tf.app.run()
  # print(pytest.config.getoption("--cmdopt"))
  sys.argv = ['test_confUtil.py', '--gpu_num=4']
  app.run(main)

  assert 1
