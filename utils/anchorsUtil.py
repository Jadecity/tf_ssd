import numpy as np
import math

def get_scales(layer_num):
  """
  Compute default scales.
  :param layer_num:
  :return:
  """
  smin = 0.2
  smax = 0.9
  scales = [smin + (smax - smin)*(k - 1)/(layer_num - 1) for k in range(1, layer_num + 1)]

  return scales


def anchorBox(input_shape, layer_index, scales, ratios):
  sqrt_ratios = [math.sqrt(x) for x in ratios]
  boxes = []

  sk = scales[layer_index]
  for ratio in sqrt_ratios:
    width = sk * ratio
    height = sk / ratio
    boxes.append([width, height])

  # Add one more for ratio 1.
  if layer_index < len(scales) - 1:
    s_prime = math.sqrt(sk * scales[layer_index + 1])
  else:
    s_prime = math.sqrt(sk * 107.5)

  boxes.append([s_prime, s_prime])

  anchor_num = len(ratios) + 1
  anchors = np.zeros([input_shape[0] * input_shape[1] * anchor_num, 4])
  for r in range(input_shape[0]):
    cy = (r + 0.5) / float(input_shape[0])
    for c in range(input_shape[1]):
      cx = (c + 0.5) / float(input_shape[1])
      for k, box in enumerate(boxes):
        idx = r * input_shape[1] * anchor_num + c * anchor_num + k
        anchors[idx, :] = [cx, cy, box[0], box[1]]

  return anchors


scales = get_scales(6)
ratios = [1, 2, 3, 1/2, 1/3]
layers_shape = [[10, 10], [10, 10], [5, 5], [3, 3], [3, 3], [1, 1]]

anchors = {
  'resnet_v2_50/block3': anchorBox(layers_shape[0], 0, scales, ratios),
  'resnet_v2_50/block4': anchorBox(layers_shape[1], 1, scales, ratios),
  'block-1': anchorBox(layers_shape[2], 2, scales, ratios),
  'block-2': anchorBox(layers_shape[3], 3, scales, ratios),
  'block-3': anchorBox(layers_shape[4], 4, scales, ratios),
  'block-4': anchorBox(layers_shape[5], 5, scales, ratios),
}


def get_layer_anchors(layer_name):
  return anchors[layer_name]

def get_all_layer_anchors():
  # return np.stack([anchors['resnet_v2_50/block3'],
  #                  anchors['resnet_v2_50/block4'],
  #                  anchors['block-1'],
  #                  anchors['block-2'],
  #                  anchors['block-3'],
  #                  anchors['block-4']])
  return np.array([[0.5, 0.5, 0.2, 0.2],
                   [0.5, 0.5, 0.3, 0.3]])