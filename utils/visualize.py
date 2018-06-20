from matplotlib import pyplot as plt
import matplotlib.patches as patches

def visulizeBBox(img, bboxes, hold=False):
  if not hold:
    fig, ax = plt.subplots(1)
  else:
    ax = plt.gca()

  ax.clear()
  ax.imshow(img)
  for box in bboxes:
    # Create a Rectangle patch
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

  plt.draw()