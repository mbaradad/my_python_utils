from my_python_utils.common_utils import *

def plot(im):
  im, min_range, max_range = preprocess_im_to_plot(im)

  if len(im.shape) == 3:
    plt.imshow(im.transpose((1, 2, 0)))
    plt.show()
  if len(im.shape) == 2:
    plt.imshow(im, cmap='gray')
    plt.show()



