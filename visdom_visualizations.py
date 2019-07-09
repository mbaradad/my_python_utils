import plotly.plotly as py
import plotly.graph_objs as go

import visdom
import plotly.plotly as py
import plotly.graph_objs as go

from skvideo.io import FFmpegWriter as VideoWriter
import tempfile

import urllib
import numpy as np

import os
import warnings

from skimage.transform import resize

if not 'NO_VISDOM' in os.environ.keys():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import visdom
    global_vis = visdom.Visdom(port=12890, server='http://vision05', use_incoming_socket=False)

def visdom_heatmap(heatmap, window=None, env=None, vis=None):
  trace = go.Heatmap(z=heatmap)
  data = [trace]
  layout = go.Layout()
  fig = go.Figure(data=data, layout=layout)
  global_vis.plotlyplot(fig, win=window, env=env)


def imshow_vis(im, title=None, window=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.win_exists(title)
  vis.image(im, win=window, opts=opts, env=env)

def visdom_dict(dict_to_plot, title=None, window=None, env=None, vis=None, simplify_floats=True):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  properties = []
  dict_to_plot_sorted_keys = [ k for k in dict_to_plot.keys()]
  dict_to_plot_sorted_keys.sort()
  for k in dict_to_plot_sorted_keys:
    if type(dict_to_plot[k]) is float and simplify_floats:
      properties.append({'type': 'text', 'name': str(k), 'value': '{:.2f}'.format(dict_to_plot[k])})
    else:
      properties.append({'type': 'text', 'name': str(k), 'value': str(dict_to_plot[k])})
  vis.properties(properties, win=window, opts=opts, env=env)

def vidshow_vis(video, title=None, window=None, env=None, vis=None, biggest_dim=None, fps=10):
  if vis is None:
    vis = global_vis
  if video.shape[1] == 1 or video.shape[1] == 3:
    video = video.transpose(0,2,3,1)
  if video.shape[-1] == 1:
    #if one channel, replicate it
    video = np.tile(video,(1,1,1,3))
  opts = dict()
  if not title is None:
    opts['caption'] = title
    opts['fps'] = fps
  if not video.dtype is np.uint8:
    video = np.array(video * 255, dtype='uint8')
  vis.win_exists(title)
  if window is None:
    window = title

  videofile = '/tmp/%s.ogv' % next(tempfile._get_candidate_names())
  writer = VideoWriter(videofile, inputdict={'-r': str(fps)})
  for i in range(video.shape[0]):
    if biggest_dim is None:
      actual_frame = video[i]
    else:
      actual_frame = np.transpose(scale_image_biggest_dim(np.transpose(video[i]), biggest_dim))
    writer.writeFrame(actual_frame)
  writer.close()
  vis.video(videofile=videofile, win=window, opts=opts, env=env)


def scale_image_biggest_dim(im, biggest_dim):
  #if it is a video, resize inside the video
  if im.shape[1] > im.shape[2]:
    scale = im.shape[1] / (biggest_dim + 0.0)
  else:
    scale = im.shape[2] / (biggest_dim + 0.0)
  target_imshape = (int(im.shape[1]/scale), int(im.shape[2]/scale))
  if im.shape[0] == 1:
    im = myimresize(im[0], target_shape=(target_imshape))[None,:,:]
  else:
    im = myimresize(im, target_shape=target_imshape)
  return im

def myimresize(img, target_shape):
  max = img.max(); min = img.min()
  if max > min:
    img = (img - min)/(max - min)
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    img = np.transpose(resize(np.transpose(img, (1,2,0)), target_shape, mode='constant', anti_aliasing=True), (2,0,1))
  else:
    img = resize(img, target_shape, mode='constant', anti_aliasing=True)
  if max > min:
    return (img*(max - min) + min)
  else:
    return img

if __name__ == '__main__':
  heatmap = [[1, 20, 30],
   [20, 1, 60],
   [30, 60, 1]]
  heatmap = np.random.normal(scale=1, size=(36,10))
  visdom_heatmap(np.array(heatmap))