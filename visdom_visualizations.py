import plotly.plotly as py
import plotly.graph_objs as go

import visdom
import plotly.plotly as py
import plotly.graph_objs as go

from skvideo.io import FFmpegWriter
import tempfile

import urllib
import numpy as np

import os
import warnings
import cv2

if not 'NO_VISDOM' in os.environ.keys():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import visdom
    global_vis = visdom.Visdom(port=890, server='http://vision05', use_incoming_socket=True)

def visdom_heatmap(heatmap, window=None, env=None, vis=None):
  trace = go.Heatmap(z=heatmap)
  data = [trace]
  layout = go.Layout()
  fig = go.Figure(data=data, layout=layout)
  global_vis.plotlyplot(fig, win=window, env=env)

def visdom_default_window_title_and_vis(win, title, vis):
  if win is None and title is None:
    win = title = 'None'
  elif win is None:
    win = str(title)
  elif title is None:
    title = str(win)
  if vis is None:
    vis = global_vis
  return win, title, vis

def imshow_vis(im, title=None, win=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opts['title'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.image(im, win=win, opts=opts, env=env)

def visdom_dict(dict_to_plot, title=None, window=None, env='PYCHARN_RUN', vis=None, simplify_floats=True):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  properties = []
  dict_to_plot_sorted_keys = [k for k in dict_to_plot.keys()]
  dict_to_plot_sorted_keys.sort()
  for k in dict_to_plot_sorted_keys:
    if type(dict_to_plot[k]) is float and simplify_floats:
      properties.append({'type': 'text', 'name': str(k), 'value': '{:.2f}'.format(dict_to_plot[k])})
    else:
      properties.append({'type': 'text', 'name': str(k), 'value': str(dict_to_plot[k])})
  vis.properties(properties, win=window, opts=opts, env=env)


def vidshow_file_vis(videofile, title=None, window=None, env=None, vis=None, fps=10):
  opts = dict()
  if not title is None:
    opts['caption'] = title
    opts['fps'] = fps
  if vis is None:
    vis = global_vis
  vis.win_exists(title)
  if window is None:
    window = title
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

class MyVideoWriter():
  def __init__(self, *args, **kwargs):
    self.video_writer = FFmpegWriter(*args, **kwargs)

  def writeFrame(self, im):
    if len(im.shape) == 3 and im.shape[0] == 3:
      transformed_image = im.transpose((1,2,0))
    elif len(im.shape) == 2:
      transformed_image = np.concatenate((im[:,:,None], im[:,:,None], im[:,:,None]), axis=-1)
    else:
      transformed_image = im
    self.video_writer.writeFrame(transformed_image)

  def close(self):
    self.video_writer.close()

# encoded as apple ProRes mov
# ffmpeg -i input.avi -c:v prores_ks -profile:v 3 -c:a pcm_s16le output.mov
# https://video.stackexchange.com/questions/14712/how-to-encode-apple-prores-on-windows-or-linux
def get_video_writer(videofile, fps=10, verbosity=0):
  writer = MyVideoWriter(videofile + '.mov', verbosity=verbosity, inputdict={'-r': str(fps)},
                                                                  outputdict={
                                                                    '-c:v': 'prores_ks',
                                                                    '-profile:v': '3',
                                                                    '-c:a': 'pcm_s16le'})
  return writer

def vidshow_vis(frames, title=None, window=None, env=None, vis=None, biggest_dim=None, fps=10):
  if vis is None:
    vis = global_vis
  if frames.shape[1] == 1 or frames.shape[1] == 3:
    frames = frames.transpose(0, 2, 3, 1)
  if frames.shape[-1] == 1:
    #if one channel, replicate it
    frames = np.tile(frames, (1, 1, 1, 3))
  if not frames.dtype is np.uint8:
    frames = np.array(frames * 255, dtype='uint8')
  videofile = '/tmp/%s.mp4' % next(tempfile._get_candidate_names())
  writer = MyVideoWriter(videofile, inputdict={'-r': str(fps)})
  for i in range(frames.shape[0]):
    if biggest_dim is None:
      actual_frame = frames[i]
    else:
      actual_frame = np.array(np.transpose(scale_image_biggest_dim(np.transpose(frames[i]), biggest_dim)), dtype='uint8')
    writer.writeFrame(actual_frame)
  writer.close()
  vidshow_file_vis(videofile, title=title, window=window, env=env, vis=vis, fps=fps)

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
    if img.shape[0] == 3:
      img = np.transpose(cv2.resize(np.transpose(img, (1,2,0)), target_shape[::-1]), (2,0,1))
    else:
      img = cv2.resize(img[0], target_shape[::-1])[None,:,:]
  else:
    img = cv2.resize(img, target_shape[::-1])
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