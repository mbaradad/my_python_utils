import plotly.graph_objs as go

import skvideo
skvideo.setFFmpegPath('/usr/bin/')
from skvideo.io import FFmpegWriter, FFmpegReader

import tempfile

import numpy as np
import time
import imageio

import os
import warnings
import cv2

import math

from multiprocessing import Queue, Process
import datetime

import torch

PYCHARM_VISDOM='PYCHARM_RUN'

if not 'NO_VISDOM' in os.environ.keys():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import visdom
    global_vis = visdom.Visdom(port=12890, server='http://visiongpu09', use_incoming_socket=True)

def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list

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

def visdom_dict(dict_to_plot, title=None, window=None, env=PYCHARM_VISDOM, vis=None, simplify_floats=True):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  dict_to_plot_sorted_keys = [k for k in dict_to_plot.keys()]
  dict_to_plot_sorted_keys.sort()
  html = '''<table style="width:100%">'''
  for k in dict_to_plot_sorted_keys:
    v = dict_to_plot[k]
    html += '<tr> <th>{}</th> <th>{}</th> </tr>'.format(k, v)
  html += '</table>'
  vis.text(html, win=window, opts=opts, env=env)

def vidshow_file_vis(videofile, title=None, window=None, env=None, vis=None, fps=10):
  # if it fails, check the ffmpeg version.
  # Depending on the ffmpeg version, sometimes it does not work properly.
  opts = dict()
  if not title is None:
    opts['title'] = title
    opts['caption'] = title
    opts['fps'] = fps
  if vis is None:
    vis = global_vis
  vis.win_exists(title)
  if window is None:
    window = title
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

class MyVideoWriter():
  def __init__(self, file, fps=None, *args, **kwargs):
    if not fps is None:
      kwargs['inputdict'] = {'-r': str(fps)}
    self.video_writer = FFmpegWriter(file, *args, **kwargs)

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

class MyVideoReader():
  def __init__(self, video_file):
    if video_file.endswith('.m4v'):
      self.vid = imageio.get_reader(video_file, format='.mp4')
    else:
      self.vid = imageio.get_reader(video_file)
    self.frame_i = 0

  def get_next_frame(self):
    try:
      return np.array(self.vid.get_next_data().transpose((2,0,1)))
    except:
      return None

  def get_n_frames(self):
    return int(math.floor(self.get_duration_seconds() * self.get_fps()))

  def get_duration_seconds(self):
    return self.vid._meta['duration']

  def get_fps(self):
    return self.vid._meta['fps']

  def position_cursor_frame(self, i):
    assert i < self.get_n_frames()
    self.frame_i = i
    self.vid.set_image_index(self.frame_i)

  def get_frame_i(self, i):
    old_frame_i = self.frame_i
    self.position_cursor_frame(i)
    frame = self.get_next_frame()
    self.position_cursor_frame(old_frame_i)
    return frame

  def is_opened(self):
    return not self.vid.closed

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
  # if it does not work, change the ffmpeg. It was failing using anaconda ffmpeg default video settings,
  # and was switched to the machine ffmpeg.
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

  os.chmod(videofile, 0o777)
  vidshow_file_vis(videofile, title=title, window=window, env=env, vis=vis, fps=fps)
  return videofile

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

def myimresize(img, target_shape, interpolation_mode=cv2.INTER_NEAREST):
  assert interpolation_mode in [cv2.INTER_NEAREST, cv2.INTER_AREA]
  max = img.max(); min = img.min()
  uint_mode = img.dtype == 'uint8'

  assert len(target_shape) == 2, "Passed shape {}. Should only be (height, width)".format(target_shape)
  if max > min and not uint_mode:
    img = (img - min)/(max - min)
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    if img.shape[0] == 3:
      img = np.transpose(cv2.resize(np.transpose(img, (1,2,0)), target_shape[::-1]), (2,0,1))
    else:
      img = cv2.resize(img[0], target_shape[::-1], interpolation=interpolation_mode)[None,:,:]
  else:
    img = cv2.resize(img, target_shape[::-1], interpolation=interpolation_mode)
  if max > min and not uint_mode:
    return (img*(max - min) + min)
  else:
    return img

class ThreadedVisdomPlotter():
  # plot func receives a dict and gets what it needs to plot
  def __init__(self, plot_func, use_threading=True, queue_size=10, force_except=False):
    self.queue = Queue(queue_size)
    self.plot_func = plot_func
    self.use_threading = use_threading
    self.force_except = force_except
    def plot_results_process(queue, plot_func):
        # to avoid wasting time making videos
        while True:
            try:
                if queue.empty():
                    time.sleep(1)
                    if queue.full():
                        print("Plotting queue is full!")
                else:
                    actual_plot_dict = queue.get()
                    env = actual_plot_dict['env']
                    time_put_on_queue = actual_plot_dict.pop('time_put_on_queue')
                    visdom_dict({"queue_put_time": time_put_on_queue}, title=time_put_on_queue, window='params', env=env)
                    print("Plotting...")
                    plot_func(**actual_plot_dict)
                    continue
            except Exception as e:
                if self.force_except:
                  raise e
                print('Plotting failed wiht exception: ')
                print(e)
    if self.use_threading:
      Process(target=plot_results_process, args=[self.queue, self.plot_func]).start()

  def _detach_tensor(self, tensor):
    if tensor.is_cuda:
      tensor = tensor.detach().cpu()
    tensor = np.array(tensor.detach())
    return tensor

  def _detach_dict_or_list_torch(self, list_or_dict):
    # We put things to cpu here to avoid er
    if type(list_or_dict) is dict:
      to_iter = list(list_or_dict.keys())
    elif type(list_or_dict) is list:
      to_iter = list(range(len(list_or_dict)))
    else:
      return list_or_dict
    for k in to_iter:
      if type(list_or_dict[k]) is torch.Tensor:
        list_or_dict[k] = self._detach_tensor(list_or_dict[k])
      else:
        list_or_dict[k] = self._detach_dict_or_list_torch(list_or_dict[k])
    return list_or_dict

  def clear_queue(self):
    while not self.queue.empty():
      self.queue.get()

  def is_queue_full(self):
    if not self.use_threading:
      return False
    else:
      return self.queue.full()

  def n_queue_elements(self):
      if not self.use_threading:
        return 0
      else:
        return self.queue.qsize()

  def put_plot_dict(self, plot_dict):
    try:
      assert type(plot_dict) is dict
      assert 'env' in plot_dict, 'Env to plot not found in plot_dict!'
      plot_dict = self._detach_dict_or_list_torch(plot_dict)
      if self.use_threading:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        plot_dict['time_put_on_queue'] = timestamp
        self.queue.put(plot_dict)
      else:
        self.plot_func(**plot_dict)
    except Exception as e:
      if self.force_except:
        raise e
      print('Putting onto plot queue failed with exception:')
      print(e)



if __name__ == '__main__':
  def plot_func(env):
    time.sleep(1)
    #raise Exception("Test exception!")
  a = ThreadedVisdomPlotter(plot_func,  use_threading=True, queue_size=10, force_except=False)
  for k in range(20):
    a.put_plot_dict({'env': 'env'})
    if a.is_queue_full():
      a.clear_queue()
    print(a.queue.qsize())

  heatmap = [[1, 20, 30],
   [20, 1, 60],
   [30, 60, 1]]
  heatmap = np.random.normal(scale=1, size=(36,10))
  visdom_heatmap(np.array(heatmap))


