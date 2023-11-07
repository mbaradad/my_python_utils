import plotly.express as px
import pandas as pd

def write_and_view_figure(fig, title, output_dir, height=600, width=900, show_on_visdom=False):
  print("Showing figure. If it gets stuck and running on vision cluster, change machine! For example visiongpu40 worked at some point.")
  fig.show()
  kwargs = dict(height=height, width=width)
  output_file = output_dir + "/{}.pdf".format(title)
  fig.write_image(output_file, **kwargs)
  tmp_file_imshow  = "/tmp/{}.png".format(title)
  fig.write_image(tmp_file_imshow , **kwargs)
  if show_on_visdom:
    from my_python_utils.common_utils import imshow, cv2_imread
    figure = cv2_imread(tmp_file_imshow)
    imshow(figure, title=title)
  return output_file, figure

