from my_python_utils.common_utils import *

from torchvision.datasets.folder import pil_loader

if __name__ == '__main__':
  parser = argparse.ArgumentParser('')

  parser.add_argument('--directory', type=str, default='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/final_glsl_datasets/twigl_and_shadertoy_mix_none_384')
  parser.add_argument('--extension', type=str, default='.jpg')
  parser.add_argument('--resolution', type=int, default=384)
  parser.add_argument('--parallel', action='store_true')

  opt = parser.parse_args()

  print("Listing all files recursively!")
  all_imgs = find_all_files_recursively(opt.directory, extension=opt.extension, prepend_folder=True)
  print("Testing {} images".format(len(all_imgs)))

  def test_one_image(img_file):
    if os.path.exists(img_file):
      # open image, if it fails create again
      img = pil_loader(img_file)
      if not img.size == (opt.resolution, opt.resolution):
        print("Failed img file : " + img_file)

  process_in_parallel_or_not(test_one_image, all_imgs, parallel=opt.parallel)