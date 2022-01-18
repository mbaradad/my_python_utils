from my_python_utils.common_utils import *

from torchvision.datasets.folder import pil_loader

if __name__ == '__main__':
  parallel = True
  RESOLUTION = 256

  directory = '/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/images_glsl_generated/stylegan2_generated_images'

  all_imgs = find_all_files_recursively(directory, extension='.jpg', prepend_folder=True)
  print("Testing {} images".format(len(all_imgs)))

  def test_one_image(img_file):
    if os.path.exists(img_file):
      # open image, if it fails create again
      img = pil_loader(img_file)
      if not img.size == (RESOLUTION, RESOLUTION):
        print("Failed img file : " + img_file)

  process_in_parallel_or_not(test_one_image, all_imgs, parallel=parallel)