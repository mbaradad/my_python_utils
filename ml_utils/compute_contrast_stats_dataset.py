import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from my_python_utils.common_utils import *

from my_python_utils.ml_utils.datasets import ImageFilelist
from torch.utils.data.dataloader import DataLoader


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-root', type=str, default='/data/vision/torralba/datasets/imagenet_pytorch/train')
  parser.add_argument('--folder-levels', type=int, default=1, help='Level of folders. 0 corresponds to the directory directly containing images')
  parser.add_argument('--store-parameters', action='store_true')
  parser.add_argument('--parameters-path', type=str, default='/tmp')
  parser.add_argument('--N-samples', type=int, default=-1)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--num-workers', type=int, default=1)

  valid_extensions = ['jpg', 'jpeg', 'png']

  args = parser.parse_args()

  images_to_test = []

  subdirs = get_subdirs(args.dataset_root, args.folder_levels)
  images_to_test = []
  print("Listing all subdirs {}".format(len(subdirs)))
  for subdir in tqdm(subdirs):
    images_to_test.extend([k for k in listdir(subdir, prepend_folder=True) if k.split('.')[-1].lower() in valid_extensions])
  if args.N_samples > 0:
    # sample at random
    if len(images_to_test) < args.N_samples:
      raise Exception("--N_samples {} is bigger than total images in the dataset {}".format(args.N_samples, len(images_to_test)))
    else:
      images_to_test = random.sample(images_to_test, args.N_samples)

  std = np.zeros((3), dtype='float64')
  print("Computing mean and std...")
  # we do it in a single pass with:
  # http://mathcentral.uregina.ca/QQ/database/QQ.09.06/h/murtaza1.html

  dataset = ImageFilelist(images_to_test)
  loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

  # the actual implementation for typical values used, though it's not the actual std but the average per image
  # https://github.com/pytorch/vision/issues/1439

  def compute_and_print_metrics(means, stds):
    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

    print("Computed mean {}".format(mean))
    print("Computed std {}".format(std))

  n_pixels = 0
  pixels_sum = np.zeros((3), dtype='float64')
  pixels_sum_squared = np.zeros((3), dtype='float64')
  means = []
  stds = []
  for i, images in enumerate(tqdm(loader)):
    means.append(torch.mean(images, dim=(-1,-2)))
    stds.append(torch.std(images, dim=(-1,-2)))

    if (i + 1) % 100 == 0:
      compute_and_print_metrics(means, stds)



