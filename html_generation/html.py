import sys
sys.path.append('.')
sys.path.append('..')
from my_python_utils.common_utils import *

import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os

class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        os.makedirs(self.web_dir, exist_ok=True)
        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, width=400, im_names=None, no_names=False):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for i, im in enumerate(ims):
                    if im_names is None:
                        im_name = im.split('/')[-1]
                    else:
                        im_name = im_names[i]
                    with td(style="word-wrap: break-word; width:%dpx" % width, halign="center", valign="top"):
                        with p():
                            if not no_names:
                                p(im_name)
                            with a(href=im):
                                img(style="width:%dpx" % width, src=im)


    def save(self):
        """save the current content to the HMTL file"""
        html_file = '{}/{}.html'.format(self.web_dir, self.title)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def make_html_at_random(dataset_dir, output_dir, n_samples_per_page=300, pages=10):
    # we need to first list all samples
    dataset_name = dataset_dir.split('/')[-1]

    folders = listdir(dataset_dir + '/train', prepend_folder=True)
    print("Listing all dirs to find all images, as then we will sample them at random. This can take a while.")
    all_images = []
    for img_f in tqdm(folders):
        images = sorted([(img_f, k) for k in listdir(img_f) if k.endswith('.png') or k.endswith('.jpg')])
        all_images.extend(images)

    random.shuffle(all_images)
    web_dir = '{}/{}'.format(output_dir, dataset_name)

    for page, actual_images in enumerate(chunk_list(all_images, n_samples_per_page)):
        html = HTML(web_dir, title='samples_at_random_' + str(page).zfill(6))
        html.add_header('Samples at random' + dataset_dir.split('/')[-1])

        for chunk_imgs in chunk_list(actual_images, 30):
            imgs = ['../../../' + '/'.join(img_f.split('/')[-4:]) + '/' + k for img_f, k in chunk_imgs]
            html.add_images(imgs, width=128)
        html.save()
        if page >= pages:
            return


def make_html_dirs_sorted(dataset_dir, n_folders=5):
    dataset_name = dataset_dir.split('/')[-1]
    for img_f in sorted(listdir(dataset_dir + '/train', prepend_folder=True))[:n_folders]:
        web_dir = '{}/{}'.format(output_dir, dataset_name)

        html = HTML(web_dir, title=img_f.split('/')[-1])
        html.add_header('Sorted samples ' + dataset_dir.split('/')[-1])
        images = sorted([k for k in listdir(img_f) if k.endswith('.png') or k.endswith('.jpg')])
        for chunk_imgs in chunk_list(images, 30):
            imgs = ['../../../' + '/'.join(img_f.split('/')[-4:]) + '/' + k for k in chunk_imgs]
            html.add_images(imgs, width=128)
        html.save()

if __name__ == '__main__':
    # output_dir = 'release_datasets_samples_html'
    output_dir = 'python_datasets_samples_html'

    all_dataset_dirs = ['/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/datasets_python_generated']

    samples_at_random = False

    output_dir += '/samples_at_random' if samples_at_random else '/sorted_samples'

    os.makedirs(output_dir, exist_ok=True)
    for datasets_dir in all_dataset_dirs:
        all_datasets_in_dir = listdir(datasets_dir, prepend_folder=True)
        # all_datasets_in_dir = [k for k in listdir(datasets_dir, prepend_folder=True) if 'feature_visualizations_' in k]
        for dataset_dir in tqdm(all_datasets_in_dir):
            if samples_at_random:
                make_html_at_random(dataset_dir, output_dir)
            else:
                make_html_dirs_sorted(dataset_dir)