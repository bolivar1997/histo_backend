import numpy as np
import openslide

import random

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy import misc
from scipy.spatial.distance import cdist

random.seed(57)




def recreate_image(codebook, labels, w, h, count):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1


    misc.imsave('result/%d.png' % count, image)
    return image



def fit_kmeans(slide_path, colors, stride, tiles_shape):

    ## using stride 10k x 10k take 1k x 1k matrix take random 1k pixels
    # then collect all pixel and run kmeans

    slide = openslide.OpenSlide(slide_path)

    res = []
    count = 0
    for i in range(1, slide.dimensions[0] // stride[0]):
        for j in range(1, slide.dimensions[1] // stride[1]):
            img = slide.read_region((i * stride[0], j * stride[1]), 0, tiles_shape)
            img = np.array(img, dtype=np.float64)

            stddev = img.std()

            if stddev > 40:
                count += 1

                w, h, d = tuple(img.shape)
                image_array = np.reshape(img, (w * h, d))

                res.append(shuffle(image_array, random_state=0)[:2000])

    img_pix = np.concatenate(res, axis=0)
    kmeans = KMeans(n_clusters=colors, random_state=13, verbose=0).fit(img_pix)

    return kmeans


def predict(slide_path, img_shape, kmeans_model):
    obj = openslide.OpenSlide(slide_path)

    count = 0
    for i in range(30, obj.dimensions[0] // img_shape[0]):
        for j in range(30, obj.dimensions[1] // img_shape[1]):

            count += 1
            img = obj.read_region((i * img_shape[0], j * img_shape[1]), 0, img_shape)
            image_array = np.reshape(img, (img_shape[0] * img_shape[1], 4))

            labels = kmeans_model.predict(image_array)
            recreate_image(kmeans_model.cluster_centers_, labels, img_shape[0], img_shape[1], count)


if __name__ == '__main__':
    kmeans = fit_kmeans('180458_19.svs', 256, (10000, 10000), (1000, 1000))

    predict('180458_19.svs', (1000, 1000), kmeans)