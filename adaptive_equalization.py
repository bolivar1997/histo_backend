import numpy as np
import openslide

import random
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy import misc
import utils

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


def process_tile(slide, x, y, tiles_shape, pixels_to_take, stddev_threshold = 40):
    img = slide.read_region((x, y), 1, tiles_shape)
    img = np.array(img, dtype=np.float64)

    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    stddev = img.std()

    if stddev > stddev_threshold:
        return shuffle(image_array, random_state=0)[:pixels_to_take]
    else:
        return np.array([])


def fit_kmeans(slide_path, colors, stride, tiles_shape):
    ## using stride 10k x 10k take 1k x 1k matrix take random 1k pixels
    # then collect all pixel and run kmeans

    slide = openslide.OpenSlide(slide_path)

    res = []

    for i in range(1, slide.dimensions[0] // stride[0]):
        for j in range(1, slide.dimensions[1] // stride[1]):
            x = i * stride[0]
            y = j * stride[1]

            pixels = process_tile(slide, x, y, tiles_shape, pixels_to_take=2000)
            if pixels != []:
                res.append(pixels)

    img_pix = np.concatenate(res, axis=0)
    kmeans = KMeans(n_clusters=colors, random_state=13, verbose=0).fit(img_pix)
    utils.save_model('kmeans_model', kmeans)


    return kmeans


def predict(slide_path, img_shape):

    kmeans_model = utils.load_model('kmeans_model')

    obj = openslide.OpenSlide(slide_path)


    box_to_check = [(40500, 51500), (64500, 63000), (51500, 79000), (60000, 37000), (39500, 30500), (46200, 73200)]
    count = 0

    for x, y in box_to_check:

        count += 1
        img = obj.read_region((x, y), 0, img_shape)

        misc.imsave('result/%d_source.png' % count, img)

        image_array = np.reshape(img, (img_shape[0] * img_shape[1], 4))


        labels = kmeans_model.predict(image_array)
        recreate_image(kmeans_model.cluster_centers_, labels, img_shape[0], img_shape[1], count)


if __name__ == '__main__':
    kmeans = fit_kmeans('180458_19.svs', 256, (10000, 10000), (1000, 1000))

    predict('180458_19.svs', (1000, 1000))