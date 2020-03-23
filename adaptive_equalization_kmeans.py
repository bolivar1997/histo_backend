import numpy as np
import openslide

import random
import pickle
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy import misc
import utils
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


def process_tile(slide, x, y, tiles_shape, stddev_threshold = 40):
    img = slide.read_region((x, y), 1, tiles_shape)
    img = np.array(img, dtype=np.float64)

    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    stddev = img.std()

    if stddev > stddev_threshold:
        return shuffle(image_array, random_state=0)[:2000]
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

            pixels = process_tile(slide, x, y, tiles_shape)
            if pixels != []:
                res.append(pixels)


    img_pix = np.concatenate(res, axis=0)
    kmeans = KMeans(n_clusters=colors, random_state=13, verbose=0).fit(img_pix)
    utils.save_model('kmeans_model', kmeans)


    return kmeans



def predict(slide_path, img_shape):
    palette = utils.load_object("pallete.out")
    obj = openslide.OpenSlide(slide_path)

    box_to_check = [(40500, 51500), (29500, 52500), (64500, 63000), (51500, 79000), (49170, 74200), (51500, 74000), (51500, 37650)]
    count = 0

    for x, y in box_to_check:
        count += 1
        img = obj.read_region((x, y), 1, img_shape)

        rgb = img.convert('RGB')
        quantized = rgb.quantize(palette=palette)

        rgb.save('result/%d_source.png' % count)
        quantized.save('result/%d.png' % count)


if __name__ == '__main__':

    predict('data/180458_19.svs', (1000, 1000))