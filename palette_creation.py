from glob import glob

import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import utils
import openslide
import re
from time import time



def parse_xml_big_sparse(name, strideX, strideY, tile_shape = (1000, 1000)):

    tree = ET.parse(name + '.xml')
    root = tree.getroot()

    obj = openslide.OpenSlide(name + '.svs')

    res = []

    for coord in root.iter('Coordinate'):
        x = int(coord.get('X').split(',')[0])
        y = int(coord.get('Y').split(',')[0])

        img = obj.read_region((x - tile_shape[0] // 2, y - tile_shape[1] // 2), 1, tile_shape)
        img = np.array(img, dtype=np.float64)[:, :, :3]

        pixels = img[::strideX, ::strideY]
        pixels = np.reshape(pixels, (pixels.shape[0] * pixels.shape[1], 3))
        res.append(pixels)

    res = np.concatenate(res, axis=0)

    return res


def parse_xml_small_condense(name, tile_shape = (20, 20)):

    tree = ET.parse(name + '.xml')
    root = tree.getroot()

    obj = openslide.OpenSlide(name + '.svs')

    res = []

    for coord in root.iter('Coordinate'):
        x = int(re.split("[,.]+", coord.get('X'))[0])
        y = int(re.split("[,.]+", coord.get('Y'))[0])

        img = obj.read_region((x - tile_shape[0] // 2, y - tile_shape[1] // 2), 1, tile_shape)
        img = np.array(img, dtype=np.float64)[:, :, :3]

        pixels = np.reshape(img, (tile_shape[0] * tile_shape[1], 3))
        res.append(pixels)

    res = np.concatenate(res, axis=0)

    return res




if __name__ == '__main__':

    res = []



    for openslide_path in glob('data/*.xml'):
        name = openslide_path.split('.')[0]

        pixels = parse_xml_small_condense(name, (15, 15))
        res.append(pixels)


    img_pix = np.concatenate(res, axis=0)

    print(img_pix.shape)
    rgb = Image.fromarray(img_pix[np.newaxis, :, :3].astype(np.uint8))

    s1 = time()
    img = rgb.quantize(colors = 256, method=1, kmeans=1)

    s2 = time()

    print(s2 - s1)


    utils.save_object(path='pallete2.out', obj=img)
