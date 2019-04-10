from PIL import Image
import PIL
import numpy as np

def rgb2gray(image):
    im = Image.fromarray(image.astype('uint8'))
    im = im.convert('L')
    return np.asarray(im)

def scale01(image):
    image = (image-np.min(image))/(np.max(image)-np.min(image)+1e-12)
    return image

def resize(image,width=84,height=84):
    im = Image.fromarray(image.astype('uint8'))
    im = im.resize((height,width),PIL.Image.ANTIALIAS)
    return np.asarray(im)
