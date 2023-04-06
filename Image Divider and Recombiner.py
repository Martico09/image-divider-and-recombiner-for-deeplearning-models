import numpy as np
import cv2 as cv

def divider(img):
    size = 128 #set the dimentions for your model like (128,128,3)
    h, w, c = img.shape
    pad_h = size - (h % size)
    pad_w = size - (w % size)
    img = cv.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv.BORDER_CONSTANT, value=0)
    parts = np.split(img, img.shape[0]//size, axis=0)
    parts = [np.split(part, part.shape[1]//size, axis=1) for part in parts]
    parts = [part for sublist in parts for part in sublist]
    return parts

def combiner(parts, original_shape):
    size = 128 #set the dimentions for your model like (128,128,3)
    h, w, c = original_shape
    pad_h = size - (h % size)
    pad_w = size - (w % size)
    h_parts = h // size if h % size == 0 else h // size + 1
    w_parts = w // size if w % size == 0 else w // size + 1
    img = np.zeros((h_parts*size, w_parts*size, c), dtype=np.float64)
    for i, part in enumerate(parts):
        row = i // w_parts
        col = i % w_parts
        img[row*size:(row+1)*size, col*size:(col+1)*size, :] = part[:size, :size, :]
    img = img[:h, :w, :]
    return img
