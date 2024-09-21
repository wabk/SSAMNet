import numpy as np
from skimage.transform import rotate, resize
from scipy.ndimage import shift
import random

def random_rotate(array, max_angle):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        random_angle = random.randint(-max_angle, max_angle)

        for i in range(array.shape[0]):
            array_out[i] = rotate(array[i], random_angle, preserve_range=True)

        return array_out
    else:
        return array

def random_shift(array, max_shift):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        random_x = random.randint(-max_shift, max_shift)
        random_y = random.randint(-max_shift, max_shift)

        for i in range(array.shape[0]):
            array_out[i] = shift(array[i], (random_x, random_y))

        return array_out
    else:
        return array

def random_flip(array):
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)
        
        for i in range(array.shape[0]):
            array_out[i] = np.fliplr(array[i])

        return array_out
    else:
        return array

def random_resize(array, min_factor, max_factor, INPUT_DIM=224):
    if random.random() > 0.5:
        random_factor = random.uniform(min_factor, max_factor)
        h, w = int(array.shape[1] * random_factor), int(array.shape[2] * random_factor)

        array_out = np.zeros((array.shape[0], INPUT_DIM, INPUT_DIM))

        for i in range(array.shape[0]):
            if h > INPUT_DIM:
                y1 = int(int(h / 2) - (INPUT_DIM / 2))
                y2 = int(int(h / 2) + (INPUT_DIM / 2))
            else:
                y1 = 0
                y2 = h

            if w > INPUT_DIM:
                x1 = int(int(w / 2) - (INPUT_DIM / 2))
                x2 = int(int(w / 2) + (INPUT_DIM / 2))
            else:
                x1 = 0
                x2 = w

            sl = array[i, y1:y2, x1:x2]
            array_out[i] = resize(sl, (INPUT_DIM, INPUT_DIM), anti_aliasing=True, preserve_range=True)
        return array_out

    else:
        return array

def center_resize(array, INPUT_DIM=224):

    h, w = int(array.shape[1]), int(array.shape[2])

    array_out = np.zeros((array.shape[0], INPUT_DIM, INPUT_DIM))

    for i in range(array.shape[0]):
        if h > INPUT_DIM:
            y1 = int(int(h / 2) - (INPUT_DIM / 2))
            y2 = int(int(h / 2) + (INPUT_DIM / 2))
        else:
            y1 = 0
            y2 = h

        if w > INPUT_DIM:
            x1 = int(int(w / 2) - (INPUT_DIM / 2))
            x2 = int(int(w / 2) + (INPUT_DIM / 2))
        else:
            x1 = 0
            x2 = w

        sl = array[i, y1:y2, x1:x2]
        array_out[i] = resize(sl, (INPUT_DIM, INPUT_DIM), anti_aliasing=True, preserve_range=True)

    return array_out


