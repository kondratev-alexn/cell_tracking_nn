import numpy as np
from keras.layers import *

# generates patches of size (patch_size, patch_size) from img and binary (!) label, where patches are taken from 'border_offset' pixels from border and with 'stride'
# by default, border_offset is 0 and stride = patch_size
def patch_generator(img, label, patch_size, label_percentage_threshold, stride=None, border_offset=0):
    if stride is None:
        stride = patch_size
    top_left_x = range(border_offset, img.shape[1] - border_offset - 2 * patch_size, stride)
    top_left_y = range(border_offset, img.shape[0] - border_offset - 2 * patch_size, stride)
    for x in top_left_x:
        for y in top_left_y:
            label_patch = label[x:x + patch_size, y:y + patch_size]
            avrg = np.average(label_patch)
            if avrg > label_percentage_threshold:
                yield img[x:x + patch_size, y:y + patch_size], label_patch


# normalize to [0,1]
def normalize(img):
    min = img.min()
    max = img.max()
    return (img - min) / (max - min)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


from keras.preprocessing.image import ImageDataGenerator


# training generator that modifies both x and y
def train_generator(x_train, y_train, batch_size=32):
    data_gen_args = dict(
        shear_range=10,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')

    datagen_data = ImageDataGenerator(**data_gen_args)
    datagen_label = ImageDataGenerator(**data_gen_args)

    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    data_generator = datagen_data.flow(x_train, batch_size=batch_size, seed=seed)
    label_generator = datagen_label.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(data_generator, label_generator)
    for x, y in train_generator:
        yield x, y
