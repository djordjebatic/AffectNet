#!/usr/bin/env python

import keras
import numpy as np
from imgaug import augmenters as iaa
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import image


class CustomDataGenerator(keras.utils.Sequence):
    """
    Takes input of image paths and corresponding labels and generates batches of tensor image data with real-time
    data augmentation. The data will be looped over (in batches).
    """

    def __init__(self, images_paths, labels, batch_size=64, image_dimensions=(224, 224, 3), shuffle=False,
                 augment=False):
        self.images_paths = images_paths  # array of image paths.
        self.labels = labels  # array of labels
        self.batch_size = batch_size  # batch size.
        self.dim = image_dimensions  # image dimensions.
        self.shuffle = shuffle  # shuffle Boolean (default False).
        self.augment = augment  # augment Boolean (default False).
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # select data and load images
        labels = np.array([self.labels[k] for k in indexes])
        images = [image.img_to_array(image.load_img(self.images_paths[k], target_size=(224, 224))) for k in indexes]
        # preprocess and augment data
        if self.augment:
            images = self.augmentor(images)
        images = np.array([preprocess_input(img) for img in images])
        return images, labels

    def augmentor(self, images):
        """Apply data augmentation"""
        sometimes = lambda aug: iaa.Sometimes(1, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of images
                sometimes(iaa.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                ))
            ]
        )
        return seq.augment_images(images)
