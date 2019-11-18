import logging
import glob
import os
from os.path import join, splitext, basename
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from utils.data_augmentor import FlipLeftRight, FlipTopBottom, Brightness, Contrast


logger = logging.getLogger(__name__)


def get_shape_input():
    return (None, 128, 128, 1)


def get_shape_input_flatten():
    return (None, 128 * 128 * 1)


def get_shape_label():
    return (None,)


def num_classes():
    return 2


def process_after_load(img):
    #img = img.resize((IMG_SHAPE, IMG_SHAPE))
    img = ImageOps.autocontrast(img, 1.0)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    # img = img.filter(ImageFilter.EMBOSS)

    img = np.array(img, np.float32)

    # other preprocessing will be done at class level
    # img = np.array(img, np.float32) / 255.0 * 2 - 1

    return img


def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img = process_after_load(img)

    return img


def preprocess_image(arr, normalize=True, centered=True):
    if normalize:
        arr = arr.astype(np.float32) / 255
    if centered:
        arr = arr.astype(np.float32) * 2. - 1.

    return arr


def _get_adapted_dataset(split, label=None, centered=False, flatten=False):
    # copied from yasumura-san's code
    path = '/storage/dataset/koito_head_light'

    train_list = sorted(glob.glob(os.path.join(path, 'train', '*.png')))
    train_images = []
    for image_path in train_list:
        img = load_image(image_path)
        train_images.append(img)

    sh = get_shape_input()

    train_images = np.array(train_images).reshape(-1, sh[1], sh[2], sh[3])
    train_labels = np.ones(len(train_list))

    test_normal_list = sorted(
        glob.glob(join(path, 'test/normal', '*.png')))
    test_anomaly_list = sorted(
        glob.glob(join(path, 'test/anomaly', '*.png')))
    test_images = []
    for image_path in test_normal_list + test_anomaly_list:
        img = load_image(image_path)
        test_images.append(img)

    test_images = np.array(test_images).reshape(-1, sh[1], sh[2], sh[3])
    test_labels = np.concatenate((np.ones(len(test_normal_list)),
                                  np.zeros(len(test_anomaly_list))),
                                 axis=0)

    if split == 'train':
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)


class DataGenerator:
    def __init__(self,
                 split="train",
                 batch=24,
                 anomaly_class=3,  # unused
                 normalize=True,
                 centered=True,
                 shuffle=True):
        self.split = split
        self.batch = batch
        self.anomaly_class = anomaly_class
        self.shuffle = shuffle
        self.logger = logging.getLogger(name="DataGenerator Class")
        self.normalize = normalize
        self.centered = centered

        self.logger.info('Checking data directories.')

        path = '/storage/jira/anomaly/dataset/'
        sh = get_shape_input()

        if split == "train":
            file_list = sorted(glob.glob(join(path, 'train', '*.png')))
            self.labels = np.zeros(len(file_list))
        elif split in ["valid", "test"]:
            normal_list = sorted(glob.glob(join(path, 'test/normal', '*.png')))
            anomaly_list = sorted(glob.glob(join(path, 'test/anomaly', '*.png')))
            file_list = normal_list + anomaly_list
            self.labels = np.concatenate((np.zeros(len(normal_list)),
                                          np.ones(len(anomaly_list))),
                                         axis=0)
        else:
            raise Exception("type not in train,valid,test.")

        self.file_list = file_list
        imgs = [load_image(ip) for ip in self.file_list]
        # process images
        self.images = np.array(imgs).reshape(-1, sh[1], sh[2], sh[3])

        print(self.images.shape)
        print(self.labels.shape)

        self.logger.info('{} images found for {} dataset.'.format(
            len(self.images), self.split))

        self.on_epoch_end()

    def __len__(self):
        """ no. of batches per epoch """
        return int(np.floor(len(self.png_list) / self.batch))

    def __getitem__(self, batch_id):
        """ generate one batch of data """
        from_b = (batch_id) * self.batch
        to_b = (batch_id + 1) * self.batch
        if len(self.indexes) < to_b:
            to = len(self.indexes) - 1
        ba_ids = self.indexes[from_b:to_b]
        inputs, outputs = self._get_data_from_indexes(ba_ids)
        inputs = np.stack(inputs, axis=0)

        if self.split == "train":
            lr_flip = FlipLeftRight()
            tb_flip = FlipTopBottom()
            inputs = [ lr_flip(tb_flip(img)) for img in inputs]

            brightness = Brightness()
            contrast = Contrast()
            inputs = [ brightness(contrast( np.concatenate([img, img, img], axis=2) )) for img in inputs]
        elif self.split not in ["valid", "test"]:
            raise Exception("type not in trein,valid,test.")
            
        inputs = [np.expand_dims(img[:,:,0], axis=2) for img in inputs ]
            
        inputs = np.array([preprocess_image(arr, self.normalize, self.centered) for arr in inputs])
        # inputs, outputs = self._preprocess_data(fl, im, an)

        return inputs, outputs

    def on_epoch_end(self):
        """ do things after keras epoch ending """
        self.indexes = self._get_new_indexes()

    def _get_new_indexes(self):
        indexes = np.arange(len(self.images), dtype=int)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def _get_data_from_indexes(self, ids):
        imgs = [self.images[i] for i in ids]
        lbls = [self.labels[i] for i in ids]
        return imgs, lbls
