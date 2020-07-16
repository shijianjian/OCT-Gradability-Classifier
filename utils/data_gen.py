import numpy as np
import pandas as pd
import os
from tensorflow.python.keras.utils import Sequence

from utils.data_processing_3d import DataProcessing3D

ROOT_COLUMN = 'root'
NAME_COLUMN = 'filename'
LABEL_COLUMN = 'gradability'


class DataSequence3D(Sequence):
    """ Prepare a data sequence for 3D images.
    dataframe: Dataframe. CSV dataframe
    batch_size: integer. input batch size
    choose_only: integer. randomly pick some number of inputs.
    augment_fn: python function. For data augmentation.
    shuffle: bool. To shuffle the inputs or not.
    image_preprocess_fn: function. more preprocessing can be done here by introducing more.
    target_shape: shape-like array. The shape of input for your model.
    input_shape: shape-like array. The shape of original data.
    """
    def __init__(self, dataframe, labelnames, batch_size, data_balancing=True, choose_only=None, augment_fn=None, resize=None, shuffle=True, image_preprocess_fn=None, target_shape=(200, 1024, 200, 1), input_shape=(200, 1024, 200, 1)):
        self.dataframe = dataframe
        self.labelnames = labelnames
        self.batch_size = batch_size
        self.data_balancing = data_balancing
        self.shuffle = shuffle
        self.choose_only = choose_only
        self.augment_fn = augment_fn
        self.image_preprocess_fn = image_preprocess_fn
        self.target_shape = target_shape
        self.input_shape = input_shape
        self.resize = resize

        self.image_processor = DataProcessing3D()
        self.reshuffle_and_choose()
        print(self.labelnames)

    def reshuffle_and_choose(self):
        if self.choose_only is not None:
            _data = self.dataframe.groupby(LABEL_COLUMN) \
                        .apply(lambda x: x.sample(self.choose_only // len(self.labelnames), replace=True))
            # TODO: exclude selected
            if self.choose_only - len(_data) > 0:
                rest_data = self.dataframe.sample(n=self.choose_only - len(_data))
                data = pd.concat([_data, rest_data]).reset_index(drop=True)
            else:
                data = _data
        else:
            data = self.dataframe
        if self.shuffle:
            self.data = data.sample(frac=1).reset_index(drop=True)
        else:
            self.data = data

        self.labelnames = sorted(self.labelnames)

    def __len__(self):
        """ Output the number of total data.
        """
        return len(self.data) // self.batch_size

    def _img_reader(self, image_path, image_preprocess_fn):
        if image_path.endswith(".npy"):
            image = np.expand_dims(np.load(image_path), axis=-1)
        else:
            with open(image_path) as f:
                image = np.fromfile(f, np.uint8)
                if len(image) == 67108864:
                    image = image.reshape((128, 1024, 512))
                    image = np.transpose(np.array([
                        skimage.transform.rescale(image[:, i, :], (200 / 128, 200 / 512), mode='constant') for i in range(image.shape[1])]), (1, 0, 2))
                    image = np.expand_dims(image, axis=-1)
                else:
                    image = image.reshape((
                        self.input_shape[0],
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[3]))

        if image_preprocess_fn is not None:
            image = image_preprocess_fn(image)

        data = self.image_processor.load_image(image)

        if self.resize is not None:
            resize_factor = np.empty(len(self.resize))
            for i in range(len(self.resize)):
                resize_factor[i] = self.resize[i] / self.input_shape[i]
            self.resize_factor = resize_factor
            return data.zoom(self.resize_factor)

        return data

    def _getimages(self, batch_files, image_preprocess_fn=None, augment_fn=None):
        batch_features = np.zeros((self.batch_size, *self.target_shape))
        for i, image_path in enumerate(batch_files):

            data = self._img_reader(image_path, image_preprocess_fn)

            if augment_fn is not None and (np.random.random() > 0.5):
                data = augment_fn(data).normalize()
            else:
                data = data.normalize()

            batch_features[i] = data.get_image()
        return np.array(batch_features)

    def _one_hot(self, labels):
        batch_labels = np.zeros((self.batch_size, len(self.labelnames)))
        for i in range(self.batch_size):
            label = np.zeros(len(self.labelnames))
            label[list(self.labelnames).index(labels[i])] = 1
            batch_labels[i] = label
        return batch_labels

    def _get_raw_batch(self, idx):
        batch_data = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        if (idx + 1) * self.batch_size > len(self.data):
            batch_data = batch_data.append(self.data.iloc[:(idx + 1) * self.batch_size - len(self.data)])
        batch_files = batch_data[ROOT_COLUMN] + os.path.sep + batch_data[NAME_COLUMN]
        batch_labels = batch_data[LABEL_COLUMN]
        return batch_files.values, batch_labels.values

    def __getitem__(self, idx):
        batch_files, batch_labels = self._get_raw_batch(idx)
        batch_X = self._getimages(batch_files, image_preprocess_fn=self.image_preprocess_fn, augment_fn=self.augment_fn)
        batch_y = self._one_hot(batch_labels)
        return batch_X, batch_y

    def on_epoch_end(self):
        self.reshuffle_and_choose()


class TrainingSequence(DataSequence3D):
    """ Prepare a data sequence for 3D images for training. Containing image augmentation methods.
    csv_paths: string, path to the csv file which contains three columns: folder, filename, label
    batch_size: integer. input batch size
    resize: shape-like array. zoom the image before any preprocessing.
    choose_only: integer. randomly pick some number of inputs.
    image_preprocess_fn: function. more preprocessing can be done here by introducing more.
    shuffle: bool. To shuffle the inputs or not.
    crop_shape: shape-like array. randomly crop an area. Do the padding or zooming by crop_zoom_padding_ratio.
    crop_zoom_padding_ratio: float. 0 means zooming all the time. 1 means padding all the time. Zooming is a very expensive operation.
                        Recommand giving around 0.8.
    denoise: bool. Apply Guassian denoise on sigma 0.5 to 1.5 range.
    rotation_range: integer. rotate the 3D image on 3 dimensions randomly.
    flip: bool. Flip the image on 3 dimensions randomly.
    shuffle: bool. To shuffle the inputs or not.
    target_shape: shape-like array. The shape of input for your model.
    input_shape: shape-like array. The shape of original data.
    """
    def __init__(self, csv_paths=None, dataframe=None, batch_size=1, resize=None, choose_only=None, shuffle=True, crop_shape=None, shift_range=None, image_preprocess_fn=None,
                 crop_zoom_padding_ratio=0, denoise=False, rotation_range=None, flip=False, target_shape=(200, 1024, 200, 1), input_shape=(200, 1024, 200, 1)):

        self.csv_paths = csv_paths
        self.crop_shape = crop_shape
        self.shift_range = shift_range
        self.crop_zoom_padding_ratio = crop_zoom_padding_ratio
        self.rotation_range = rotation_range
        self.flip = flip
        self.denoise = denoise

        if dataframe is None:
            if type(csv_paths) == str:
                dataframe = pd.read_csv(csv_paths, header=0, index_col=None)
            elif type(csv_paths) == list:
                dfs = []
                for csv in csv_paths:
                    dfs.append(pd.read_csv(csv, header=0, index_col=None))
                dataframe = pd.concat(dfs).reset_index(drop=True)

        labelnames = sorted(dataframe[LABEL_COLUMN].unique())

        super().__init__(dataframe=dataframe, labelnames=labelnames, batch_size=batch_size, choose_only=choose_only, augment_fn=self._augment_fn,
                         shuffle=shuffle, resize=resize, image_preprocess_fn=image_preprocess_fn, target_shape=target_shape, input_shape=input_shape)

        if self.data_balancing:
            self._balance()

    def _balance(self):
        data = [self.data]
        x = self.data.groupby([LABEL_COLUMN]).count()
        _max = x[NAME_COLUMN].max()
        for label in x.index:
            num = x.get_value(label, NAME_COLUMN)
            if num != _max:
                # SAMPLED FROM THE WHOLE DATASET
                rest_data = self.dataframe.sample(n=_max - num)
                data.append(rest_data)
        if len(data) == 1:
            self.data = data[0]
        else:
            self.data = pd.concat(data).reset_index(drop=True)

    def _augment_fn(self, data):

        if self.crop_shape is not None:
            if np.random.random() > 0.5:
                data.random_crop3d(self.crop_shape, np.random.random() > self.crop_zoom_padding_ratio, False)

        if self.shift_range is not None:
            shift_factor = []
            for i, factor in enumerate(self.shift_range):
                shifting = np.random.randint(-factor, factor + 1)
                shift_factor.append(shifting)
            data.shift(shift_factor=shift_factor)

        if self.denoise:
            data.guassian_filter(np.random.random() * 2)

        # Rotate image on 3 dimensions
        if self.rotation_range is not None:
            degree = int(np.random.random() * self.rotation_range) * 2 - self.rotation_range
            data.rotate(degree, axes=(0, 2))

        # Flip image
        if self.flip:
            if np.random.random() > 0.8:
                data.flip('ver')
            if np.random.random() > 0.8:
                data.flip('hor')

        return data


class TestingSequence(DataSequence3D):
    """ Prepare a data sequence for 3D images for training.
    csv_path: string, path to the csv file which contains three columns: folder, filename, label
    batch_size: integer. input batch size
    resize: shape-like array. zoom the image before any preprocessing.
    choose_only: integer. randomly pick some number of inputs.
    image_preprocess_fn: function. more preprocessing can be done here by introducing more.
    shuffle: bool. To shuffle the inputs or not.
    target_shape: shape-like array. The shape of input for your model.
    input_shape: shape-like array. The shape of original data.
    """
    def __init__(self, csv_paths=None, dataframe=None, batch_size=1, resize=None, choose_only=None, shuffle=True,
                 image_preprocess_fn=None, target_shape=(200, 1024, 200, 1), input_shape=(200, 1024, 200, 1)):

        self.csv_paths = csv_paths

        if dataframe is None:
            if type(csv_paths) == str:
                dataframe = pd.read_csv(csv_paths, header=0, index_col=None)
            elif type(csv_paths) == list:
                dfs = []
                for csv in csv_paths:
                    dfs.append(pd.read_csv(csv, header=0, index_col=None))
                dataframe = pd.concat(dfs).reset_index(drop=True)

        labelnames = sorted(dataframe[LABEL_COLUMN].unique())
        super().__init__(dataframe=dataframe, labelnames=labelnames, batch_size=batch_size, choose_only=choose_only, augment_fn=None,
                         shuffle=shuffle, image_preprocess_fn=image_preprocess_fn, target_shape=target_shape, input_shape=input_shape)

    def get_labels(self):
        labels = self.data[LABEL_COLUMN].values
        batch_labels = np.zeros(len(labels))
        for i in range(len(labels)):
            batch_labels[i] = list(self.labelnames).index(labels[i])
        return batch_labels


class PredictSequence(DataSequence3D):
    """
    Predict sequence returns data only.

    data_path: string. a path-like string of the target folder.
    batch_size: integer. input batch size
    resize: shape-like array. zoom the image before any preprocessing.
    shuffle: bool. To shuffle the inputs or not.
    choose_only: integer. randomly pick some number of inputs.
    image_preprocess_fn: function. more preprocessing can be done here by introducing more.
    target_shape: shape-like array. The shape of input for your model.
    input_shape: shape-like array. The shape of original data.
    """
    def __init__(self, csv_paths=None, data_path=None, dataframe=None, batch_size=1, resize=None, image_preprocess_fn=None, target_shape=(200, 1024, 200, 1), input_shape=(200, 1024, 200, 1)):
        if data_path is not None and dataframe is not None:
            raise ValueError('data_path, dataframe values cannot coexist.')
        if data_path is None and dataframe is None:
            raise ValueError('data_path, dataframe values have to define one.')
        if csv_paths is not None:
            self.csv_paths = csv_paths
            if type(csv_paths) == str:
                dataframe = pd.read_csv(csv_paths, header=0, index_col=None)
            elif type(csv_paths) == list:
                dfs = []
                for csv in csv_paths:
                    dfs.append(pd.read_csv(csv, header=0, index_col=None))
                dataframe = pd.concat(dfs).reset_index(drop=True)
        if data_path is not None:
            dataframe = pd.DataFrame({
                ROOT_COLUMN: [data_path for p in os.listdir(data_path)],
                NAME_COLUMN: [p for p in os.listdir(data_path)],
                LABEL_COLUMN: [np.nan for p in os.listdir(data_path)]
            })
        # Label column should not be required.
        if LABEL_COLUMN in dataframe.columns:
            dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].astype(str)
        else:
            dataframe[LABEL_COLUMN] = np.nan
        labelnames = sorted(dataframe[LABEL_COLUMN].unique())
        self._ground_truth = []
        self.ground_truth = []
        super().__init__(dataframe=dataframe, labelnames=labelnames, batch_size=batch_size, choose_only=None, augment_fn=None,
                         shuffle=False, image_preprocess_fn=image_preprocess_fn, target_shape=target_shape, input_shape=input_shape)

    def __getitem__(self, idx):
        batch_files, gt = super()._get_raw_batch(idx)
        self._ground_truth.extend(gt)
        batch_X = super()._getimages(batch_files, image_preprocess_fn=self.image_preprocess_fn, augment_fn=None)
        return batch_X

    def on_epoch_end(self):
        self.ground_truth = self._ground_truth
        self._ground_truth = []


if __name__ == '__main__':

    import os
    import time

    train_sequence = TrainingSequence(
        csv_path='./sample_data/datasheets/train.csv',
        batch_size=1,
        shuffle=True,
        choose_only=10,
        flip=True,
        target_shape=(200, 1024, 200, 1),
        input_shape=(200, 1024, 200, 1)
    )
