from scipy.ndimage import (
    gaussian_filter,
    rotate,
    zoom
)
import numpy as np
import cv2


class DataProcessing3D():

    def __init__(self):
        self.applied = []
        pass

    def load_image(self, image):
        """ Takes image shape (Depth, Width, Height, Channels)
        """
        assert image.shape[-1] == 1, "channel must == 1"
        assert len(image.shape) == 4, "must be DHWC"
        self.image = np.array(image[..., 0])
        self.applied = []
        return self

    def transpose(self, transpose=(0, 1, 2, 3)):
        """
        transpose: won't transpose under default setting (0, 1, 2, 3).
        """
        self.image = self.image.transpose(*transpose)
        self.applied.append('tr')
        return self

    def flip(self, direction='ver'):
        if direction == 'ver':
            self.image = np.flip(self.image, 0)
            self.applied.append('fd')
            return self
        if direction == 'depth':
            self.image = np.flip(self.image, 1)
            self.applied.append('fv')
            return self
        if direction == 'hor':
            self.image = np.flip(self.image, 2)
            self.applied.append('fh')
            return self
        raise ValueError('Direction %s unimplemneted.' % direction)

    def guassian_filter(self, sigma=0.5):
        self.image = gaussian_filter(self.image, sigma=sigma)
        self.applied.append('gf')
        return self

    def nl_denoise_3d(self):
        """
        Along the depth axis shall be denoised by 5, 10, 5 settings
        While the other dimension shall be denoised by 5, 5, 5.

        image: 3d array (200, 1024, 200)
        """
        img = []
        for i in range(self.image.shape[0]):
            img.append(cv2.fastNlMeansDenoising(self.image[i, :, :], 5, 10, 5))

        img = np.array(img)
        n_img = []
        for i in range(img.shape[1]):
            n_img.append(cv2.fastNlMeansDenoising(img[:, i, :], 5, 5, 5))
        # reshape it back since the second denoising happend in the middle
        self.image = np.transpose(n_img, (1, 0, 2))
        return self

    def normalize(self):
        self.image = self.image / 255
        self.applied.append('nm')
        return self

    def rotate(self, degree, axes=(0, 2)):
        self.image = rotate(self.image, degree, axes=axes, reshape=False)
        self.applied.append('r')
        return self

    def zoom(self, factor, mode='nearest'):
        self.image = zoom(self.image, factor, mode=mode)
        return self

    def shift(self, shift_factor=(4, 8, 4), mode='constant'):

        def shift_along_dim(image, shifting, axis=0):
            padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
            if shifting == 0:
                return image
            elif shifting > 0:
                _image = image.take(indices=range(shifting, image.shape[axis]), axis=axis)
                padding[axis] = (0, shifting)
            else:
                shifting = -shifting
                _image = image.take(indices=range(0, image.shape[axis] - shifting), axis=axis)
                padding[axis] = (shifting, 0)
            _image = np.pad(_image, padding, mode=mode)
            return _image

        assert len(shift_factor) == len(self.image.shape), "Not same dimension"

        for i, factor in enumerate(shift_factor):
            self.image = shift_along_dim(self.image, factor, axis=i)

        return self

    def random_crop3d(self, shape, zoom=False, total_random=False):
        """ It would crop and pad the image randomly. Also do random shifting as well.
            But all the dimensions will always be bigger than the definition
            shape: crop shape
            zoom: if zoom, it will call zoom() to zoom the image up to the origin
        """

        def _d(img_shape_d, shape_d):

            freedom_1 = img_shape_d - shape_d
            if freedom_1 != 0:
                # Cropping one side on 3 dimensions
                random_1 = np.random.randint(freedom_1)
            else:
                random_1 = 0

            if total_random and not freedom_1 == random_1:
                # Cropping another side on 3 dimensions
                random_1_1 = np.random.randint(freedom_1 - random_1)
            else:
                random_1_1 = freedom_1 - random_1

            if random_1 == 0 or freedom_1 - random_1 - random_1_1 == random_1:
                random_shifting_1 = 0
            else:
                random_shifting_1 = np.random.randint(-random_1, freedom_1 - random_1 - random_1_1)

            return freedom_1, random_1, random_1_1, random_shifting_1

        def _f(image):
            tech = ''
            img_shape = image.shape
            freedom_1 = img_shape[0] - shape[0]
            freedom_2 = img_shape[1] - shape[1]
            freedom_3 = img_shape[2] - shape[2]

            freedom_1, random_1, random_1_1, random_shifting_1 = _d(img_shape[0], shape[0])
            freedom_2, random_2, random_2_1, random_shifting_2 = _d(img_shape[1], shape[1])
            freedom_3, random_3, random_3_1, random_shifting_3 = _d(img_shape[2], shape[2])

            image = image[
                random_1:random_1 + shape[0] + random_1_1,
                random_2:random_2 + shape[1] + random_2_1,
                random_3:random_3 + shape[2] + random_3_1,
                :]
            tech += 'c'
            if zoom:
                factor = [
                    img_shape[0] / image.shape[0],
                    img_shape[1] / image.shape[1],
                    img_shape[2] / image.shape[2],
                ]
                if len(image.shape) == 4:
                    factor.append(1)
                self.image = image
                image = self.zoom(factor).image
                tech += 'z'
            else:
                image = np.pad(image, [
                    [random_1 + random_shifting_1, freedom_1 - random_1 - random_1_1 - random_shifting_1],
                    [random_2 + random_shifting_2, freedom_2 - random_2 - random_2_1 - random_shifting_2],
                    [random_3 + random_shifting_3, freedom_3 - random_3 - random_3_1 - random_shifting_3],
                    [0, 0]
                ], mode='constant')
                tech += 'p'

                self.applied.append(tech)
            return image

        self.image = _f(self.image)

        return self

    def get_image(self):
        return np.expand_dims(self.image, -1)
